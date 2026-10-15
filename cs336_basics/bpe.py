# 文件路径: cs336_basics/bpe.py

import json
from collections import deque
from collections.abc import Iterable, Iterator
from pathlib import Path

import regex as re


# GPT-2 风格的预分词正则：把语料切成不会跨越的 "pre-token" 片段，
# BPE 训练时只统计片段内部的相邻字节对，不会跨片段合并。
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def get_stats(ids: list[int]) -> dict:
    from collections import Counter
    return Counter(zip(ids, ids[1:]))

def merge(ids: list[int], pair: tuple[int, int], idx: int) -> list[int]:
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

class BPETokenizer:
    """Byte-level BPE tokenizer; vocabulary IDs need not equal byte values."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        # Copy caller-owned data so adding special tokens cannot change it.
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(dict.fromkeys(special_tokens or []))
        if any(not token for token in self.special_tokens):
            raise ValueError("Special tokens must be nonempty strings")
        self._byte_to_id = {token: token_id for token_id, token in self.vocab.items()}
        next_id = max(self.vocab, default=-1) + 1
        self._special_ids = {}
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self._byte_to_id:
                self.vocab[next_id] = token_bytes
                self._byte_to_id[token_bytes] = next_id
                next_id += 1
            self._special_ids[token] = self._byte_to_id[token_bytes]

        self._merge_ranks = {}
        for rank, pair in enumerate(self.merges):
            self._merge_ranks.setdefault(pair, rank)
        self._pretoken_pattern = re.compile(PAT)
        # Longest match wins when two special tokens begin at the same position.
        self._special_pattern = (
            re.compile("|".join(re.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True)))
            if self.special_tokens else None
        )
        self._max_special_length = max(map(len, self.special_tokens), default=0)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | Path,
        merges_filepath: str | Path,
        special_tokens: list[str] | None = None,
    ) -> "BPETokenizer":
        """Load either of two explicit, lossless serialization formats.

        Native JSON: vocab is {id: [byte, ...]}, merges is a list of pairs of
        byte arrays. A trainer's bytes can thus be serialized using list(bytes).
        GPT-2: vocab is {escaped_byte_string: id}, merges is the corresponding
        space-separated text file, optionally starting with a #version header.
        Files are always read as UTF-8, independently of the system locale.
        """
        with open(vocab_filepath, encoding="utf-8") as file:
            stored_vocab = json.load(file)
        with open(merges_filepath, encoding="utf-8") as file:
            stored_merges = file.read()

        if all(isinstance(value, list) for value in stored_vocab.values()):
            vocab = {int(token_id): bytes(value) for token_id, value in stored_vocab.items()}
            merges = [(bytes(left), bytes(right)) for left, right in json.loads(stored_merges)]
        else:
            # GPT-2 maps non-printable bytes to Unicode code points >= 256.
            visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
            byte_decoder = {chr(byte): byte for byte in visible}
            for offset, byte in enumerate(byte for byte in range(256) if byte not in visible):
                byte_decoder[chr(256 + offset)] = byte

            def unescape(token: str) -> bytes:
                return bytes(byte_decoder[character] for character in token)

            vocab = {token_id: unescape(token) for token, token_id in stored_vocab.items()}
            merges = []
            for line in stored_merges.splitlines():
                if not line.strip() or line.startswith("#version:"):
                    continue
                left, right = line.split()
                merges.append((unescape(left), unescape(right)))
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """Encode text without merging across pre-token or special-token boundaries."""
        return list(self._encode_text(text))

    def _pieces(self, text: str) -> Iterator[tuple[int, int, bool]]:
        """Yield (start, end, is_special) spans without building a split list."""
        cursor = 0
        if self._special_pattern is not None:
            for special in self._special_pattern.finditer(text):
                for match in self._pretoken_pattern.finditer(text, cursor, special.start()):
                    yield match.start(), match.end(), False
                yield special.start(), special.end(), True
                cursor = special.end()
        for match in self._pretoken_pattern.finditer(text, cursor):
            yield match.start(), match.end(), False

    def _encode_pretoken(self, text: str) -> Iterator[int]:
        tokens = [bytes([byte]) for byte in text.encode("utf-8")]
        while len(tokens) > 1:
            # Merge rank is training order, not frequency, length, or token ID.
            best_pair = None
            best_rank = len(self.merges)
            for left, right in zip(tokens, tokens[1:]):
                rank = self._merge_ranks.get((left, right), len(self.merges))
                if rank < best_rank:
                    best_pair, best_rank = (left, right), rank
            if best_pair is None:
                break
            merged = []
            index = 0
            while index < len(tokens):
                if index + 1 < len(tokens) and (tokens[index], tokens[index + 1]) == best_pair:
                    merged.append(tokens[index] + tokens[index + 1])
                    index += 2
                else:
                    merged.append(tokens[index])
                    index += 1
            tokens = merged
        for token in tokens:
            yield self._byte_to_id[token]

    def _encode_text(self, text: str) -> Iterator[int]:
        for start, end, is_special in self._pieces(text):
            piece = text[start:end]
            if is_special:
                yield self._special_ids[piece]
            else:
                yield from self._encode_pretoken(piece)

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily encode concatenated input, including tokens split across chunks.

        Retain unfinished pre-tokens and possible special-token prefixes. Working
        memory depends on the longest unfinished pre-token, not total file size;
        an arbitrarily long word or whitespace run can still grow the buffer.
        Input items are str, so a UTF-8 character cannot be split into raw bytes.
        """
        pending = ""
        for chunk in iterable:
            # Limit each scan even when the caller supplies a very large item.
            for offset in range(0, len(chunk), 8192):
                pending += chunk[offset:offset + 8192]
                safe_limit = len(pending) - self._max_special_length
                tail = deque()
                consumed = 0
                for span in self._pieces(pending):
                    # Lookahead must itself be outside the uncertain suffix.
                    # An incomplete special token can change the preceding
                    # whitespace split once it becomes a segment boundary.
                    if span[1] > safe_limit:
                        break
                    tail.append(span)
                    # Keep two trailing spans: appending text may complete a
                    # contraction ("'" + "r" + "e") or change a whitespace match.
                    if len(tail) <= 2:
                        continue
                    start, end, is_special = tail.popleft()
                    piece = pending[start:end]
                    if is_special:
                        yield self._special_ids[piece]
                    else:
                        yield from self._encode_pretoken(piece)
                    consumed = end
                pending = pending[consumed:]
        yield from self._encode_text(pending)

    def decode(self, ids: list[int]) -> str:
        # A UTF-8 character may span multiple tokens: decode only after joining.
        return b"".join(self.vocab[token_id] for token_id in ids).decode("utf-8", errors="replace")

def train_bpe(text: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 1. 初始词表：特殊 token 优先占用最小 ID（0, 1, 2, ...），
    #    256 个单字节 token 跟在后面。字节的值 0~255 刚好可以作为它们的 ID。
    special_tokens = special_tokens or []
    vocab: dict[int, bytes] = {
        i: token.encode("utf-8") for i, token in enumerate(special_tokens)
    }
    vocab.update({i + len(special_tokens): bytes([i]) for i in range(256)})

    # 2. 预分词 + 聚合：把文本切成互不跨越的片段，再把"内容相同"的片段
    #    合并成一条记录，只保留它的出现次数（相同片段不重复统计）。
    #    字节 b 在词表里占 ID = b + len(special_tokens)，所以整段序列要平移。
    #
    #    先用特殊 token 把文本切成若干"普通文本段"：特殊 token 本身要完整保留、
    #    绝不参与合并（否则会学出 b'<|' 这种被拆开的脏 token）。这里用 re.split
    #    按特殊 token 切分，切出的特殊 token 段会被丢弃，不进入 word_counts；
    #    但它们早已在初始 vocab 里占好了 id，所以最终词表依然完整。
    shift = len(special_tokens)
    word_counts: dict[tuple[int, ...], int] = {}

    if special_tokens:
        split_pat = "|".join(re.escape(tok) for tok in special_tokens)
        segments = re.split(split_pat, text)
    else:
        segments = [text]

    for segment in segments:
        for m in re.finditer(PAT, segment):
            key = tuple(b + shift for b in m.group(0).encode("utf-8"))
            word_counts[key] = word_counts.get(key, 0) + 1

    # 3. 建立四个数据结构：
    #    tokens:      片段 id -> 当前 token 序列
    #    counts:      片段 id -> 该片段出现次数
    #    pair_counts: pair -> 全局出现总次数（计分板，O(1) 取最大）
    #    pair_to_ids: pair -> 含有它的片段 id 集合（倒排索引，只处理受影响片段）
    tokens: dict[int, list[int]] = {}
    counts: dict[int, int] = {}
    pair_counts: dict[tuple[int, int], int] = {}
    pair_to_ids: dict[tuple[int, int], set[int]] = {}

    for wid, (seq, cnt) in enumerate(word_counts.items()):
        ids = list(seq)
        tokens[wid] = ids
        counts[wid] = cnt
        for pair in zip(ids, ids[1:]):
            pair_counts[pair] = pair_counts.get(pair, 0) + cnt
            pair_to_ids.setdefault(pair, set()).add(wid)

    # 4. 循环合并：每次在计分板上取次数最大的 pair，
    #    平局时按 token 的字节串字典序取更大者（见 PDF 2.4 节）。
    merges: list[tuple[bytes, bytes]] = []
    num_merges = vocab_size - len(vocab)

    for _ in range(num_merges):
        if not pair_counts:
            break  # 没有可合并的相邻对了
        pair = max(pair_counts, key=lambda p: (pair_counts[p], (vocab[p[0]], vocab[p[1]])))

        new_id = len(vocab)
        merges.append((vocab[pair[0]], vocab[pair[1]]))
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]

        # 只处理"含这个 pair"的片段，其余片段一概不碰
        for wid in list(pair_to_ids.get(pair, ())):
            ids = tokens[wid]
            cnt = counts[wid]

            # 4a. 撤销这个片段的旧贡献：每个旧 pair 撤回 cnt 次，
            #     并从倒排索引里移除该片段
            for old in zip(ids, ids[1:]):
                remaining = pair_counts.get(old, 0) - cnt
                if remaining <= 0:
                    pair_counts.pop(old, None)
                else:
                    pair_counts[old] = remaining
                s = pair_to_ids.get(old)
                if s is not None:
                    s.discard(wid)
                    if not s:
                        pair_to_ids.pop(old, None)

            # 4b. 本地合并
            new_ids = merge(ids, pair, new_id)

            # 4c. 登记新贡献：每个新 pair 加上 cnt 次，并加入倒排索引
            for new_pair in zip(new_ids, new_ids[1:]):
                pair_counts[new_pair] = pair_counts.get(new_pair, 0) + cnt
                pair_to_ids.setdefault(new_pair, set()).add(wid)

            # 4d. 更新片段自身
            tokens[wid] = new_ids
            # 4e. 合并到只剩 1 个 token 的片段不再产生 pair，从记录里移除
            if len(new_ids) < 2:
                counts.pop(wid, None)

    return vocab, merges
