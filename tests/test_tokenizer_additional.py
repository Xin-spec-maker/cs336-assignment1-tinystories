"""Additional coverage; the supplied assignment tests are left unchanged."""

import json

import pytest

from cs336_basics.bpe import BPETokenizer


def test_initialization_preserves_inputs_and_existing_special_ids():
    vocab = {10: b"a", 30: b"<end>"}
    tokenizer = BPETokenizer(vocab, [], ["<end>", "<new>", "<new>"])
    assert vocab == {10: b"a", 30: b"<end>"}
    assert tokenizer.vocab == {10: b"a", 30: b"<end>", 31: b"<new>"}
    assert tokenizer.special_tokens == ["<end>", "<new>"]


def test_empty_special_token_is_rejected():
    with pytest.raises(ValueError, match="nonempty"):
        BPETokenizer({}, [], [""])


def test_decode_joins_bytes_before_utf8_decoding():
    tokenizer = BPETokenizer({i: bytes([i]) for i in range(256)}, [])
    assert tokenizer.decode([]) == ""
    assert tokenizer.decode(list("中文🙃\x00".encode())) == "中文🙃\x00"
    assert tokenizer.decode([0xFF, 0x61, 0xE4]) == "\ufffda\ufffd"


def test_from_files_native_format(tmp_path):
    vocab = {0: b"a", 1: b"b", 2: b"ab", 3: b"\x00\xff"}
    vocab_path, merges_path = tmp_path / "vocab.json", tmp_path / "merges.json"
    vocab_path.write_text(json.dumps({i: list(token) for i, token in vocab.items()}), encoding="utf-8")
    merges_path.write_text(json.dumps([[[97], [98]]]), encoding="utf-8")
    tokenizer = BPETokenizer.from_files(vocab_path, merges_path, ["<end>"])
    assert tokenizer.vocab == {**vocab, 4: b"<end>"}
    assert tokenizer.merges == [(b"a", b"b")]
    assert tokenizer.decode([2, 3]) == "ab\x00\ufffd"


def test_from_files_gpt2_format():
    from pathlib import Path

    fixtures = Path(__file__).parent / "fixtures"
    tokenizer = BPETokenizer.from_files(fixtures / "gpt2_vocab.json", fixtures / "gpt2_merges.txt")
    assert tokenizer.vocab[0] == b"!"
    assert tokenizer.vocab[220] == b" "
    assert tokenizer.decode([15496, 11, 995, 0]) == "Hello, world!"
    assert tokenizer.merges[0] == (b" ", b"t")


@pytest.fixture(scope="module")
def gpt2_tokenizer():
    from pathlib import Path

    fixtures = Path(__file__).parent / "fixtures"
    return BPETokenizer.from_files(
        fixtures / "gpt2_vocab.json",
        fixtures / "gpt2_merges.txt",
        ["<|endoftext|>", "<|endoftext|><|endoftext|>", "[a+b]", "中文标记"],
    )


@pytest.mark.parametrize("text", [
    "Hello, world! We're testing UTF-8: 中文🙃.\n",
    "a<|endoftext|><|endoftext|>b<|endoftext|>c",
    "x[a+b]y中文标记z",
    "  \n\n\t  hello  \n\nword\n\n ",
    "we're I'll they've isn't it's I'M 'r 'l 'v !!'re",
    "",
])
def test_stream_matches_whole_text_at_every_split(gpt2_tokenizer, text):
    expected = gpt2_tokenizer.encode(text)
    for split in range(len(text) + 1):
        assert list(gpt2_tokenizer.encode_iterable([text[:split], "", text[split:]])) == expected
    assert list(gpt2_tokenizer.encode_iterable(iter(text))) == expected
    assert gpt2_tokenizer.decode(expected) == text


def test_stream_is_lazy(gpt2_tokenizer):
    def source():
        yield "hello world " * 10
        raise AssertionError("Read the next input before producing available tokens")

    stream = gpt2_tokenizer.encode_iterable(source())
    assert next(stream) == gpt2_tokenizer.encode("hello")[0]
    stream.close()


def test_merge_order_and_shifted_byte_ids():
    vocab = {i + 10: bytes([i]) for i in range(256)}
    vocab.update({300: b"bc", 301: b"ab", 302: b"abc"})
    # A longest-vocabulary-match encoder would incorrectly choose "abc".
    tokenizer = BPETokenizer(vocab, [(b"b", b"c"), (b"a", b"b")])
    assert tokenizer.encode("abc") == [107, 300]
    assert tokenizer.encode("abc abc") == [107, 300, 42, 107, 300]


def test_large_input_item_and_split_special_token(gpt2_tokenizer):
    text = "hello " * 1364 + "<|endoftext|><|endoftext|>中文\n" + "world " * 2000
    assert list(gpt2_tokenizer.encode_iterable([text])) == gpt2_tokenizer.encode(text)


def test_whitespace_before_incomplete_special_token(gpt2_tokenizer):
    tokenizer = BPETokenizer(gpt2_tokenizer.vocab, gpt2_tokenizer.merges, ["[x]"])
    assert list(tokenizer.encode_iterable(["\n\n", "[x", "]"])) == tokenizer.encode("\n\n[x]")


@pytest.mark.parametrize("special_tokens", [None, ["a", "ab", "abc", "[x]"]])
def test_random_chunk_boundaries(gpt2_tokenizer, special_tokens):
    import random

    rng = random.Random(336)
    tokenizer = BPETokenizer(gpt2_tokenizer.vocab, gpt2_tokenizer.merges, special_tokens)
    fragments = ["a", "ab", "abc", "[x]", " ", "\t", "\n", "\r", "'", "r", "e", "ll", "ve", "!", "中", "🙃"]
    for _ in range(100):
        text = "".join(rng.choices(fragments, k=40))
        cuts = sorted({0, len(text), *[rng.randrange(len(text) + 1) for _ in range(20)]})
        chunks = (text[left:right] for left, right in zip(cuts, cuts[1:]))
        assert list(tokenizer.encode_iterable(chunks)) == tokenizer.encode(text)
        assert tokenizer.decode(tokenizer.encode(text)) == text
