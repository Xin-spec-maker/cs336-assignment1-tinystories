"""Prepare a reproducible TinyStories subset with this project's own BPE.

Only training documents are used to learn merges. Raw text, tokenizer artifacts,
and memory-mappable uint16 arrays stay under the ignored data directory.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .bpe import BPETokenizer, train_bpe

EOS = "<|endoftext|>"
DATASET = "roneneldan/TinyStories"


def sha256_file(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def document_key(document):
    return hashlib.sha256(document.strip().encode("utf-8")).hexdigest()


def read_documents(path):
    return [part.strip() for part in path.read_text(encoding="utf-8").split(EOS) if part.strip()]


def download_prefix(session, url, path, count):
    """Keep the first count complete documents, without downloading the full file."""
    if path.exists():
        if len(read_documents(path)) != count:
            raise ValueError(f"Existing raw file has the wrong document count: {path}")
        return
    pending = b""
    separator = EOS.encode()
    completed = 0
    temporary = path.with_suffix(".part")
    with session.get(url, stream=True, timeout=(30, 90)) as response:
        response.raise_for_status()
        with temporary.open("wb") as output:
            for chunk in response.iter_content(65536):
                pending += chunk
                while separator in pending:
                    document, pending = pending.split(separator, 1)
                    if not document.strip():
                        continue
                    output.write(document + separator)
                    completed += 1
                    if completed == count:
                        break
                if completed == count:
                    break
    if completed != count:
        raise ValueError(f"Only found {completed}/{count} complete documents in {url}")
    temporary.replace(path)


def encode_documents(tokenizer, documents, path):
    start = time.perf_counter()
    count = 0
    raw_path = path.with_suffix(".tokens.tmp")
    with raw_path.open("wb") as output:
        for index, document in enumerate(documents, 1):
            text = document + EOS + "\n"
            ids = tokenizer.encode(text)
            if index <= 3 and tokenizer.decode(ids) != text:
                raise ValueError("Tokenizer roundtrip failed on real source text")
            np.asarray(ids, dtype=np.uint16).tofile(output)
            count += len(ids)
            if index % 2000 == 0:
                print(f"{path.name}: encoded {index}/{len(documents)} stories, {count:,} tokens", flush=True)
    raw = np.memmap(raw_path, mode="r", dtype=np.uint16, shape=(count,))
    result = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint16, shape=(count,))
    for offset in range(0, count, 1_000_000):
        result[offset:offset + 1_000_000] = raw[offset:offset + 1_000_000]
    result.flush()
    del result, raw
    raw_path.unlink()
    return {"documents": len(documents), "tokens": count, "sha256": sha256_file(path),
            "encoding_seconds": time.perf_counter() - start}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/tinystories-core"))
    parser.add_argument("--train-documents", type=int, default=20000)
    parser.add_argument("--valid-documents", type=int, default=1000)
    parser.add_argument("--bpe-documents", type=int, default=2000)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--revision", default=None, help="Hugging Face dataset commit; resolves main if omitted")
    args = parser.parse_args(argv)
    if not 257 <= args.vocab_size <= 65536:
        parser.error("uint16 vocabulary size must be between 257 and 65536")
    if min(args.train_documents, args.valid_documents, args.bpe_documents) <= 0:
        parser.error("document counts must be positive")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        parser.error("Prepared output already exists; use another directory to avoid replacing training data")
    session = requests.Session()
    retries = Retry(total=4, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    revision = args.revision
    if revision is None:
        response = session.get(f"https://huggingface.co/api/datasets/{DATASET}", timeout=30)
        response.raise_for_status()
        revision = response.json()["sha"]
    raw = output / "raw"
    raw.mkdir(exist_ok=True)
    source = {}
    for split, count in (("train", args.train_documents), ("valid", args.valid_documents)):
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/TinyStoriesV2-GPT4-{split}.txt"
        path = raw / f"{split}-{count}.txt"
        print(f"Downloading {count} complete {split} stories from dataset revision {revision}", flush=True)
        download_prefix(session, url, path, count)
        source[split] = {"url": url, "requested_documents": count, "bytes": path.stat().st_size,
                         "sha256": sha256_file(path)}
    train = read_documents(raw / f"train-{args.train_documents}.txt")
    candidates = read_documents(raw / f"valid-{args.valid_documents}.txt")
    train_keys = {document_key(document) for document in train}
    valid = [document for document in candidates if document_key(document) not in train_keys]
    if not valid:
        raise ValueError("Validation split is empty after overlap removal")
    bpe_count = min(args.bpe_documents, len(train))
    bpe_text = "".join(document + EOS + "\n" for document in train[:bpe_count])
    print(f"Training our BPE on {bpe_count} training stories ({len(bpe_text.encode()):,} bytes)", flush=True)
    start = time.perf_counter()
    vocab, merges = train_bpe(bpe_text, args.vocab_size, [EOS])
    bpe_seconds = time.perf_counter() - start
    vocab_path, merges_path = output / "vocab.json", output / "merges.json"
    vocab_path.write_text(json.dumps({i: list(value) for i, value in vocab.items()}), encoding="utf-8")
    merges_path.write_text(json.dumps([[list(a), list(b)] for a, b in merges]), encoding="utf-8")
    tokenizer = BPETokenizer.from_files(vocab_path, merges_path, [EOS])
    manifest = {
        "dataset": DATASET, "revision": revision, "selection": "first complete documents of each official split",
        "text_normalization": "strip document-edge whitespace, append EOS and newline",
        "source": source, "validation_exact_train_overlaps_removed": len(candidates) - len(valid),
        "bpe_training_documents": bpe_count, "bpe_training_bytes": len(bpe_text.encode()),
        "bpe_seconds": bpe_seconds, "vocab_size": len(vocab), "dtype": "uint16",
        "eos_token": EOS, "eos_token_id": tokenizer.encode(EOS)[0],
        "tokenizer_sha256": {"vocab.json": sha256_file(vocab_path), "merges.json": sha256_file(merges_path)},
    }
    for split, documents in (("train", train), ("valid", valid)):
        manifest[split] = encode_documents(tokenizer, documents, output / f"{split}.npy")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
