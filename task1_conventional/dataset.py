"""Text corpus preparation: Brown corpus (NLTK data repository on GitHub) -> sentences.

Source : https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/brown.zip
         (Brown Corpus, ~1M words of 1960s American English, 500 documents, 15 genres)
Offline fallback: place ``brown.zip`` (same file) in ``data/raw/`` and re-run; nothing else
is needed. ``prepare_dataset`` never fabricates text.

Pipeline
  1. read every document (one sentence per line, tokens "word/TAG");
  2. strip POS tags, detokenise (re-attach punctuation, contractions, quote marks);
  3. keep sentences of 30..120 printable-ASCII characters (limits frame length so that
     a sentence fits in a handful of RBs and the simulation stays CPU-friendly);
  4. remove exact duplicate sentences;
  5. split **by document** (no document contributes to both sets -> no train/test leakage
     from neighbouring sentences), then delete any test sentence that also occurs in train;
  6. the test set is a seeded random sample of N_TEST sentences from the held-out documents.
"""
from __future__ import annotations

import io
import json
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

BROWN_URL = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/brown.zip"
MIN_CHARS, MAX_CHARS = 30, 120
_CLITIC = re.compile(r"^(n't|'s|'re|'ve|'ll|'d|'m|'S|'RE|'VE|'LL|'D|'M|N'T)$")


def _detokenize(tokens: list[str]) -> str:
    out: list[str] = []
    open_quote = True
    glue_next = False
    for tok in tokens:
        if tok in ("``", "''", '"'):
            if tok == "``" or (tok == '"' and open_quote):
                out.append('"')
                glue_next = True
                open_quote = False
            else:
                if out:
                    out[-1] += '"'
                else:
                    out.append('"')
                open_quote = True
            continue
        if glue_next and out:
            out[-1] += tok
            glue_next = False
        elif tok in {",", ".", ";", ":", "!", "?", ")", "%"} or _CLITIC.match(tok):
            if out:
                out[-1] += tok
            else:
                out.append(tok)
        elif tok == "(" or tok == "$":
            out.append(tok)
            glue_next = True
        else:
            out.append(tok)
    return " ".join(out)


def _parse_brown_line(line: str) -> str | None:
    toks = []
    for item in line.split():
        word = item.rsplit("/", 1)[0] if "/" in item else item
        if word:
            toks.append(word)
    if len(toks) < 4:
        return None
    return _detokenize(toks)


def _load_zip(raw_dir: Path) -> bytes:
    zpath = raw_dir / "brown.zip"
    if zpath.exists():
        return zpath.read_bytes()
    try:
        with urllib.request.urlopen(BROWN_URL, timeout=60) as r:
            data = r.read()
    except Exception as e:                                     # documented fallback
        raise RuntimeError(
            f"Could not download the Brown corpus ({e}).\n"
            f"Download {BROWN_URL} manually and save it as {zpath}, then re-run.") from e
    raw_dir.mkdir(parents=True, exist_ok=True)
    zpath.write_bytes(data)
    return data


def prepare_dataset(data_dir="data", n_test: int = 2000, test_doc_fraction: float = 0.15,
                    seed: int = 2024, min_train: int = 20000) -> dict:
    """Build and store train/test sentence files. Returns the statistics dictionary."""
    data_dir = Path(data_dir)
    raw = _load_zip(data_dir / "raw")
    docs: dict[str, list[str]] = {}
    seen: set[str] = set()
    n_raw_lines = 0
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        for name in sorted(zf.namelist()):
            base = name.split("/")[-1]
            if not re.fullmatch(r"c[a-r]\d\d", base):
                continue
            sents = []
            for line in zf.read(name).decode("latin-1").splitlines():
                if not line.strip():
                    continue
                n_raw_lines += 1
                s = _parse_brown_line(line)
                if s is None or not (MIN_CHARS <= len(s) <= MAX_CHARS):
                    continue
                if not all(32 <= ord(c) < 127 for c in s):
                    continue
                if s in seen:
                    continue
                seen.add(s)
                sents.append(s)
            docs[base] = sents
    rng = np.random.default_rng(seed)
    names = sorted(docs)
    rng.shuffle(names)
    n_test_docs = max(1, int(round(test_doc_fraction * len(names))))
    test_docs, train_docs = names[:n_test_docs], names[n_test_docs:]
    train = [s for d in sorted(train_docs) for s in docs[d]]
    train_set = set(train)
    test_pool = [s for d in sorted(test_docs) for s in docs[d] if s not in train_set]
    if len(test_pool) < n_test:
        raise RuntimeError(f"only {len(test_pool)} test sentences available (< {n_test})")
    idx = np.sort(rng.choice(len(test_pool), size=n_test, replace=False))
    test = [test_pool[i] for i in idx]
    if len(train) < min_train:
        raise RuntimeError(f"only {len(train)} training sentences (< {min_train})")
    for sub, lines in (("train", train), ("test", test)):
        (data_dir / sub).mkdir(parents=True, exist_ok=True)
        (data_dir / sub / f"{sub}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    train_chars, test_chars = set("".join(train)), set("".join(test))
    stats = {
        "source": BROWN_URL,
        "corpus": "Brown Corpus (NLTK distribution)",
        "seed": seed,
        "n_documents": len(docs),
        "n_test_documents": len(test_docs),
        "raw_nonempty_lines": n_raw_lines,
        "n_sentences_after_filtering": len(seen),
        "length_filter_chars": [MIN_CHARS, MAX_CHARS],
        "n_train_sentences": len(train),
        "n_test_sentences": len(test),
        "train_test_exact_overlap": len(set(train) & set(test)),
        "train_vocab_chars": len(train_chars),
        "test_vocab_chars": len(test_chars),
        "test_chars_unseen_in_train": sorted(test_chars - train_chars),
        "mean_train_chars_per_sentence": float(np.mean([len(s) for s in train])),
        "mean_test_chars_per_sentence": float(np.mean([len(s) for s in test])),
    }
    (data_dir / "processed").mkdir(parents=True, exist_ok=True)
    (data_dir / "processed" / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def load_split(split: str, data_dir="data") -> list[str]:
    p = Path(data_dir) / split / f"{split}.txt"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing - run notebook 01 or `python main.py --experiment dataset`")
    return p.read_text(encoding="utf-8").splitlines()
