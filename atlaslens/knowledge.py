"""Small retrieval store (RAG) over the lab's written knowledge.

Sources: every .md / .txt file in skills/ and knowledge/. Drop protocols, antibody sheets,
paper notes or past analysis reports into knowledge/ and they become searchable.
Retrieval is TF-IDF, which needs no embedding model or API key and is enough for a few hundred
pages. For a larger corpus, replace `search` with an embedding retriever; the agent only
depends on this function's signature.
"""
from __future__ import annotations
import re
from pathlib import Path
from .common import ROOT

_index = None


def _chunks():
    for folder in ("skills", "knowledge"):
        for path in sorted((ROOT / folder).glob("**/*")):
            if path.suffix.lower() not in (".md", ".txt") or not path.is_file():
                continue
            text = path.read_text(errors="ignore")
            parts = re.split(r"\n(?=#{1,3} )", text)          # split at markdown headings
            for part in parts:
                for i in range(0, len(part), 1500):           # cap chunk length
                    piece = part[i:i + 1500].strip()
                    if len(piece) > 40:
                        yield {"source": str(path.relative_to(ROOT)), "text": piece}


def build_index():
    from sklearn.feature_extraction.text import TfidfVectorizer
    global _index
    docs = list(_chunks())
    if not docs:
        _index = ([], None, None)
        return _index
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    _index = (docs, vec, vec.fit_transform(d["text"] for d in docs))
    return _index


def search(query: str, k: int = 4) -> list[dict]:
    """Return the k most relevant knowledge chunks: [{'source', 'text', 'score'}]."""
    docs, vec, mat = _index or build_index()
    if not docs:
        return []
    scores = (mat @ vec.transform([query]).T).toarray().ravel()
    order = scores.argsort()[::-1][:k]
    return [{**docs[i], "score": round(float(scores[i]), 3)} for i in order if scores[i] > 0]
