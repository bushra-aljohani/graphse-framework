"""Okapi BM25 over concept documents (name plus provenance text)."""
from __future__ import annotations

import math
import re
from collections import Counter

TOKEN = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> list[str]:
    return TOKEN.findall(text.casefold())


class BM25:
    def __init__(self, documents: dict[str, str], k1: float = 1.5, b: float = 0.75):
        self.ids = sorted(documents)
        self.docs = [tokens(documents[i]) for i in self.ids]
        self.k1, self.b = k1, b
        self.avgdl = sum(map(len, self.docs)) / max(len(self.docs), 1)
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    def rank(self, query: str) -> list[tuple[str, float]]:
        q = tokens(query)
        out = []
        for i, (tf, doc) in enumerate(zip(self.tf, self.docs)):
            s = 0.0
            for t in q:
                f = tf.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(doc) / self.avgdl))
            out.append((self.ids[i], s))
        return sorted(out, key=lambda x: (-x[1], x[0]))
