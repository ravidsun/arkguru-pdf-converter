"""Query rewriting: search several phrasings of one question, then fuse.

A single embedding of the user's exact words assumes they used the corpus's
vocabulary. In this corpus they very often have not. ``chunks.ts`` is
``to_tsvector('english', text)``, which keeps diacritics in the lexeme, so the
accented corpus spelling and a plain ASCII query are simply different tokens.
Measured over 109,163 chunks: the accented form of "Surya" matches 16,432 chunks
and plain "Surya" matches 230, a 71x gap, while "Mrityubhaga" matches 0 against
183 for the accented form. Sanskrit-versus-English naming ("bhava" / "house")
compounds it.

Rewriters produce variants; the caller searches each and fuses the ranked runs
with ``common.rrf.reciprocal_rank_fusion``. Nothing here needs a network by
default:

  * ``NoopRewriter``    -- returns the question unchanged (the default)
  * ``LexiconRewriter`` -- substitutes known term variants from
    ``golden/domain_lexicon.json``, offline and deterministic
  * ``OllamaRewriter``  -- paraphrases with the local model, used only when
    Ollama is reachable, so inference stays on the machine

Select one with ``phase3.retrieval.rewriter`` in config/config.yaml.
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Optional, Protocol, Sequence

log = logging.getLogger("phase3.rewrite")

_LEXICON_DEFAULT = Path(__file__).parent / "golden" / "domain_lexicon.json"
_TOKEN = re.compile(r"(\W+)")


def _fold(s: str) -> str:
    """Drop diacritics so an accented corpus term and its ASCII form compare equal."""
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c))


class QueryRewriter(Protocol):
    name: str

    def rewrite(self, query: str) -> list[str]:
        """Return the query plus any alternative phrasings. Index 0 is the original."""


class NoopRewriter:
    """Default. Keeps behaviour identical to searching the raw question."""

    name = "noop"

    def rewrite(self, query: str) -> list[str]:
        return [query]


class LexiconRewriter:
    """Swap known domain terms for their other spellings.

    One variant per term group present in the query, so a two-term question
    yields at most ``max_variants`` extra searches rather than a combinatorial
    explosion.
    """

    name = "lexicon"

    def __init__(self, groups: Sequence[Sequence[str]], max_variants: int = 3):
        self.max_variants = max_variants
        self._by_term: dict[str, list[str]] = {}
        for variants in groups:
            allv: list[str] = []
            for v in variants:
                for form in (v, _fold(v)):
                    if form not in allv:
                        allv.append(form)
            for v in allv:
                self._by_term.setdefault(v.lower(), allv)

    @classmethod
    def load(cls, path: Path | str = _LEXICON_DEFAULT,
             max_variants: int = 3) -> "LexiconRewriter":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([g["variants"] for g in data["groups"]],
                   max_variants=max_variants)

    def _matches(self, query: str) -> list[tuple[int, str, list[str]]]:
        """Positions of tokens that belong to a known term group."""
        out = []
        for i, tok in enumerate(_TOKEN.split(query)):
            group = self._by_term.get(tok.lower())
            if group and len(group) > 1:
                out.append((i, tok, group))
        return out

    def rewrite(self, query: str) -> list[str]:
        parts = _TOKEN.split(query)
        matches = self._matches(query)
        if not matches:
            return [query]

        variants = [query]
        # Rewrite every recognised term at once, taking each group's alternatives
        # in declared order. The lexicon lists the accented Sanskrit spelling
        # first, which is the form the corpus overwhelmingly uses, so variant 1
        # is the highest-value rewrite. Cycling by offset from the matched token
        # instead would leave that spelling unreachable: from "Surya", stepping
        # 1..3 through the Sun group never lands on the accented form even though
        # it matches 16,432 chunks against 230 for the ASCII spelling.
        for step in range(self.max_variants):
            rewritten = list(parts)
            changed = False
            for idx, tok, group in matches:
                alts = [g for g in group if g.lower() != tok.lower()]
                if step < len(alts):
                    rewritten[idx] = alts[step]
                    changed = True
            candidate = "".join(rewritten)
            if changed and candidate not in variants:
                variants.append(candidate)
        return variants


class OllamaRewriter:
    """Paraphrase with the local model. Skipped silently when Ollama is absent."""

    name = "ollama"

    _PROMPT = ("Rewrite this search query {n} different ways, each on its own "
               "line, keeping the meaning identical. Prefer alternative names "
               "for technical terms. Output only the rewrites.\n\nQuery: {q}\n")

    def __init__(self, model: str, host: str = "http://localhost:11434",
                 n: int = 2, timeout: int = 20):
        self.model = model
        self.host = host.rstrip("/")
        self.n = n
        self.timeout = timeout

    def rewrite(self, query: str) -> list[str]:
        import urllib.error
        import urllib.request

        body = json.dumps({
            "model": self.model,
            "prompt": self._PROMPT.format(n=self.n, q=query),
            "stream": False,
        }).encode()
        req = urllib.request.Request(f"{self.host}/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                text = json.load(r).get("response", "")
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            log.info("Ollama unreachable (%s); searching the query as written", e)
            return [query]
        extra = [l.strip(" -*\t") for l in text.splitlines() if l.strip()]
        return [query] + extra[:self.n]


def build_rewriter(name: Optional[str], cfg: dict) -> QueryRewriter:
    """Construct a rewriter from a config name. Unknown names fall back to noop."""
    rc = cfg.get("retrieval", {}) if cfg else {}
    if not name or name == "noop":
        return NoopRewriter()
    if name == "lexicon":
        return LexiconRewriter.load(
            rc.get("lexicon_path", _LEXICON_DEFAULT),
            max_variants=int(rc.get("rewriter_variants", 3)))
    if name == "ollama":
        serve = cfg.get("serve", {}) if cfg else {}
        return OllamaRewriter(model=serve.get("model_tag", "domain-slm"),
                              n=int(rc.get("rewriter_variants", 2)))
    log.warning("unknown rewriter %r; using noop", name)
    return NoopRewriter()
