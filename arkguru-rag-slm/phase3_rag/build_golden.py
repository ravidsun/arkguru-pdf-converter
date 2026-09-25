"""Build a stratified golden set with retrieval relevance labels.

Retrieval metrics need to know which chunks *should* come back, so each row
carries ``relevant_chunk_ids`` alongside the question. Rows are grouped into
strata so a change can be judged per reasoning type instead of on one average
that hides regressions.

    python -m phase3_rag.build_golden --out phase3_rag/golden/golden_qa.jsonl

Two question generators:

  * ``gemini``  -- used when ``GEMINI_API_KEY`` is set. Build time only; nothing
    here runs at inference time.
  * ``offline`` -- deterministic fallback with no network and no model. Picks the
    chunk's most distinctive lexemes (rarest by document frequency) and rewrites
    them through ``domain_lexicon.json``, so the question asks about the chunk in
    vocabulary the chunk does not itself use. That is weaker than an authored
    question but it targets the real failure mode: a user typing "Saturn" or
    "Shani" at a corpus that says "Sani" with diacritics.

Both are filtered by the same guards, so a generated set cannot be trivially
easy:

  * a question sharing a verbatim 5-gram with its source chunk is rejected,
    otherwise the lexical leg matches for free and the benchmark measures nothing;
  * ``synthesis`` items must draw on at least two distinct ``source_id`` values.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Protocol, Sequence

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("phase3.build_golden")

STRATA = ("lookup", "synthesis", "numeric", "procedural",
          "negation", "conflict", "transliteration")

_WORD = re.compile(r"[^\W\d_]{3,}", re.UNICODE)
_LEXICON_DEFAULT = Path(__file__).parent / "golden" / "domain_lexicon.json"

# Words that pass a document-frequency band but say nothing about the subject.
# Two kinds: English function words that survive 'english' stemming in this
# mostly-transliterated corpus (df 5-8000, e.g. must=8726), and the publishing
# furniture of scanned magazines (page=2512, book=5257). Domain-plausible nouns
# such as time, day, degree, sunset are deliberately NOT here.
_STOPLIST = frozenset("""
will shall must should would could might may can cannot have has had been were
was are being from these those this that with within without into onto upon unto
also such when then than thus hence therefore however whereas while about above
below after before again further once here there where which what whom whose
very only other others another some many much more most less least same both
each every either neither because although though since until unless
said says say told tell given give gives taken take takes made make makes
page pages book books chapter chapters section sections volume volumes part parts
para paras article articles editor editors issue issues journal ezine magazine
author authors publisher published reply replies email mail yahoo gmail http https
www com net org address phone mobile contact subscription subscribe reader readers
figure table tables note notes example examples following above-mentioned
agent owner view views case cases point points line lines word words name names
second third fourth fifth kindly please thank thanks regards sincerely
""".split())

# Signals used to pick chunks that can actually support a stratum, rather than
# inventing a question the text does not answer. These are POSIX patterns for
# Postgres `~*`, so word boundaries are \y, not \b.
_PROCEDURAL_SQL = (r"\y(first|then|next|afterwards|finally|step[[:space:]]*[0-9]|"
                   r"procedure|calculate|compute|deduct|subtract|divide|multiply)\y")
_NEGATION_SQL = (r"\y(not|never|unless|except|without|cannot|avoid|devoid|"
                 r"neither|nor)\y")


@dataclass
class GoldenRow:
    question: str
    answer: str
    stratum: str
    relevant_chunk_ids: list[str]
    source_ids: list[str] = field(default_factory=list)
    block_type: str = "text"
    query_terms: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass
class Candidate:
    """One or more chunks that together support a question."""
    chunk_ids: list[str]
    texts: list[str]
    sections: list[str]
    source_ids: list[str]
    block_type: str


# --------------------------------------------------------------------------
# guards
# --------------------------------------------------------------------------
def ngrams(text: str, n: int = 5) -> set[tuple[str, ...]]:
    toks = [t.lower() for t in _WORD.findall(text)]
    return {tuple(toks[i:i + n]) for i in range(max(0, len(toks) - n + 1))}


def leaks_verbatim(question: str, sources: Iterable[str], n: int = 5) -> bool:
    """True when the question copies an n-gram straight out of a source chunk."""
    q = ngrams(question, n)
    if not q:
        return False
    return any(q & ngrams(s, n) for s in sources)


def _fold(s: str) -> str:
    """Strip diacritics: 'Sani' with macrons -> plain ASCII."""
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c))


# --------------------------------------------------------------------------
# lexicon
# --------------------------------------------------------------------------
class Lexicon:
    """Maps a surface term to its other spellings, including ASCII folds."""

    def __init__(self, groups: Sequence[dict]):
        self._by_term: dict[str, list[str]] = {}
        self.groups: list[list[str]] = []
        for g in groups:
            variants = list(dict.fromkeys(g["variants"]))
            # an ASCII fold of an accented variant is itself a real user spelling
            folded = [_fold(v) for v in variants]
            allv = list(dict.fromkeys(variants + folded))
            self.groups.append(allv)
            for v in allv:
                self._by_term.setdefault(v.lower(), allv)

    @classmethod
    def load(cls, path: Path | str = _LEXICON_DEFAULT) -> "Lexicon":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["groups"])

    def variants(self, term: str) -> list[str]:
        return self._by_term.get(term.lower(), [])

    def alternative(self, term: str, prefer_ascii: bool = False) -> Optional[str]:
        """A spelling of ``term`` that is not ``term`` itself."""
        opts = [v for v in self.variants(term) if v.lower() != term.lower()]
        if not opts:
            return None
        if prefer_ascii:
            ascii_opts = [o for o in opts if o == _fold(o)]
            if ascii_opts:
                # shortest ASCII form is the most likely thing a user types
                return min(ascii_opts, key=len)
        return opts[0]

    def expand(self, text: str, prefer_ascii: bool = False) -> str:
        """Rewrite known terms in ``text`` to a different spelling."""
        out = []
        for tok in re.split(r"(\W+)", text):
            alt = self.alternative(tok, prefer_ascii=prefer_ascii) if tok else None
            out.append(alt if alt else tok)
        return "".join(out)


# --------------------------------------------------------------------------
# generators
# --------------------------------------------------------------------------
class QuestionGenerator(Protocol):
    name: str

    def generate(self, stratum: str, cand: Candidate) -> Optional[tuple[str, str]]:
        """Return (question, answer) or None to skip this candidate."""


_TEMPLATES = {
    "lookup": "What do the texts say about {terms}?",
    "synthesis": "How do different sources treat {terms}?",
    "numeric": "Which values are tabulated for {terms}?",
    "procedural": "What is the sequence of steps for {terms}?",
    "negation": "When does {terms} not apply?",
    "conflict": "Where do the sources disagree about {terms}?",
    "transliteration": "What is the significance of {terms}?",
}


class OfflineGenerator:
    """TF-IDF terms in a document-frequency middle band, respelled via the lexicon.

    Ranking purely by rarity is wrong for this corpus. It is transliterated
    Sanskrit, so the genuinely topical words are *common* (``lagna`` appears in
    23,026 chunks, ``saturn`` in 17,101) while the rarest tokens are OCR debris,
    author names and email fragments. Terms are therefore taken from a middle
    band of document frequency, must occur at least twice in the chunk unless
    they are domain or heading vocabulary, and are scored ``tf * log(N/df)``.
    """

    name = "offline"

    def __init__(self, lexicon: Lexicon, doc_freq: dict[str, int], n_docs: int,
                 max_terms: int = 4, min_df: int = 5, max_df_frac: float = 0.25,
                 max_domain_terms: int = 1):
        self.lex = lexicon
        self.df = doc_freq
        self.n_docs = max(1, n_docs)
        self.max_terms = max_terms
        self.min_df = min_df
        self.max_df_frac = max_df_frac
        self.max_domain_terms = max_domain_terms

    def _domain_terms(self, text: str) -> list[str]:
        """Lexicon vocabulary actually present in the text, in order."""
        found = []
        for tok in _WORD.findall(text):
            if self.lex.variants(tok) and tok.lower() not in {f.lower() for f in found}:
                found.append(tok)
        return found

    def distinctive_terms(self, texts: Sequence[str],
                          sections: Sequence[str] = ()) -> list[str]:
        import math
        from collections import Counter

        tf: Counter = Counter()
        for t in texts:
            tf.update(w.lower() for w in _WORD.findall(t) if len(w) >= 4)
        if not tf:
            return []

        heading_terms = {w.lower() for s in sections for w in _WORD.findall(s or "")}
        domain_terms = {w.lower() for t in texts for w in self._domain_terms(t)}
        df_ceiling = self.max_df_frac * self.n_docs

        scored: list[tuple[float, str]] = []
        for w, freq in tf.items():
            if w in _STOPLIST:
                continue
            df = self.df.get(w, 0)
            if df < self.min_df or df > df_ceiling:
                continue
            if freq < 2 and w not in heading_terms and w not in domain_terms:
                continue
            scored.append((freq * math.log(self.n_docs / df), w))
        if not scored:
            return []
        scored.sort(key=lambda sw: (-sw[0], sw[1]))

        # One domain term names the subject, which is what makes the query look
        # like something a user would type. The rest are the most distinctive
        # terms available, which is what makes the query identify *this* chunk.
        # Filling every slot with domain vocabulary produced queries like
        # "Mangala, Chandra, Shani, Surya" that match thousands of chunks, since
        # those words are common here (saturn appears in 17,101 of 109,163).
        domain_pick = [w for _, w in scored if w in domain_terms][:self.max_domain_terms]
        specific = [w for _, w in scored if w not in domain_terms]
        picked = domain_pick + specific[:self.max_terms - len(domain_pick)]
        if not picked:
            picked = [w for _, w in scored[:self.max_terms]]
        return picked[:self.max_terms]

    def generate(self, stratum: str, cand: Candidate) -> Optional[tuple[str, str]]:
        terms = self.distinctive_terms(cand.texts, cand.sections)
        if len(terms) < 2:
            return None

        # Respelling is deliberately confined to the transliteration stratum.
        # Applied everywhere it would strip the corpus's own vocabulary out of
        # every question, and since both legs are lexical under the hashing
        # embedder that would pin baseline recall near zero and leave the
        # lexical and pool fixes with nothing measurable to move. Keeping the
        # native surface forms elsewhere isolates the two effects: six strata
        # measure retrieval mechanics, transliteration measures the spelling gap.
        if stratum == "transliteration":
            respelled = [self.lex.alternative(t, prefer_ascii=True) or t
                         for t in terms]
            if respelled == terms:
                return None  # nothing to respell, so it would not test anything
            terms = respelled

        phrase = ", ".join(dict.fromkeys(terms))
        question = _TEMPLATES[stratum].format(terms=phrase)
        answer = _condense(cand.texts)
        return question, answer


class GeminiGenerator:
    """Authors questions with Gemini. Build time only; requires GEMINI_API_KEY."""

    name = "gemini"
    _URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
            "{model}:generateContent")

    _PROMPT = """You are building a retrieval benchmark over a corpus of Vedic and \
Krishnamurti Paddhati (KP) astrology texts.

Write ONE question of type "{stratum}" that is fully answerable from the passage(s) \
below, plus its short answer.

Rules:
- Do NOT reuse any five consecutive words from the passage. Paraphrase.
- Prefer the spelling variant the passage does NOT use (if it says "Surya" with \
diacritics, ask about "Sun"; if it says "bhava", ask about "house").
- "{stratum}" means: {stratum_hint}
- Answer in at most three sentences, grounded only in the passage(s).

Return strict JSON: {{"question": "...", "answer": "..."}}

PASSAGE(S):
{passages}
"""

    _HINTS = {
        "lookup": "a single fact stated in the passage",
        "synthesis": "a point requiring both passages, which come from different books",
        "numeric": "a value, degree, count or period read out of the table",
        "procedural": "the ordered steps or prerequisites of a calculation",
        "negation": "a condition under which the rule does not hold",
        "conflict": "a place where the two sources do not agree",
        "transliteration": "a fact, asked using the English or ASCII name of a Sanskrit term",
    }

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash",
                 timeout: int = 60):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def generate(self, stratum: str, cand: Candidate) -> Optional[tuple[str, str]]:
        import urllib.error
        import urllib.request

        passages = "\n\n---\n\n".join(
            f"[{sid}] {txt[:4000]}" for sid, txt in zip(cand.source_ids, cand.texts))
        prompt = self._PROMPT.format(stratum=stratum,
                                     stratum_hint=self._HINTS[stratum],
                                     passages=passages)
        body = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.4, "maxOutputTokens": 512},
        }).encode()
        req = urllib.request.Request(
            self._URL.format(model=self.model) + f"?key={self.api_key}",
            data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                payload = json.load(r)
        except urllib.error.URLError as e:
            log.warning("gemini call failed (%s); skipping candidate", e)
            return None
        try:
            text = payload["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            log.warning("unexpected gemini payload; skipping candidate")
            return None
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
            return str(obj["question"]).strip(), str(obj["answer"]).strip()
        except (json.JSONDecodeError, KeyError):
            return None


def _condense(texts: Sequence[str], limit: int = 600) -> str:
    joined = " ".join(" ".join(t.split()) for t in texts)
    return joined[:limit]


# --------------------------------------------------------------------------
# candidate sampling
# --------------------------------------------------------------------------
def load_doc_freq(store, cache: Optional[str] = None) -> tuple[dict[str, int], int]:
    """Lexeme -> document count over the whole corpus, plus the document total.

    ``ts_stat`` over 109k rows takes a couple of seconds and yields ~398k
    lexemes, so the result is cached to JSON. Sampling was tried first and gave
    unusable frequencies: a slice ordered by ``chunk_id`` is not representative.
    """
    if cache and Path(cache).exists():
        data = json.loads(Path(cache).read_text(encoding="utf-8"))
        log.info("loaded document frequencies from %s (%d lexemes)",
                 cache, len(data["df"]))
        return {k: int(v) for k, v in data["df"].items()}, int(data["n_docs"])

    with store._connect() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {store.chunks} WHERE is_parent = false")
        n_docs = int(cur.fetchone()[0])
        cur.execute("SELECT word, ndoc FROM ts_stat(%s)",
                    (f"SELECT ts FROM {store.chunks} WHERE is_parent = false",))
        df = {w: int(n) for w, n in cur.fetchall()}
    log.info("computed document frequencies: %d lexemes over %d chunks",
             len(df), n_docs)
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        Path(cache).write_text(json.dumps({"n_docs": n_docs, "df": df}),
                               encoding="utf-8")
    return df, n_docs


def _rows_to_candidate(rows) -> Candidate:
    return Candidate(
        chunk_ids=[r[0] for r in rows],
        texts=[r[1] for r in rows],
        sections=[r[2] or "" for r in rows],
        source_ids=[r[3] or "" for r in rows],
        block_type=rows[0][4] or "text",
    )


def sample_candidates(store, stratum: str, n: int, rng: random.Random,
                      min_chars: int = 400) -> list[Candidate]:
    """Pick chunks that can actually support ``stratum``."""
    base = (f"SELECT chunk_id, text, section, source_id, "
            f"coalesce(meta->>'block_type','text') FROM {store.chunks} "
            f"WHERE is_parent = false ")
    out: list[Candidate] = []

    if stratum == "numeric":
        q = base + ("AND meta->>'block_type' = 'table' AND length(text) > %s "
                    "ORDER BY md5(chunk_id) LIMIT %s")
        args = (min_chars, n)
    elif stratum == "procedural":
        q = base + ("AND coalesce(meta->>'block_type','text') = 'text' "
                    "AND length(text) > %s AND text ~* %s "
                    "ORDER BY md5(chunk_id) LIMIT %s")
        args = (min_chars, _PROCEDURAL_SQL, n)
    elif stratum == "negation":
        q = base + ("AND coalesce(meta->>'block_type','text') = 'text' "
                    "AND length(text) > %s AND text ~* %s "
                    "ORDER BY md5(chunk_id) LIMIT %s")
        args = (min_chars, _NEGATION_SQL, n)
    else:
        q = base + ("AND coalesce(meta->>'block_type','text') = 'text' "
                    "AND length(text) > %s ORDER BY md5(chunk_id || %s) LIMIT %s")
        args = (min_chars, stratum, n)

    with store._connect() as conn, conn.cursor() as cur:
        cur.execute(q, args)
        singles = cur.fetchall()

    if stratum in ("synthesis", "conflict"):
        # pair chunks from different books so the item genuinely needs both
        by_source: dict[str, list] = {}
        for r in singles:
            by_source.setdefault(r[3] or "", []).append(r)
        sources = [s for s in by_source if s]
        rng.shuffle(sources)
        for i in range(0, len(sources) - 1, 2):
            a, b = by_source[sources[i]], by_source[sources[i + 1]]
            if a and b:
                out.append(_rows_to_candidate([a[0], b[0]]))
            if len(out) >= n:
                break
        return out

    return [_rows_to_candidate([r]) for r in singles]


def build(store, generator: QuestionGenerator, per_stratum: int,
          strata: Sequence[str], seed: int = 7) -> list[GoldenRow]:
    rng = random.Random(seed)
    rows: list[GoldenRow] = []
    for stratum in strata:
        # oversample: guards and generators both reject candidates
        cands = sample_candidates(store, stratum, per_stratum * 6, rng)
        kept = 0
        rejected_leak = 0
        for cand in cands:
            if kept >= per_stratum:
                break
            if stratum in ("synthesis", "conflict") and len(set(cand.source_ids)) < 2:
                continue
            made = generator.generate(stratum, cand)
            if not made:
                continue
            question, answer = made
            if leaks_verbatim(question, cand.texts):
                rejected_leak += 1
                continue
            rows.append(GoldenRow(
                question=question, answer=answer, stratum=stratum,
                relevant_chunk_ids=list(cand.chunk_ids),
                source_ids=list(dict.fromkeys(cand.source_ids)),
                block_type=cand.block_type,
                query_terms=_WORD.findall(question)[-6:],
            ))
            kept += 1
        log.info("%-16s kept=%-3d from %-4d candidates (verbatim-rejected %d)",
                 stratum, kept, len(cands), rejected_leak)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the stratified golden set")
    ap.add_argument("--datastore-config", default="config/datastore.yaml")
    ap.add_argument("--out", default="phase3_rag/golden/golden_qa.jsonl")
    ap.add_argument("--per-stratum", type=int, default=25)
    ap.add_argument("--strata", nargs="*", default=list(STRATA))
    ap.add_argument("--generator", choices=("auto", "gemini", "offline"),
                    default="auto")
    ap.add_argument("--gemini-model", default="gemini-1.5-flash")
    ap.add_argument("--lexicon", default=str(_LEXICON_DEFAULT))
    ap.add_argument("--df-cache", default="phase3_rag/golden/doc_freq.json",
                    help="cache for corpus document frequencies")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)

    from common.datastore_config import open_chunk_store
    store = open_chunk_store(a.datastore_config)
    lexicon = Lexicon.load(a.lexicon)

    key = os.environ.get("GEMINI_API_KEY")
    use_gemini = a.generator == "gemini" or (a.generator == "auto" and key)
    if use_gemini and not key:
        log.error("generator=gemini but GEMINI_API_KEY is not set")
        return 2
    if use_gemini:
        gen: QuestionGenerator = GeminiGenerator(key, model=a.gemini_model)
    else:
        log.info("GEMINI_API_KEY not set; using the offline generator")
        df, n_docs = load_doc_freq(store, cache=a.df_cache)
        gen = OfflineGenerator(lexicon, df, n_docs)

    rows = build(store, gen, a.per_stratum, a.strata, seed=a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(r.to_json() for r in rows) + "\n", encoding="utf-8")
    log.info("wrote %d rows to %s (generator=%s)", len(rows), out, gen.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
