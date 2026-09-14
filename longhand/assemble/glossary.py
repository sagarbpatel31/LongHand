"""Terminology carryover (§5) — no external dependencies.

Three mechanisms, in increasing strength:
- `stt_prompt()` chaining: the tail of the most recent transcript, so per-segment
  ASR has some context (§5.1).
- running glossary `keyterms()`: rare/proper-noun terms accumulated across the
  session, fed as `keyterms_prompt` to later segments; self-reinforcing (§5.2).
- `consistency_pass()`: post-hoc clustering of rare terms by a phonetic key + edit
  distance, normalizing minority variants to the majority form (§5.3). Uses a
  hand-rolled phonetic key + `wer.edit_distance` — metaphone would need a dep.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from ..bench.wer import edit_distance

# A small common-word list so the lowercase-rare-token heuristic doesn't flag
# ordinary words. Not exhaustive — ranking + caps + the consistency-pass gates
# handle the long tail.
_COMMON = frozenset(
    """the be to of and a in that have i it for not on with he as you do at this but his
    by from they we say her she or an will my one all would there their what so up out if
    about who get which go me when make can like time no just him know take people into year
    your good some could them see other than then now look only come its over think also back
    after use two how our work first well way even new want because any these give day most us
    are was were been being has had did does said get got make made through toward onto with
    into over under between while during before after above below here there where when very
    much many more less each every both few such same other another around across along
    state states thing things place places part parts kind number numbers""".split()
)

_VOWELS = frozenset("aeiou")
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")


@dataclass(frozen=True)
class GlossaryConfig:
    max_terms: int = 100
    max_chars: int = 8000
    tail_chars: int = 200
    situational: str = "Technical dictation; keep terminology consistent."


def _norm(token: str) -> str:
    return re.sub(r"[^a-z0-9]", "", token.lower())


def extract_terms(text: str) -> list[str]:
    """Candidate rare/proper-noun terms. Recall-heavy; ranking/caps prune noise."""
    tokens = _TOKEN_RE.findall(text)
    out: list[str] = []
    seen: set[str] = set()
    for i, tok in enumerate(tokens):
        core = _norm(tok)
        if not core or core in seen:
            continue
        capitalized = i > 0 and re.fullmatch(r"[A-Z][A-Za-z0-9]+", tok) is not None
        lowercase_rare = len(core) >= 6 and core.isalpha() and core not in _COMMON
        digit_mixed = len(core) >= 3 and any(c.isdigit() for c in core) and any(c.isalpha() for c in core)
        if capitalized or lowercase_rare or digit_mixed:
            out.append(tok)
            seen.add(core)
    return out


@dataclass
class Glossary:
    config: GlossaryConfig = field(default_factory=GlossaryConfig)

    def __post_init__(self) -> None:
        self._counts: Counter[str] = Counter()  # keyed by normalized core
        self._display: dict[str, str] = {}  # core -> best surface form
        self._tail: str = ""

    def observe(self, text: str) -> None:
        if not text:
            return
        for term in extract_terms(text):
            core = _norm(term)
            self._counts[core] += 1
            # Prefer a capitalized surface form for display when we see one.
            best = self._display.get(core)
            if best is None or (term[:1].isupper() and not best[:1].isupper()):
                self._display[core] = term
        self._tail = (self._tail + " " + text).strip()[-2 * self.config.tail_chars :]

    def keyterms(self) -> list[str]:
        ranked = sorted(self._counts, key=lambda c: (-self._counts[c], -len(c), c))
        out: list[str] = []
        chars = 0
        for core in ranked:
            if len(out) >= self.config.max_terms:
                break
            disp = self._display[core]
            if chars + len(disp) > self.config.max_chars:
                continue  # skip (a single huge term shouldn't starve the rest)
            out.append(disp)
            chars += len(disp)
        return out

    def stt_prompt(self) -> str:
        tail = self._tail[-self.config.tail_chars :]
        return f"{self.config.situational} Recent context: {tail}".strip()

    def term_counts(self) -> dict[str, int]:
        return {self._display[c]: n for c, n in self._counts.items()}


# --- consistency pass (§5.3) ------------------------------------------------


def _phonetic_key(core: str) -> str:
    """Coarse phonetic bucket: c/q->k, z->s, x->ks; vowels->a; collapse runs.

    Buckets 'kubernetes' and 'coobernetes' together (metaphone-lite, no dep).
    """
    s = re.sub(r"[^a-z]", "", core.lower())
    s = s.replace("c", "k").replace("q", "k").replace("z", "s").replace("x", "ks")
    out: list[str] = []
    prev = ""
    for ch in s:
        c = "a" if ch in _VOWELS else ch
        if c != prev:
            out.append(c)
        prev = c
    return "".join(out)


def _split(token: str) -> tuple[str, str, str]:
    """(leading non-word, word core, trailing non-word)."""
    lead = re.match(r"^[^\w]*", token).group()
    trail = re.search(r"[^\w]*$", token).group()
    end = len(token) - len(trail) if trail else len(token)
    return lead, token[len(lead) : end], trail


def consistency_pass(
    text: str,
    *,
    max_edit_distance: int = 2,
    min_len: int = 5,
    min_cluster: int = 2,
) -> str:
    """Normalize rare-term variants to the majority form (deterministic)."""
    tokens = text.split()
    counts: Counter[str] = Counter()
    surfaces: dict[str, Counter[str]] = defaultdict(Counter)
    first_idx: dict[str, int] = {}
    for tok in tokens:
        _, word, _ = _split(tok)
        core = _norm(word)
        if not core:
            continue
        counts[core] += 1
        surfaces[core][word] += 1
        first_idx.setdefault(core, len(first_idx))

    candidates = [c for c in counts if len(c) >= min_len and c not in _COMMON]

    # union-find over candidates: union if same phonetic key OR edit distance <= max
    parent = {c: c for c in candidates}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    by_key: dict[str, list[str]] = defaultdict(list)
    for c in candidates:
        by_key[_phonetic_key(c)].append(c)
    for group in by_key.values():
        for other in group[1:]:
            union(group[0], other)
    for i, a in enumerate(candidates):
        for b in candidates[i + 1 :]:
            if abs(len(a) - len(b)) <= max_edit_distance and edit_distance(list(a), list(b)) <= max_edit_distance:
                union(a, b)

    clusters: dict[str, list[str]] = defaultdict(list)
    for c in candidates:
        clusters[find(c)].append(c)

    remap: dict[str, str] = {}  # minority core -> majority surface
    for members in clusters.values():
        total = sum(counts[c] for c in members)
        if total < min_cluster or len(members) < 2:
            continue
        ranked = sorted(members, key=lambda c: (-counts[c], -len(c), first_idx[c]))
        majority, runner_up = ranked[0], ranked[1]
        if counts[majority] == counts[runner_up]:
            continue  # no strict majority -> leave the cluster alone
        majority_surface = surfaces[majority].most_common(1)[0][0]
        for c in members:
            if c != majority:
                remap[c] = majority_surface

    if not remap:
        return text

    rebuilt: list[str] = []
    for tok in tokens:
        lead, word, trail = _split(tok)
        core = _norm(word)
        rebuilt.append(lead + remap[core] + trail if core in remap else tok)
    return " ".join(rebuilt)
