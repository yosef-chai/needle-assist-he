# -*- coding: utf-8 -*-
"""Hebrew phrase matching, with no Home Assistant and no model in sight.

This module answers one question: *does any of these Hebrew phrases appear in
this sentence, and which one?* :mod:`slot_match` builds the phrase lists out of
a real installation's area and entity registries; keeping the matching itself
here means it can be measured against the held-out corpus without a running
Home Assistant.

Why this is not simply ``phrase in utterance``
----------------------------------------------

Hebrew fuses its function words onto the following word. A room called ``סלון``
is said as ``בסלון`` (in the), ``לסלון`` (to the), ``מהסלון`` (from the),
``ובסלון`` (and in the) - the room's own name never appears as a standalone
token. Substring search does find ``סלון`` inside ``בסלון``, but it also finds
``גן`` inside ``מזגן`` and ``אור`` inside ``מאוורר``, and lighting the garden
because someone mentioned the air conditioner is exactly the failure this
module exists to prevent. So the prefixes are matched explicitly and the rest
of the word is fenced off with a letter boundary.

Three tolerances are layered on top, each strictly weaker than the last and
each tried only after the stronger ones find nothing:

``strict``      prefix-aware, both boundaries enforced.
``glued``       drop the left boundary. Speech-to-text drops spaces, and the
                corpus contains ``שואב האבקבגראז'`` - one token, two words.
``despaced``    compare with every space removed, which recovers the opposite
                error: ``בפרוז דור`` for ``בפרוזדור``.
``fuzzy``       bounded edit distance, counting a transposition as one edit.
                Recovers ``במשרדד``, ``במתבח``, ``במסדורן`` - the corpus's
                realistic misspellings.

Ordering them this way keeps precision where it matters: a sentence that
matches something exactly never reaches the passes that guess.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Final, Iterable

# Hebrew points, accents and cantillation. Nobody types these at an assistant,
# but they arrive from copied text and from some keyboards, and an unstripped
# holam turns an exact match into a miss.
_MARKS: Final = re.compile("[֑-ֽֿ-ׇ]")

# Quote-like characters Hebrew uses inside words - ממ"ד, ג'ירפה - and which
# arrive in several Unicode spellings depending on the keyboard.
_PUNCT: Final = str.maketrans({
    '"': "", "'": "", "׳": "", "״": "",   # geresh, gershayim
    "‘": "", "’": "", "“": "", "”": "",
    "-": " ", "–": " ", "_": " ", ",": " ", ".": " ", "!": " ", "?": " ",
})

# The particles Hebrew glues to the front of a noun. Stacking is real and
# common - ו+ב+ה gives "ובהסלון", כש+ב gives "כשבמטבח" - but it is *ordered*,
# and the order is what makes this safe. An earlier version allowed any three
# of these letters in any order, which let "המוגן" (protected) parse as
# ה+מ+ו + "גן" and quietly routed "במרחב המוגן" to the garden. Written as the
# real morphology, conjunction then subordinator then preposition then article,
# that reading does not exist: the article can only come last.
_PREFIX_LETTERS: Final = "ובהלכמש"
_PREFIX_CHAIN: Final = "(?:ו)?(?:כש|ש)?(?:[בלכמ])?(?:ה)?"
_PREFIX_REQUIRED: Final = "(?=[%s])%s" % (_PREFIX_LETTERS, _PREFIX_CHAIN)

# The same chain, anchored, with the preposition captured. ב/ל/מ/כ are the
# locative ones - "in the", "to the", "from the" - and their presence is the
# only evidence a token is naming a *place*. The fuzzy pass leans on it.
_CHAIN_AT_START: Final = re.compile("^(?:ו)?(?:כש|ש)?([בלכמ])?(ה)?")

# A match must not start or end mid-word. Latin letters are in the class too,
# because Home Assistant entities are very often named in English even in a
# Hebrew household ("Living Room Lamp").
_LETTER: Final = "א-תA-Za-z"

_HEB_LETTER: Final = re.compile("[א-ת]")

# Five Hebrew letters change shape at the end of a word. Which shape gets typed
# is not reliable - phone keyboards, speech-to-text and fast typists all produce
# ``בגנ`` for ``בגן`` and ``במםד`` for ``בממד`` - and the two forms are the same
# letter, so folding them loses nothing. Position is not lost either: a final
# form only ever occurs word-finally to begin with.
_FINALS: Final = str.maketrans("ךםןףץ", "כמנפצ")


def normalise(text: str) -> str:
    """Lower-case, de-point, de-punctuate, fold finals, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = _MARKS.sub("", text)
    text = text.translate(_PUNCT).translate(_FINALS)
    return " ".join(text.casefold().split())


def _strip_article(word: str) -> str:
    """Drop a definite ה so ``הסלון`` and ``סלון`` index identically.

    Guarded on length: ``הר`` and ``הן`` are words, not articles, and stripping
    them would leave a single letter that matches almost everywhere.
    """
    return word[1:] if len(word) > 2 and word.startswith("ה") else word


def _word_pattern(word: str, prefix_min: int = 0) -> str:
    """One word, allowing the Hebrew prefix chain in front of it."""
    if not _HEB_LETTER.search(word):
        return re.escape(word)
    chain = _PREFIX_REQUIRED if prefix_min else _PREFIX_CHAIN
    return chain + re.escape(word)


def _body(words: list[str], prefix_min: int = 0) -> str:
    """A whole phrase: word patterns joined by whitespace."""
    head = _word_pattern(words[0], prefix_min)
    return r"\s+".join([head] + [_word_pattern(w) for w in words[1:]])


def _flatten(text: str) -> tuple[str, set[int]]:
    """Text with spaces removed, plus the offsets where words began."""
    flat, starts, at = [], set(), 0
    for word in text.split():
        starts.add(at)
        flat.append(word)
        at += len(word)
    return "".join(flat), starts


def _distance(left: str, right: str, budget: int) -> int:
    """Damerau-Levenshtein, abandoned as soon as it passes ``budget``.

    Transpositions count as one edit because they are what fast typing and
    speech-to-text actually produce - ``מסדורן`` for ``מסדרון``.
    """
    if abs(len(left) - len(right)) > budget:
        return budget + 1
    previous2: list[int] = []
    previous = list(range(len(right) + 1))
    for i, lc in enumerate(left, start=1):
        current = [i] + [0] * len(right)
        for j, rc in enumerate(right, start=1):
            current[j] = min(previous[j] + 1, current[j - 1] + 1,
                             previous[j - 1] + (lc != rc))
            if (i > 1 and j > 1 and lc == right[j - 2] and left[i - 2] == rc):
                current[j] = min(current[j], previous2[j - 2] + 1)
        if min(current) > budget:
            return budget + 1
        previous2, previous = previous, current
    return previous[-1]


def locative(token: str) -> bool:
    """Does this token start with ב/ל/מ/כ - "in", "to", "from", "as"?

    The fuzzy pass needs this. Hebrew is dense enough that one-edit neighbours
    are ordinary words rather than typos: ``חלון`` (window) and ``סלון`` (living
    room) differ by one letter, so do ``כביסה`` (laundry) and ``כניסה``
    (entrance), and both pairs occur in the same sentences. Matching on distance
    alone turned "open the window" into "open the living room". A misspelled
    *room*, though, is nearly always introduced by a locative preposition -
    ``במתבח``, ``בממדד``, ``במשרדד`` - and requiring one throws out the
    ordinary-word collisions while keeping every typo the corpus contains.
    """
    return bool(_CHAIN_AT_START.match(token).group(1))


def _near(window: str, needle: str, budget: int) -> bool:
    """Is ``window`` a misspelling of ``needle``, prefix chain aside?

    Only the whole chain is shaved, never an arbitrary run of letters: shaving
    ``הוו`` off ``הווילון`` leaves ``ילון``, one edit from ``סלון``, and no
    Hebrew reading of the word supports it.
    """
    chain = _CHAIN_AT_START.match(window).end()
    for start in {0, chain}:
        if _distance(window[start:], needle, budget) <= budget:
            return True
    return False


@dataclass(frozen=True)
class Match:
    """One phrase found in an utterance."""

    value: str          # what the phrase resolves to - an area_id, an entity_id
    phrase: str         # the normalised phrase that matched
    start: int          # where in the normalised utterance it began
    end: int            # and where it ended, in the same units
    strength: int       # length of the phrase; longer is more specific
    pass_name: str      # which tolerance level found it
    tier: int = 0       # 0 = the installation's own wording, 1 = our synonyms

    @property
    def rank(self) -> tuple[int, int]:
        """Sort key for "which of these matches is the better answer".

        Length first, because a longer phrase is more specific: an area called
        ``חדר`` must not beat one called ``חדר ילדים``. Tier only breaks ties,
        which is exactly where it is needed - a house with a room named
        ``שירותים`` and another named ``מקלחת`` produces two candidates for the
        word ``שירותים`` of identical length, one of them because that is
        literally the room's name and the other because our synonym table lists
        it under ``washroom``. The house wins.
        """
        return (self.strength, -self.tier)


class PhraseIndex:
    """Maps Hebrew surface phrases onto opaque values.

    Values are whatever the caller wants back - an ``area_id``, an
    ``entity_id``. Several phrases may carry the same value; the longest one
    that matches wins, so an area called ``חדר`` never beats an area called
    ``חדר ילדים`` on a sentence that says both.
    """

    def __init__(self) -> None:
        self._strict: list[tuple[re.Pattern, str, str, int, int]] = []
        self._glued: list[tuple[re.Pattern, str, str, int, int]] = []
        self._despaced: list[tuple[tuple[str, ...], str, str, int, int]] = []
        self._fuzzy: list[tuple[list[str], str, str, int, int]] = []
        self._seen: set[tuple[str, str]] = set()

    def add(self, phrase: str, value: str, tier: int = 0) -> None:
        """Index one surface form. Empties and exact duplicates are ignored."""
        norm = normalise(phrase)
        raw = norm.split()
        words = [_strip_article(w) for w in raw if w]
        if not words or (norm, value) in self._seen:
            return
        self._seen.add((norm, value))

        strength = len(norm)
        self._strict.append((
            re.compile("(?<![%s])%s(?![%s])" % (_LETTER, _body(words), _LETTER)),
            value, norm, strength, tier))

        # The glued pass gives up the left boundary, so it needs a different
        # guard or it matches inside unrelated words: "גן" (garden) sits inside
        # "מזגן" (air conditioner), and lighting the garden because someone
        # mentioned the air conditioner is the exact failure this module exists
        # to prevent. The guard follows from *why* the boundary is dropped - a
        # lost space in front of a prefixed word - so the prefix is made
        # mandatory. "שואב האבקבגראז'" still matches, because the ב of
        # "בגראז'" is right there; "מזגן" does not, because the letter before
        # "גן" is ז, which is not a prefix. The length floor keeps two- and
        # three-letter room names out of a pass that cannot fence them.
        if len("".join(words)) >= 4:
            self._glued.append((
                re.compile("%s(?![%s])" % (_body(words, prefix_min=1), _LETTER)),
                value, norm, strength, tier))

        # Both spellings go in: the article is stripped for regex matching, but
        # the utterance keeps it, and "בכל הח דרים" only rejoins into
        # "בכלהחדרים" - not into "בכלחדרים".
        needles = tuple({"".join(words), "".join(raw)})
        self._despaced.append((needles, value, norm, strength, tier))
        self._fuzzy.append((words, value, norm, strength, tier))

    def extend(self, phrases: Iterable[str], value: str, tier: int = 0) -> None:
        for phrase in phrases:
            self.add(phrase, value, tier)

    def __len__(self) -> int:
        return len(self._strict)

    # -- lookup -------------------------------------------------------------
    def find_occurrences(self, utterance: str, fuzzy: bool = True) -> list[Match]:
        """Every mention, in the order it was said, repeats included.

        "open the blind in the kitchen, then the air conditioning in the kids'
        room, then close the blind in the kitchen" is three mentions of two
        rooms, and a caller pairing rooms with tool calls needs all three.

        Positions are only comparable within one pass, which is all that is
        needed: the first pass that finds anything is the one that is returned.
        """
        text = normalise(utterance)
        for finder in (self._find_strict, self._find_glued, self._find_despaced):
            found = finder(text)
            if found:
                return found
        return self._find_fuzzy(text) if fuzzy else []

    def find_all(self, utterance: str, fuzzy: bool = True) -> list[Match]:
        """Every distinct value mentioned, in the order it was first said."""
        seen: dict[str, Match] = {}
        for hit in self.find_occurrences(utterance, fuzzy=fuzzy):
            seen.setdefault(hit.value, hit)
        return list(seen.values())

    def find(self, utterance: str, fuzzy: bool = True) -> str | None:
        """The single most specific value mentioned, or None."""
        found = self.find_all(utterance, fuzzy=fuzzy)
        if not found:
            return None
        return max(found, key=lambda m: m.rank).value

    # Each pass returns at most one Match per value: the longest phrase that
    # matched it, at the position where that phrase occurred.
    @staticmethod
    def _collect(hits: list[Match]) -> list[Match]:
        best: dict[str, Match] = {}
        for hit in hits:
            current = best.get(hit.value)
            if current is None or hit.rank > current.rank:
                best[hit.value] = hit
        return sorted(best.values(), key=lambda m: m.start)

    def _find_strict(self, text: str) -> list[Match]:
        hits = []
        for pattern, value, phrase, strength, tier in self._strict:
            if (found := pattern.search(text)):
                hits.append(Match(value, phrase, found.start(), found.end(),
                                  strength, "strict", tier))
        return self._collect(hits)

    def _find_glued(self, text: str) -> list[Match]:
        hits = []
        for pattern, value, phrase, strength, tier in self._glued:
            if (found := pattern.search(text)):
                hits.append(Match(value, phrase, found.start(), found.end(),
                                  strength, "glued", tier))
        return self._collect(hits)

    def _find_despaced(self, text: str) -> list[Match]:
        """Recover a space that arrived in the middle of a word.

        Speech-to-text produces ``בפרוז דור`` for ``בפרוזדור``, so the phrase is
        compared with every space removed. The left boundary survives the
        flattening: a match still has to begin where an original word began,
        give or take the prefix cluster. Without that, ``גן`` would be found
        inside a flattened ``את המזגן`` again.
        """
        flat, starts = _flatten(text)
        hits = []
        for needles, value, phrase, strength, tier in self._despaced:
            for needle in needles:
                at = flat.find(needle)
                while at >= 0:
                    stop = at + len(needle)
                    # Both edges still have to line up with where words began
                    # and ended in the original. Anchoring only the left edge
                    # let "האוטו" (the car, a garage synonym) match inside
                    # "האוטומציה" and route automations to the garage.
                    if (stop in starts or stop == len(flat)) and any(
                            at - back in starts
                            and all(c in _PREFIX_LETTERS
                                    for c in flat[at - back:at])
                            for back in range(4)):
                        hits.append(Match(value, phrase, at, stop,
                                          strength, "despaced", tier))
                        break
                    at = flat.find(needle, at + 1)
        return self._collect(hits)

    def _find_fuzzy(self, text: str) -> list[Match]:
        """Last resort: a misspelling.

        Every phrase is slid across the utterance a window at a time and
        compared by edit distance, counting an adjacent transposition as one
        edit - ``מסדורן`` for ``מסדרון`` is a swap, not two substitutions.

        Two guards keep it from doing damage. The phrase must be at least four
        characters, because ``חצר`` (yard) and ``חדר`` (room) are one edit
        apart and both are ordinary words. And the token has to be introduced
        by a locative preposition - see :func:`locative`, which is what stops
        "the window" from resolving to "the living room".
        """
        tokens = text.split()
        hits = []
        for words, value, phrase, strength, tier in self._fuzzy:
            needle = "".join(words)
            if len(needle) < 4:
                continue
            budget = 1 if len(needle) <= 5 else 2
            span = len(words)
            for start in range(max(len(tokens) - span + 1, 0)):
                if not locative(tokens[start]):
                    continue
                window = "".join(tokens[start:start + span])
                if _near(window, needle, budget):
                    hits.append(Match(value, phrase, start, start + span,
                                      strength, "fuzzy", tier))
                    break
        return self._collect(hits)
