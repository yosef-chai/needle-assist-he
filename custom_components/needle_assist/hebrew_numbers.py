"""Numbers an Israeli says out loud, read back as integers.

The corpus has always *written* these - `data/hebrew_speech.py` spells every
value both ways on purpose, because Whisper emits "22 מעלות" or "עשרים ושתיים
מעלות" depending on the model and the decoding settings, and training on one
of the two forms is betting on which. Nothing has ever *read* them back. Every
number the deterministic layer takes from a sentence has been a digit:
:func:`slot_match.temperature_from` is a `\\d{1,2}` and so is everything
around it.

Measured over the training corpus, on single-clause single-call rows, counting
gold values whose digits do not appear in the sentence at all:

    slot                   as digits  in words
    temperature                  306       347
    brightness_pct               183       319
    minutes                      205       308
    position                     178       277
    volume_pct                   128       177
    percentage                    88       113
    seconds                       11        11
    hours                          4         6
    ------------------------------------------
                                            1558

Better than half of every number in the corpus is a word, and the reader could
see none of them. That is the largest single gap the deterministic layer has,
and it needs no model: the tables below are the inverse of the ones the corpus
was generated from, so they read exactly what it writes - and what a person
says, which is the same thing and the reason the generator has them.

**Unit-anchored, always.** A bare number is not a reading: "תדליק את האור
בחדר שתיים" is a room, "שים טיימר לעשרים" is minutes, and "עשרים ושתיים" on
its own is whatever the last noun said it was. Every public function here
takes the unit with it, which is why there are three of them and not one
`read_number`.

The step slots are deliberately absent. `temperature_step`,
`brightness_step_pct` and `volume_step_pct` are 1,208 rows between them and
almost none of them is a number - they are "קצת", "הרבה יותר", "משמעותית",
an idiom whose magnitude is a convention rather than a reading. `direction`
settles which way those point and the model supplies how far.
"""

from __future__ import annotations

from typing import Final

from .hebrew_text import normalise

# ---------------------------------------------------------------------------
# The words.
#
# Both genders, because the counted noun decides and Hebrew speakers get it
# right: מעלות and דקות and שניות are feminine, אחוז and ערוצים are masculine.
# A reader that only knew one gender would miss half of each slot.
#
# Written folded and de-pointed - `normalise` is applied to the utterance
# before any of this is matched - so ם/מ and ן/נ cost nothing here.
# ---------------------------------------------------------------------------

# Every table here is written the way the words are spelled and matched against
# `normalise`d text, which folds finals - so "עשרים" arrives as "עשרימ" and a
# table written with a final mem never matches it. `_folded` and `_folds` are
# applied at the point of definition rather than to the literals by hand, so a
# word added later cannot reintroduce the bug.


def _folded(table: dict[str, int]) -> dict[str, int]:
    return {normalise(key): value for key, value in table.items()}


def _folds(*words: str) -> tuple[str, ...]:
    return tuple(normalise(word) for word in words)


_UNITS: Final[dict[str, int]] = _folded({
    # feminine
    "אחת": 1, "שתיים": 2, "שתים": 2, "שתי": 2, "שלוש": 3, "ארבע": 4,
    "חמש": 5, "שש": 6, "שבע": 7, "שמונה": 8, "תשע": 9, "עשר": 10,
    # masculine
    "אחד": 1, "שניים": 2, "שנים": 2, "שני": 2, "שלושה": 3, "ארבעה": 4,
    "חמישה": 5, "שישה": 6, "ששה": 6, "שבעה": 7, "תשעה": 9, "עשרה": 10,
    # Zero, which only ever means a percentage or a position - "שים את התריס
    # על אפס" is a closed blind. 32 of the corpus's position rows say it.
    "אפס": 0,
})

#: Teens are two words in Hebrew and the second one is the marker.
_TEEN_MARK: Final = _folds("עשרה", "עשר")

_TENS: Final[dict[str, int]] = _folded({
    "עשרים": 20, "שלושים": 30, "שלשים": 30, "ארבעים": 40, "חמישים": 50,
    "חמשים": 50, "שישים": 60, "ששים": 60, "שבעים": 70, "שמונים": 80,
    "תשעים": 90,
})

_HUNDRED: Final = _folds("מאה", "מאא")

#: `_UNITS` again with the conjunction glued on, which is how the second half
#: of a compound is actually said: "עשרים **ו**שמונה".
_AND_UNITS: Final[dict[str, int]] = {
    f"{normalise('ו')}{word}": value for word, value in _UNITS.items()
}

_ALL_WORDS: Final[dict[str, int]] = {**_UNITS, **_TENS, **_AND_UNITS}

# ---------------------------------------------------------------------------
# Reading one number out of a token stream.
# ---------------------------------------------------------------------------


#: Particles Hebrew glues to the front of the number itself: "**ל**שמונה עשרה
#: מעלות", "**ב**חמישה עשר אחוז". The same list `slot_match._VALUE_PREFIXES`
#: uses, and leaving it out cost every teen in the corpus: "לשמונה" is not a
#: word this table knows, so the reader skipped it and read the עשרה after it
#: as a bare ten.
_PARTICLES: Final = _folds("ל", "ב", "כ", "מ", "ו", "ה", "לב", "על", "עד")


def _bare(word: str, extra: tuple[str, ...] = ()) -> str:
    """``word`` without a leading particle, where one can be spared.

    Only ever strips into something this module recognises - a digit, a number
    word, or one of ``extra`` - so an ordinary word that happens to open with
    ל or ב is left alone. "ל5" is two characters and still a number, which is
    why there is no minimum length here.
    """
    for particle in sorted(_PARTICLES, key=len, reverse=True):
        if not word.startswith(particle) or len(word) <= len(particle):
            continue
        stripped = word[len(particle):]
        if (stripped.isdigit() or stripped in _ALL_WORDS
                or stripped in _HUNDRED or stripped in extra):
            return stripped
    return word


def _value_at(words: list[str], i: int) -> tuple[int, int] | None:
    """``(value, index after it)`` for the number starting at ``words[i]``."""
    word = _bare(words[i])

    # A digit, which is still the other half of the corpus.
    if word.isdigit():
        return int(word), i + 1

    # מאה, alone or as the head of "מאה ועשרים".
    if word in _HUNDRED:
        if i + 1 < len(words):
            nxt = words[i + 1]
            if nxt in _TENS:
                return 100 + _TENS[nxt], i + 2
            if nxt.startswith("ו") and nxt[1:] in _TENS:
                return 100 + _TENS[nxt[1:]], i + 2
        return 100, i + 1

    # A ten, alone or as the head of "עשרים ושמונה".
    if word in _TENS:
        if i + 1 < len(words) and words[i + 1] in _AND_UNITS:
            unit = _AND_UNITS[words[i + 1]]
            if unit < 10:
                return _TENS[word] + unit, i + 2
        return _TENS[word], i + 1

    # A unit, alone or as the head of a teen: "שמונה עשרה" is 18, not 8.
    if word in _UNITS:
        value = _UNITS[word]
        if (i + 1 < len(words) and words[i + 1] in _TEEN_MARK
                and 1 <= value <= 9):
            return 10 + value, i + 2
        return value, i + 1

    return None


def numbers_in(text: str) -> list[tuple[int, int, int]]:
    """Every number in ``text`` as ``(value, first word, one past the last)``.

    Indices are into ``normalise(text).split()``, so a caller can ask what
    noun follows the number without re-tokenising.
    """
    words = normalise(text).split()
    out: list[tuple[int, int, int]] = []
    i = 0
    while i < len(words):
        found = _value_at(words, i)
        if found is None:
            i += 1
            continue
        value, end = found
        out.append((value, i, end))
        i = end
    return out


# ---------------------------------------------------------------------------
# The three unit-anchored readers.
# ---------------------------------------------------------------------------

#: Nouns that make the number in front of them a temperature.
_DEGREES: Final = _folds("מעלות", "מעלה", "מעלת")

#: Nouns that make it a percentage. `אחוזים` and the bare `אחוז` alike, and
#: the English an Israeli reaches for as readily.
_PERCENT: Final = _folds("אחוז", "אחוזים", "האחוז", "פרסנט", "פרוצנט")

_HOURS: Final = _folds("שעות", "שעה", "השעות")
_MINUTES: Final = _folds("דקות", "דקה", "הדקות")
_SECONDS: Final = _folds("שניות", "שניה", "שנייה", "השניות")

#: Idioms that are a percentage without naming one. The generator's own
#: `FRACTION_PCT`, read backwards.
#: `לגמרי` and `עד הסוף` are deliberately absent. They mean "all the way",
#: and which way is the verb's business: "תסגור את התריס לגמרי" is a position
#: of **0**, and "תפסיק לגמרי" is not a percentage at all - it is 114 media
#: rows that this table read as 100% before it was measured.
_PCT_IDIOM: Final[dict[str, int]] = {
    "חצי": 50, "בחצי": 50, "רבע": 25, "שלושה רבעי": 75,
    "מקסימום": 100, "למקסימום": 100, "פול": 100, "הכי חזק": 100,
}

#: And `FRACTION_MIN`, likewise. Minutes, because that is the unit the timer
#: schema counts in when the value is not a whole number of hours.
_MIN_IDIOM: Final[dict[str, int]] = {
    "רבע שעה": 15, "חצי שעה": 30, "שלושת רבעי שעה": 45,
    "שעה וחצי": 90, "שעתיים": 120,
}

#: The hour nouns that count more than one, and the half that can be hung on
#: them. Singular deliberately absent: "שעה וחצי" is already a whole idiom in
#: the table above at ninety minutes, and reading it here would count the half
#: twice. See :func:`_half_past_the_hour`.
_PLURAL_HOURS: Final = _folds("שעות", "שעתיים", "השעות")
_HALF: Final = _folds("וחצי", "חצי")

#: Every way the sentence can say *hours*, for the one question the
#: decomposition asks of it. `_HOURS` alone is not enough: שעתיים is a word
#: rather than a number and a noun, and it lives in `_MIN_IDIOM`.
_HOUR_WORDS: Final = (*_HOURS, *_folds("שעתיים"))

_PCT_IDIOM_F: Final[dict[str, int]] = {
    normalise(k): v for k, v in _PCT_IDIOM.items()}
_MIN_IDIOM_F: Final[dict[str, int]] = {
    normalise(k): v for k, v in _MIN_IDIOM.items()}

#: A temperature this house could be asked for. The schema's own bounds; a
#: number outside them was a channel, a room or a year.
_TEMP_RANGE: Final = (5, 35)


def _anchored(text: str, nouns: tuple[str, ...]) -> set[int]:
    """The distinct values in ``text`` immediately followed by one of ``nouns``.

    "Immediately" allows one word of slack, because Hebrew puts a hedge there
    and speech-to-text puts noise there: "בסביבות עשרים ושמונה מעלות" is the
    number, the noun, and nothing between them, but "עשרים ושמונה בערך מעלות"
    happens too.
    """
    words = [_bare(w, nouns) for w in normalise(text).split()]
    found: set[int] = set()
    for value, _start, end in numbers_in(text):
        for step in (0, 1):
            if end + step < len(words) and words[end + step] in nouns:
                found.add(value)
                break
    return found


def degrees_in(text: str) -> int | None:
    """The target temperature the sentence names, or ``None``.

    ``None`` when it names none and when it names two - a sentence supporting
    two readings settles nothing, which is the rule every table in
    :mod:`slot_match` follows.
    """
    if not text:
        return None
    found = {v for v in _anchored(text, _DEGREES)
             if _TEMP_RANGE[0] <= v <= _TEMP_RANGE[1]}
    return found.pop() if len(found) == 1 else None


def percent_in(text: str) -> int | None:
    """The percentage the sentence names, as a number or as an idiom."""
    if not text:
        return None
    found = {v for v in _anchored(text, _PERCENT) if 0 <= v <= 100}
    phrase = normalise(text)
    # Multi-word idioms first and they win outright, exactly as the
    # multi-word keys do in `slot_match.setting_from`: "שלושה רבעי" contains
    # no percent noun and must not be read as the bare "רבע".
    for key, value in _PCT_IDIOM_F.items():
        if " " in key and key in phrase:
            return value
    if not found:
        singles = tuple(k for k in _PCT_IDIOM_F if " " not in k)
        words = [_bare(w, singles) for w in phrase.split()]
        for key in singles:
            if key in words:
                found.add(_PCT_IDIOM_F[key])
    return found.pop() if len(found) == 1 else None


def _half_past_the_hour(words: list[str]) -> bool:
    """Is there a half hung on the end of an hour count - "שעתיים וחצי"?

    `_MIN_IDIOM` carries "שעה וחצי" and "שעתיים" as whole idioms, and the
    singular one is complete where the plural one is not: "שעתיים" matched,
    the reader took its two hours, and the trailing half fell on the floor. A
    household asking for two and a half hours got a timer half an hour short.

    Glued as well as spaced, because the speech noise glues these -
    "שעתייםוחצי" is in the corpus. Measured over the v11 corpus on every timer
    clause naming one: **11 agree, 0 disagree**.
    """
    # The particle comes off first - "שעתיים" arrives as "לשעתיים" far more
    # often than bare - and by hand rather than through `_bare`, whose
    # allow-list cannot hold "שעתייםוחצי", which is one word and two.
    stems = [next((word[len(particle):] for particle in _PARTICLES
                   if word.startswith(particle) and len(word) > len(particle)),
                  word)
             for word in words]
    for position, stem in enumerate(stems):
        before = stems[position - 1] if position else ""
        for hour in _PLURAL_HOURS:
            if stem.startswith(hour) and stem != hour and stem.endswith(_HALF):
                return True
            if stem in _HALF and before == hour:
                return True
    return False


def duration_in(text: str) -> dict[str, int]:
    """``{"hours": .., "minutes": .., "seconds": ..}`` for the countdown named.

    An empty dict means the sentence named no duration, which is the common
    case and the one that leaves the model's answer alone.

    **Canonical, not literal.** Every unit the sentence names is summed and
    the total is decomposed again with nothing overflowing its own bound, so
    "תשעים שניות" comes back as one minute and thirty seconds. That is what
    the corpus does with it and what Home Assistant's `timer.start` wants;
    `executor._seconds` sums the three back together either way, so the
    decomposition is a spelling and not a decision.

    **Promoted only as far as the sentence went.** Seconds always roll into
    minutes, because the schema caps `seconds` at 59 and there is nowhere else
    for ninety of them to go. Minutes roll into hours only when the sentence
    said *hours*: "תשעים דקות" is ninety minutes and not an hour and a half,
    and `minutes` is capped at 600 precisely so it can hold them.

    Measured over the v11 corpus on every timer clause where the reading is
    the same countdown as gold spelled differently: promoting seconds agrees
    **6 times and disagrees none**, and promoting minutes past an hour the
    sentence never named disagreed **59 times against one agreement** - and
    that one says "לשעתיים ועשרים דקות", so it named the hours too and is not
    a promotion at all. Either spelling is the same command to
    `executor._seconds`; this one is also the words the household used.
    """
    if not text:
        return {}
    phrase = normalise(text)
    total = 0
    named = False
    # Both asked of the raw phrase, before the idiom loop below eats the words
    # they read: it replaces "שעתיים" with a space, and after that neither the
    # half nor the hours are still there to be seen.
    half_hour = _half_past_the_hour(phrase.split())
    said_hours = any(word in phrase for word in _HOUR_WORDS)
    for key, value in _MIN_IDIOM_F.items():
        if key in phrase:
            total += value * 60
            named = True
            phrase = phrase.replace(key, " ")
    words = phrase.split()
    for nouns, scale, ceiling in ((_HOURS, 3600, 24),
                                  (_MINUTES, 60, 600),
                                  (_SECONDS, 1, 600)):
        found = {v for v in _anchored(phrase, nouns) if 0 < v <= ceiling}
        if len(found) == 1:
            total += found.pop() * scale
            named = True
            continue
        if found:
            # Two readings for one unit settles nothing, as everywhere else.
            return {}
        # A naked unit noun is the number as well: "טיימר שעה", "עוד דקה".
        bare_words = [_bare(w, nouns) for w in words]
        for noun in nouns:
            if noun not in bare_words:
                continue
            at = bare_words.index(noun)
            if at and _value_at(words, at - 1) is not None:
                continue
            total += scale
            named = True
            break
    if half_hour and total >= 3600:
        # The half the plural idiom left behind. Guarded on a whole hour having
        # been counted, so a bare חצי - which is also fifty per cent, and the
        # commonest entry in `_PCT_IDIOM` - can never invent a countdown.
        total += 1800
    if not named or total <= 0:
        return {}
    out: dict[str, int] = {}
    if total >= 3600 and said_hours:
        out["hours"] = total // 3600
        total %= 3600
    if total >= 60:
        out["minutes"] = total // 60
        total %= 60
    if total:
        out["seconds"] = total
    return out
