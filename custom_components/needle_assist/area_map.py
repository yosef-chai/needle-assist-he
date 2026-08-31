"""The Hebrew room lexicon: what Israelis call the rooms of a house.

One table, keyed by a canonical slug, listing the surface forms a speaker might
use for that room. It has two consumers and no logic of its own:

* :mod:`slot_match` uses it as a *bridge*. An installation whose areas are
  named ``Living Room`` still answers to ``בסלון``, because both fold onto
  ``living_room`` here. It is only ever a bridge - an area's own name and its
  Home Assistant aliases outrank anything in this file, which is what lets a
  house with a ``חדר כביסה`` work without a single entry being added.
* ``data/generate.py`` draws the same forms when it builds training sentences,
  so the lexicon and the corpus cannot drift apart.

Resolving an utterance to an ``area_id`` used to live here, keyed on the twelve
English slugs the model emits. It moved to :mod:`slot_match`, which matches
against the areas an installation actually has: measured on the same held-out
rows, the slug route is right 51.4% of the time and the sentence route 99.0%,
and only the second one can address a room that is not one of the twelve.

A household whose room is called something not listed here adds a Home
Assistant *alias* to the area (Settings > Areas > alias) and it resolves with
no code change and no retraining.
"""

from __future__ import annotations

from typing import Final

# Same Hebrew forms the training data uses, minus the locative ב prefix.
#
# Four slugs are deliberately not the most obvious English word for the room:
# ``nursery`` (kids' room), ``washroom`` (bathroom), ``terrace`` (balcony) and
# ``parking`` (garage). The model emits the slug one byte at a time, and slugs
# sharing a leading byte were being resolved by its prior rather than by the
# query, so the twelve are kept disjoint at byte 1. The natural English names
# are listed as aliases below, so an install whose areas are named "Balcony" or
# "Garage" still resolves.
AREA_ALIASES: Final[dict[str, list[str]]] = {
    "living_room": ["סלון", "הסלון", "סלון הגדול", "הסלון הגדול",
                    "חדר אורחים", "חדר האורחים", "ליווינג",
                    "חדר מגורים", "חדר המגורים",
                    "living room", "lounge"],
    "kitchen": ["מטבח", "המטבח", "פינת המטבח", "מטבחון", "המטבחון",
                "כיריים", "הכיריים", "kitchen"],
    "bedroom": ["חדר שינה", "חדר השינה", "חדר ההורים", "חדר שלי", "החדר שלי",
                "חדר שלנו", "החדר שלנו", "חדר שינה שלנו", "החדר שינה שלנו",
                "bedroom", "master bedroom"],
    "nursery": ["חדר ילדים", "חדר הילדים", "חדר של הילדים", "החדר של הילדים",
                "הילדים", "חדר תינוק", "חדר התינוק",
                "חדר של הקטן", "החדר של הקטן", "חדר של הקטנה",
                "החדר של הקטנה", "חדר של הבן", "החדר של הבן",
                "kids room", "kids_room", "children", "nursery"],
    "washroom": ["אמבטיה", "האמבטיה", "חדר אמבטיה", "חדר האמבטיה", "שירותים",
                 "השירותים", "שרותים", "מקלחת", "המקלחת", "חדר רחצה",
                 "bathroom", "toilet", "washroom", "shower"],
    "terrace": ["מרפסת", "המרפסת", "בלקון", "הבלקון", "מרפסת שמש",
                "מרפסת השמש", "balcony", "terrace", "patio"],
    "office": ["משרד", "המשרד", "חדר עבודה", "חדר העבודה", "סטודיו", "הסטודיו",
               "חדר מחשב", "חדר המחשב", "פינת עבודה", "פינת העבודה",
               "office", "study"],
    "dining_room": ["פינת אוכל", "פינת האוכל", "חדר אוכל", "חדר האוכל",
                    "שולחן האוכל", "השולחן האוכל",
                    "dining room"],
    "hallway": ["מסדרון", "המסדרון", "כניסה", "הכניסה", "פרוזדור", "הפרוזדור",
                "לובי", "הלובי", "מבואה", "המבואה",
                "כניסה לבית", "הכניסה לבית",
                "hallway", "corridor", "entrance", "foyer"],
    "garden": ["גינה", "הגינה", "חצר", "החצר", "גן", "הגן", "חצר אחורית",
               "החצר האחורית", "דשא", "הדשא", "חוץ", "החוץ",
               "garden", "yard", "backyard"],
    "parking": ["חניה", "החניה", "חנייה", "החנייה", "מוסך", "המוסך", "גראז'",
                "הגראז'", "חניון", "החניון", "אוטו", "האוטו",
                "garage", "carport", "parking", "driveway"],
    "safe_room": ["ממד", 'ממ"ד', "ממ״ד", "הממד", "חדר ביטחון", "חדר הביטחון",
                  "חדר מוגן", "החדר המוגן",
                  "מרחב מוגן", "המרחב המוגן", "מרחב המוגן",
                  "safe room", "shelter"],
}


def _norm(text: str) -> str:
    """Normalise for comparison: casefold, strip quotes and separators."""
    out = text.strip().casefold()
    for ch in ('"', "'", "״", "׳", "-", "_"):
        out = out.replace(ch, " ")
    return " ".join(out.split())


# Reverse index built once: normalised name -> canonical slug.
_ALIAS_INDEX: Final[dict[str, str]] = {
    _norm(alias): slug
    for slug, aliases in AREA_ALIASES.items()
    for alias in aliases
}
for _slug in AREA_ALIASES:
    _ALIAS_INDEX.setdefault(_norm(_slug), _slug)


# The same bridge, one level up: what Israelis call the floors of a house.
#
# Home Assistant has had a floor registry since 2024.4 and five built-in
# intents take a `floor` slot. This table exists for exactly one job - reading
# a *registry floor name* and deciding which of the five slugs the model emits
# it answers to - and it is never matched against an utterance. That
# distinction is the whole safety argument: "למעלה" is already in this
# project's weather vocabulary ("חם למעלה"), so a lexicon consulted on the
# sentence would turn a question about the heat upstairs into a command for
# every device on a floor. Consulted on a floor's *name*, it cannot.
#
# The sentence route is `slot_match.areas_on_floor`, which builds its index
# from the registry and the household's own aliases and outranks this
# entirely. See `slot_match.areas_on_floor_slug`.
FLOOR_ALIASES: Final[dict[str, list[str]]] = {
    "upper": ["קומה עליונה", "הקומה העליונה", "עליונה", "למעלה", "קומה שנייה",
              "קומה שניה", "upper", "first floor", "upstairs"],
    "lower": ["קומה תחתונה", "הקומה התחתונה", "תחתונה", "למטה", "lower",
              "downstairs"],
    "ground": ["קומת קרקע", "קומת הקרקע", "קרקע", "קומת כניסה", "כניסה",
               "ground", "ground floor"],
    "basement": ["מרתף", "המרתף", "בייסמנט", "basement", "cellar"],
    "roof": ["גג", "הגג", "קומת גג", "גג הבית", "roof", "rooftop"],
}

_FLOOR_INDEX: Final[dict[str, str]] = {
    _norm(alias): slug
    for slug, aliases in FLOOR_ALIASES.items()
    for alias in aliases
}
for _slug in FLOOR_ALIASES:
    _FLOOR_INDEX.setdefault(_norm(_slug), _slug)


# The floors a *sentence* may name, which is a different table from the one
# above and had to be measured rather than assumed.
#
# `FLOOR_ALIASES` reads a registry floor *name*; this reads an utterance, and
# until v11 there was no such table at all - deliberately, because the corpus
# had no floor rows to measure one against and the obvious candidates are the
# dangerous ones. v11 added the family, so the measurement exists now. Every
# phrase below was scored against all 30,622 corpus rows, and only the ones
# that fire on **zero** rows carrying no floor are here:
#
#     מרתף          72 right, 0 elsewhere      הגג        144 right, 0 elsewhere
#     קרקע          54 right, 0 elsewhere      קומת הגג    49 right, 0 elsewhere
#     בייסמנט       33 right, 0 elsewhere      קומה ראשונה 17 right, 0 elsewhere
#
# Two are missing on purpose and they are the two anybody would reach for
# first. `למטה` fires on **159** rows that name no floor - "תוריד את התריס
# למטה" lowers a blind, and it would have lowered every blind in the house.
# `למעלה` measures 43 right and zero collisions on this corpus, and stays out
# anyway: the corpus's own weather vocabulary contains "חם למעלה", the two
# adverbs are one word apart in Hebrew, and a table that keeps one of a
# symmetric pair because the other happened to collide is fitting the corpus
# rather than the language. Both are still reachable the way every unusual room
# is - a household that adds "למעלה" as a Home Assistant floor alias gets it,
# and has said so on purpose.
#
# Qualifying them was tried too, because "בקומה למטה" is not ambiguous the way
# "למטה" is. As a pair - `קומה למטה` and `קומה למעלה` together, so each fences
# the other - they measure 49 right, **2 wrong** and zero collisions. The two
# are `בקומה למתה`, ordinary ט/ת speech noise, and the fuzzy pass reads it as
# the opposite floor. Elsewhere in this project two disagreements out of fifty
# would ship; here they would not, because what is wrong is the *target*. A
# floor that is not resolved falls back to the room the speaker is standing in,
# and a floor resolved to the wrong one turns off the lights upstairs.
#
# Consulted at tier 1, below the household's own floor names, and only for a
# floor this installation actually has: see `slot_match.Slots._floor_index`.
# `למטה` and `למעלה` are how a household actually says which floor - "סגור את
# הווילונות למטה", "בקומה למטה" - and neither the registry names nor the
# phrases below reached them, so every one of those sentences lost its floor.
#
# The obvious worry is that both words are also directions: "תוריד את התריס
# למטה" lowers a blind. Measured over the corpus, 191 clauses say one of the
# two and they separate on a line that is already drawn - 113 belong to a tool
# that *has* a floor argument and every one of them names the floor the word
# says; the other 78 belong to `notify_send`, where "תשלח הודעה לכולם שתרדו
# למטה" is a message and not a target, and the `"floor" in takes` guard both
# pipelines already apply keeps the fill away from them. 113 agree, 0 disagree.
FLOOR_PHRASES: Final[dict[str, list[str]]] = {
    "upper": ["קומה עליונה", "הקומה העליונה", "קומה שנייה", "קומה שניה",
              "למעלה"],
    "lower": ["קומה תחתונה", "הקומה התחתונה", "למטה"],
    "ground": ["קומת קרקע", "קומת הקרקע", "קומת כניסה", "קומת הכניסה",
               "קרקע", "קומה ראשונה", "הקומה הראשונה"],
    "basement": ["מרתף", "בייסמנט"],
    "roof": ["הגג", "קומת גג", "קומת הגג", "גג הבית"],
}


def floor_slug_for_name(name: str) -> str | None:
    """Canonical floor slug for a Home Assistant floor name, if recognised."""
    return _FLOOR_INDEX.get(_norm(name))


def slug_for_name(name: str) -> str | None:
    """Canonical slug for a Home Assistant area name, if we recognise it."""
    return _ALIAS_INDEX.get(_norm(name))


def names_for_slug(slug: str) -> list[str]:
    """Every name this slug might appear under."""
    return AREA_ALIASES.get(slug, [])
