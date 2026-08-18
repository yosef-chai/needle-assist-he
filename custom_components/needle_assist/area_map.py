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


def slug_for_name(name: str) -> str | None:
    """Canonical slug for a Home Assistant area name, if we recognise it."""
    return _ALIAS_INDEX.get(_norm(name))


def names_for_slug(slug: str) -> list[str]:
    """Every name this slug might appear under."""
    return AREA_ALIASES.get(slug, [])
