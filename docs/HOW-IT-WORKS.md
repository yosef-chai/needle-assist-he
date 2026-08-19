# How it works

The short version: **a 26M-parameter model chooses the verb, and deterministic
Hebrew code decides everything else.** This document explains why the line is
drawn there, because the answer is not "the model was not good enough" — it is
that several of these jobs cannot be done by any model of any size without
making the result worse.

Every number below is measured, and the measurement is named.

---

## The pipeline

```
  utterance
     │
     ├─ 1. negation guard        "אל תדליק" → refuse, before anything runs
     ├─ 2. off-topic gate        "מה מזג האוויר בפריז" → refuse
     ├─ 3. clause split          one order per clause; the rest runs per clause
     ├─ 4. read-only gate        a question gets read-only tools, and only those
     ├─ 5. tool router           ≤5 tools declared, chosen by Hebrew keywords
     │
     ├─ 6. the model             picks one tool and fills its numeric slots
     │
     ├─ 7. direction guard       lock or unlock, and which way a step points
     ├─ 8. slot resolution       room, device, scene, message, music ← from the sentence
     └─ 9. Home Assistant        the real service call
```

Everything but step 6 is ordinary Python. Step 6 is the only inference, and
it runs once per clause.

---

## Why the sentence is cut before the model sees it

Asked for two orders at once, the model answers with one call. Measured on
the 97 multi-order rows of the held-out set: **94 of 97** come back as a
single call, for **0.0%** tool-set accuracy — usually the verb of one clause
with the room of another.

```
gold  vacuum_start{} · lock_lock{office} · cover_close{parking}
pred  vacuum_start{area: office}
```

It is not the grammar (asked directly, the engine returns two calls), not
truncation (a three-call target peaks at 157 of 192 tokens), and not the
context window (the same adapter exported at window 704 and at 160 gives
byte-identical metrics). It is that 94% of the training targets contain
exactly one call, so the prior to stop after the first is overwhelming.

So the sentence is cut at a comma, at a joining phrase (`וגם`, `ואז`,
`ואחר כך`), or at a word whose leading `ו` is glued to an action verb —
Hebrew writes "and close" as one token, `וסגור`. A cut is kept **only if
both sides contain an action verb**, which is the whole safety condition:

| sentence | cut? | why |
|---|---|---|
| `תדליק את האור בסלון וסגור את התריסים` | yes | both sides have a verb |
| `תדליק את האור בסלון ובמטבח` | no | `ובמטבח` is a room |
| `תדליק את האור והמזגן בסלון` | no | `והמזגן` is a device |
| `אה, סגור את האור` | no | a filler is not an order |
| `מה המצב של האור בסלון ובמטבח` | no | interrogatives are not action verbs |

On the same rows and the same weights: **0.0% → 75.3%** tool-set, 0.0% →
42.3% exact. On 400 seeded rows of the whole test set every family scores
identically except the multi-order one — the cut fires where it is meant to
and nowhere else — and the overall numbers rise 63.5% → 66.8% tool-set and
42.5% → 44.2% exact.

The other shape is one order over several rooms — `תכבה את האור בסלון
ובמטבח` — which stays one clause and targets both rooms. Rooms named in
*overlapping* spans are not two rooms but two readings of one: a house with
`חדר הורים` and `חדר ילדים` must not light the parents' room because the
children's was asked for.

---

## Music

`music_assistant.play_media` takes a search string and resolves it against
the library [Music Assistant](https://www.music-assistant.io/) already
indexes. So the split of labour is the same as everywhere else here:

* the **model** supplies `media_type` — one of `track / album / artist /
  playlist / radio`, evidenced by a word the speaker said;
* the **sentence** supplies the title, the artist and the room;
* **Music Assistant** does the lookup.

```
תנגן לי את אם ננעלו של עומר אדם   → media_id "אם ננעלו", artist "עומר אדם"
שים לי את האלבום שבלול של כוורת   → media_id "שבלול", album, artist "כוורת"
תנגן רדיו גלגלצ במטבח             → media_id "גלגלצ", radio, room stripped
תנגן קצת מוזיקה בסלון             → nothing named: resume, do not search
```

The tool has no argument for what to play, deliberately — Hebrew reaches a
tool argument as six-character escapes and the model gets them wrong, which
is the same measurement that moved notification text out of the model. A
residue made only of device nouns is not a title (`תפעיל את השואב` stays a
vacuum), and a house with no Music Assistant falls back to
`media_player.media_play` on the room's own speaker.

---

## Why the verb overrules the model

Every model trained for this project inverts a command sometimes, and the
worst version of that is the worst thing a home assistant can do:

    נעל את הדלת בחדר ההורים          lock the bedroom door
    lock_unlock{area: bedroom}       ...unlocked it

The Hebrew verb settles it. `נעל` is lock and `פתח` is unlock, `תדליק` is on
and `תכבה` is off, and when the words and the model disagree the words win.
Measured against the labels over all 2,379 held-out rows, on the calls whose
tool belongs to one of the ten guarded pairs:

| | | |
|---|---:|---:|
| the words agree with the label | 1337 | 86.7% |
| the words disagree | **0** | 0.00% |
| the words say nothing | 205 | 13.3% |

Zero. A signal that is never wrong when it speaks can overrule a model that is
wrong one time in eight — and the correction is deliberately narrow: it never
changes which **domain** the model chose, only which of two tools inside it.
"תפעיל את השואב" is answered by the vacuum and never reaches the light pair.

Three things had to be excluded, and the measurement found each one:

* **A retraction is not an order.** "תכבה את המנורה, לא לא, תעשה את המנורה"
  says the wrong verb first and fixes it. The verb before the correction is
  the one the speaker withdrew, so the guard stands down — and the clause
  splitter stands down on the same sentences, or the light would be switched
  off and then on again.
* **A dim is a brightness, not an off.** `עמעם` belongs under `light_turn_off`
  for routing, because a dim is a reduction, and does not belong here.
* **`נגן` is both "play!" and "the player".** "תעצור את הנגן" carries a play
  hint and a pause verb, and no amount of counting separates them. Music that
  keeps playing is not a door that opens, so play and pause are left alone.

### Which way the number points

Three arguments are a change rather than a value — the temperature step, the
brightness step and the volume step — and each is added to what the device
currently reads. Their sign is a direction, so getting it backwards is the
same defect one level down:

    בסלון הגדול חם מדי, תנמיך משמעותית          it is too hot, lower it a lot
    climate_set_temperature{temperature_step: 4}  ...raised it four degrees

Eight degrees wrong, from one character. The sentence is not ambiguous, and it
reads in three tiers because Hebrew comparatives stack: **the verb** first
(`תנמיך` down, `תגביר` up), then **the adjective** (`חלש` quiet, `חזק` loud),
and only then the bare **`פחות` / `יותר`**. The order carries the whole thing —
"יותר חלש" is *more quiet*, not more, and reading the comparative first gets
all 32 of those backwards.

Measured the same way: **1222 agree, 1 disagrees, 12 say nothing.** The one is
a sentence where the corpus's simulated speech noise corrupted the direction
word itself.

Only the sign is taken. The size — `קצת` one step, `משמעותית` four — was
measured too and reaches 96.6%, and that is not zero: two degrees short is not
eight degrees backwards, and a step too small is one the speaker repeats.

---

## Why the room is not resolved by the model

The model was fine-tuned to emit one of twelve English area slugs —
`living_room`, `kitchen`, `bedroom` and so on. Two things are wrong with that,
and neither is fixable by training harder.

**A house does not partition into twelve.** An installation with a הול, a
חדר כביסה, a מחסן or a פינת קפה has rooms the model holds no token for. Under a
slug-based design those rooms are unreachable by voice — not inaccurate,
*unreachable*.

**Two rooms can fold onto one slug.** A home with both a מקלחת (shower room) and
a שירותים (toilet) produces the same slug for both. The integration then picks
whichever came first in the registry, and the model's answer is *correct* while
the wrong room lights up. No amount of held-out accuracy surfaces that, because
the metric says the model was right.

So the room is matched out of the sentence against Home Assistant's own area
registry, including the aliases you set yourself:

| | model | resolver |
|---|---:|---:|
| area accuracy, same held-out rows | 51.4% | **99.0%** |
| synonyms held out of training entirely | — | **100%** |
| a 9-room house, 7 rooms outside the model's vocabulary | 0 | **12/12** |
| shower room vs toilet, one slug | wrong room | **4/4** |

The same move is made three more times: **device names**, **scene, script,
automation and timer names**, and **notification text**. A household whose scene
is called `מצב סרט` was never addressable through a model trained on forty
invented English slugs.

### Why not have the model copy the Hebrew span instead

It was tried and abandoned. The engine emits non-ASCII inside a tool argument as
`\uXXXX` escapes — six exact characters per Hebrew letter — and the model gets
the hex wrong. Measured output for a notification body includes a Hangul
syllable produced by dropping one hex digit. The sentence already contains the
words; reading them out of it is both cheaper and correct.

---

## The Hebrew matcher

Hebrew glues particles onto the front of a word, and the chain is ordered:
conjunction ו, then subordinator ש/כש, then preposition ב/ל/כ/מ, then the
article ה. `ובמטבח`, `שבמקלחת` and `ולסלון` all have to reach the same room.

Modelling the real chain rather than "strip up to three prefix letters" is what
stops `המרחב המוגן` from being read as a garden: `המו` is not a legal prefix
chain, so `גן` inside it is not a word.

Four passes run in order, each only if the stronger one found nothing:

| pass | what it allows | guard that stops it over-matching |
|---|---|---|
| strict | whole word, both boundaries | — |
| glued | no left boundary | needs ≥1 prefix letter and a body of ≥4 characters, so `גן` inside `מזגן` cannot match |
| despaced | spaces removed (speech-to-text glues words) | both edges must land on original word boundaries, so `האוטו` inside `האוטומציה` cannot match |
| fuzzy | bounded Damerau-Levenshtein, budget 1–2 | phrase ≥4 characters, and the token must carry a locative prefix, so `החלון` does not become `סלון` |

Every one of those guards exists because its absence produced a specific wrong
room in the corpus. Final letters (ךםןףץ) are folded throughout: speech-to-text
puts them mid-word.

---

## Why a question cannot actuate

Of 18,806 generated rows, 930 are state questions — 5.0%. The model sees "turn
on the light" seventeen times for every "is the light on". Given a shortlist of
four actuation tools and one read-only tool, the prior wins, and a question
turns the light on.

That is not fixed by training. It is fixed by not declaring the tool: Needle
compiles its decoding grammar from the declared schemas, so a tool that is not
in the shortlist cannot be emitted at all.

Eight interrogative patterns were kept, each only because it fires on **zero** of
the 15,709 actuation rows. Together they cover 82.0% of question rows at 0.00%
false positives. Three candidates were measured and rejected:

| candidate | question hits | actuation hits |
|---|---:|---:|
| state adjectives (דולק, פתוח, נעול) | 159 | 227 |
| אני רוצה | 95 | 740 |
| a phrase index instead of regexes | 84.2% recall | 12.12% |

`תשאיר את האור דולק` is an order, not a question; and the phrase index's glued
pass finds האם inside the command `אם אפשר`.

Which of the two read-only tools is also decided in code, because the model
cannot: with both declared it answered a question about a light with the
weather. The sky is checked first — weather words appear in 2.0% of device
questions, but a weather sentence names a device noun far more often than that,
so checking the device first sent 64% of weather rows to the wrong tool.

Result: 13.5% → **84.0%** tool-set accuracy on a probe of question utterances.

The `domain` a question asks about is read from the same device nouns that route
a command — 98.0% correct, against 0.9% argument F1 for the model on that slot.
חלון is a window contact; וילון and תריס are a blind. The model never once told
them apart.

---

## Why tool selection is not left to the engine

Needle runs a retrieval head when more than five tools are declared: it renders
its top five per turn and constrains the grammar to those. That head is trained
contrastively against English tool descriptions, and nothing in the fine-tuning
path touches it.

| declared | tool-set accuracy |
|---|---|
| all 41, retrieval head chooses | 6.8% |
| the row's own ≤5 | 54.4% |

An eightfold difference, and the failures are not random — the head returns a
near-constant top five. It is embedding Hebrew it cannot read.

So a keyword scorer picks at most five tools and the head never runs. Device
**nouns** weigh triple: Hebrew imperatives are shared across domains (תפתח opens
a blind, unlocks a door, and colloquially turns on a light), so the noun decides
the domain and the verb only breaks ties. Recall @5 over the whole corpus:
**99.1%**, measured the way the integration runs it - one shortlist per
clause. On the whole sentence in one piece it is 98.5%.

---

## Why refusal is not in the model either

Correct-refusal rate for every model trained here: **0.0%**. False actuation on
off-topic input: ~100%. Training a refusal class was tried; the model memorised
the training refusals and generalised none of them.

The router's family score does the job instead: 75.0% of off-topic utterances
score below the threshold, against 1.87% of genuine commands. A named room
counts for two of the three the gate asks for - nobody says "בסלון" about the
pyramids - and deliberately not three, so a room never clears the bar alone. The asymmetry is
deliberate — a refused command costs a repeat, an actuated question costs a
device moving in someone's house.

Negation is separate and stricter. The engine's own negation detector matches
`don't`, `never`, `no longer` and nothing else, and there is not one negated
command in the whole corpus, so the model has never seen one. A short list of
unambiguous Hebrew forms blocks them before inference. Bare `לא` is deliberately
**not** on that list: 327 rows use it to correct mid-sentence
("תדליק את המנורה, לא לא, תכבה"), and every one of those must still act.

---

## Two things about the model itself

**The think block stays, and that was measured.** Every training target is
`<think>…</think><tool_call>[…]</tool_call>`, and the engine never decodes the
think half — it constrains generation to the tool call. Roughly 60% of the
supervised tokens therefore train a sequence inference never produces. Removing
it looks obviously correct and is wrong:

| target | tokens/row | val loss | tool-set |
|---|---:|---:|---:|
| grounded think block | 57.1 | 0.1140 | **36.7%** |
| empty think block | 25.8 | 0.0927 | 30.0% |
| no think block | 22.8 | 0.1092 | 17.7% |

Note that validation loss ranks them wrongly and confidently: the variant with
the second-best loss is the worst model by a factor of two, because the three
have different numbers of supervised tokens and their cross-entropies are not on
the same scale.

**Hebrew is expensive in this tokenizer.** The vocabulary contains 8192 pieces
and **zero** of them are Hebrew, so Hebrew falls back to bytes at roughly two
tokens per character. Combined with a 256-token sliding attention window, that
is the real constraint on how much can be said in one utterance — and one more
reason the deterministic layers earn their place: every job moved out of the
model is tokens the model does not have to spend.

---

## Design rule

Everything above is one rule applied five times:

> If it can be decided with certainty from the sentence and the registries,
> decide it in code. Leave the model only the part that genuinely requires
> judgement.

What is left for the model is: which of five verbs, and what number goes in the
slot. That it does at 66.8% tool-set and 44.2% exact on unseen held-out rows —
and the parts around it are at 98–100%.

The rule is also why multi-order sentences went from 0% to 75% without
retraining anything. Where a capability is missing, the first question is
whether it is a judgement the model has to make, or a fact about the sentence
that code can settle. Segmentation was the second kind.
