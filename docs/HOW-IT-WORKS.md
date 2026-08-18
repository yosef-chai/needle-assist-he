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
     ├─ 3. read-only gate        a question gets read-only tools, and only those
     ├─ 4. tool router           ≤5 tools declared, chosen by Hebrew keywords
     │
     ├─ 5. the model             picks one tool and fills its numeric slots
     │
     ├─ 6. slot resolution       room, device, scene, message ← from the sentence
     └─ 7. Home Assistant        the real service call
```

Steps 1–4 and 6 are ordinary Python. Step 5 is the only inference.

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
**98.5%**.

---

## Why refusal is not in the model either

Correct-refusal rate for every model trained here: **0.0%**. False actuation on
off-topic input: ~100%. Training a refusal class was tried; the model memorised
the training refusals and generalised none of them.

The router's family score does the job instead: 73.9% of off-topic utterances
score below the threshold, against 2.54% of genuine commands. The asymmetry is
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
slot. That it does at 63.5% tool-set and 42.5% exact on unseen held-out rows —
and the parts around it are at 98–100%.
