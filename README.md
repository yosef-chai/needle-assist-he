<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="brands/custom_integrations/needle_assist/dark_icon.png">
    <img src="brands/custom_integrations/needle_assist/icon.png" width="120" alt="Needle Assist">
  </picture>
</p>

# Needle Assist — עוזר קולי בעברית ל‑Home Assistant

[![validate](https://github.com/Yosef-Chai/needle-assist-he/actions/workflows/validate.yml/badge.svg)](https://github.com/Yosef-Chai/needle-assist-he/actions/workflows/validate.yml)
[![hacs](https://img.shields.io/badge/HACS-custom-41BDF5.svg)](https://hacs.xyz)

שליטה בבית החכם בעברית — מדוברת, לא רק תקנית — **בלי ענן, בלי מנוי, בלי שהמשפטים שלך יוצאים מהבית**.

המודל, המנוע וההיגיון כולם רצים על אותה מכונה שמריצה את Home Assistant. אין קריאת רשת בזמן שיחה. הדבר היחיד שיורד מהאינטרנט הוא ספריית המנוע (כ‑14MB), פעם אחת, בהתקנה הראשונה.

```
"תדליק את האור בסלון"                        → light.turn_on
"סגור את התריסים בחדר של הילדים"              → cover.close_cover
"מה המצב של המזגן במטבח"                      → get_state   ← לא יפעיל כלום
"שים את המזגן על 23 בחדר שינה"                 → climate.set_temperature
"תפעיל את מצב לילה"                           → scene.turn_on
"אל תדליק את האור"                            → סירוב
"מה מזג האוויר בפריז"                          → סירוב
```

---

## מה זה עושה

- **41 כלים**, כולם שירותים אמיתיים של Home Assistant — תאורה, תריסים, מיזוג, מנעולים, מדיה, שקעים, שואב, מצלמות, סצנות, סקריפטים, אוטומציות, טיימרים, התראות ושאילתות מצב.
- **מזהה את החדרים של הבית שלך**, לא רשימה קבועה מראש. אם יש לך "הול", "חדר כביסה" או "פינת קפה" — הם יעבדו, כולל בכינויים שהגדרת ב‑Home Assistant.
- **מבין עברית מדוברת**: תחיליות (בסלון, ולמטבח, שבמקלחת), אותיות סופיות במקום הלא נכון, מילים דבוקות מזיהוי דיבור, וטעויות הקלדה.
- **שאלה לא יכולה להפעיל כלום.** "מה המצב של האור במטבח" מחזיר תשובה — הוא לא מדליק את האור.
- **מסרב למה שלא קשור לבית**, ומסרב לפקודות שליליות ("אל תדליק").
- עונה בעברית.

## דרישות

| | |
|---|---|
| Home Assistant | 2024.11 ומעלה |
| ארכיטקטורה | x86_64 או aarch64, glibc או musl |
| מקום בדיסק | כ‑40MB (מודל 23MB + מנוע 14MB) |
| GPU | לא נדרש. רץ על המעבד. |
| רשת | רק בהתקנה הראשונה, להורדת ספריית המנוע |

נבדק על Home Assistant OS 2026.8 על Home Assistant Green‏ (aarch64 + musl).

## התקנה

### דרך HACS (מומלץ)

1. HACS ← ⋮ ← **Custom repositories**
2. הוסף את `https://github.com/Yosef-Chai/needle-assist-he` בקטגוריה **Integration**
3. חפש **Needle Assist (Hebrew)** והתקן
4. הפעל מחדש את Home Assistant

### ידנית

העתק את `custom_components/needle_assist` אל תיקיית `custom_components` שבתצורה שלך, והפעל מחדש.

### הגדרה

**Settings ← Devices & services ← Add integration ← Needle Assist (Hebrew) ← Submit.**

זה הכול. אין מה למלא — המודל המכוונן מגיע יחד עם האינטגרציה. השדה בטופס נועד רק למי שאימן מודל משלו.

בהתקנה הראשונה יורדת ספריית המנוע (כ‑14MB) ונשמרת תחת `/config`, כך שהיא שורדת עדכוני ליבה.

לסיום: **Settings ← Voice assistants**, ובחר את Needle Assist כסוכן השיחה.

## כדאי לדעת לפני שמתחילים

**שייך חדר לכל ישות.** פקודה כמו "תדליק את האור בסלון" מגיעה לישויות דרך רישום האזורים של Home Assistant. ישות בלי אזור לא תימצא באף פקודת חדר. שווה לעבור על `Settings ← Devices & services ← Entities`, לסנן לפי ישויות בלי אזור, ולשייך.

**כינויים עוזרים.** אם קוראים לחדר "חדר הורים" אבל בבית אומרים "חדר שינה", תוסיף את זה כ‑alias לאזור. האינטגרציה קוראת את הכינויים ומתייחסת אליהם כמו לשם עצמו.

**פקודה אחת במשפט.** משפטים עם שתי פקודות ("תדליק את האור **וגם** תסגור את התריס") עדיין לא עובדים — ראה מגבלות למטה.

## אפשרויות

**Settings ← Devices & services ← Needle Assist ← Configure**

| הגדרה | ברירת מחדל | מה זה |
|---|---|---|
| סירוב לאמירות שאינן קשורות לבית | פעיל | בלי זה, שאלה על מזג האוויר בפריז עלולה להזיז מכשיר |
| מקסימום טוקנים לאמירה | 192 | קריאת כלי היא כ‑25 טוקנים; העלאה לא משפרת כלום |
| רף ביטחון | 0 | חייב להישאר 0 למודל מכוונן — ראה למטה |

כל ברירת מחדל כאן היא הערך שנמדד כטוב ביותר, לא ניחוש.

## מה נמדד

על 400 שורות מקבוצת בדיקה שלא נראתה באימון, ועל כל 18,806 השורות שנוצרו:

| | |
|---|---|
| בחירת הכלי הנכון | 63.5% |
| כלי **וגם** כל הארגומנטים | 42.5% |
| זיהוי החדר (בקוד, לא במודל) | **99.0%** |
| זיהוי חדר במילים נרדפות שלא נראו באימון | **100%** |
| ניתוב לכלי הנכון (רשימה מקוצרת) | 98.5% |
| שאלות שקיבלו כלים לקריאה בלבד, ולכן לא יכלו להפעיל כלום | 82.0% |
| פקודות אמיתיות שסווגו בטעות כשאלה | **0 מתוך 15,709** |
| סירוב נכון לאמירות מחוץ לתחום | 73.9% |
| פקודות אמיתיות שנדחו בטעות | 2.54% |

**למה זיהוי החדר גבוה בהרבה מדיוק המודל?** כי הוא לא נעשה במודל. המודל מכיר 12 קודי חדר קבועים באנגלית; בית אמיתי לא מתחלק ל‑12. אז החדר, שם המכשיר, שם הסצנה ותוכן ההתראה נקראים מהמשפט בקוד דטרמיניסטי מול הרישומים האמיתיים של Home Assistant. אותה בחירה נעשתה גם לגבי בחירת הכלי ולגבי הסירוב: **כל מה שאפשר להכריע בוודאות, מוכרע מחוץ למודל.**

הסבר מלא: [`docs/HOW-IT-WORKS.md`](docs/HOW-IT-WORKS.md).

## מגבלות ידועות

הרשימה הזו כתובה במלואה בכוונה. עדיף לדעת מראש.

- **משפט עם שתי פקודות לא עובד.** "תדליק את האור וסגור את התריס" — 0% הצלחה. תגיד שתי אמירות.
- **שאלה בלי מילת שאלה לא תמיד מזוהה ככזו.** "התריס בסלון פתוח" בלי סימן שאלה עמום גם בעברית; כ‑18% מהשאלות נופלות לשם. הן לא יפעילו כלום בטעות, אבל הן עלולות לא לענות.
- **המודל עצמו לא מסרב לכלום** — שיעור הסירוב הנכון שלו נמדד ב‑0.0%. הסירוב כולו נעשה בקוד, ותופס 73.9% מהאמירות שמחוץ לתחום. השאר יגיעו למודל.
- **רף הביטחון לא שמיש.** הכיוונון לא מעדכן את ראש הביטחון של המנוע, וקריאות נכונות שאינן באנגלית נמדדו עם ביטחון 0.0. כל רף מעל 0 ידחה פקודות תקינות. לכן ברירת המחדל 0.
- **טקסט חופשי בארגומנט נלקח מהמשפט, לא מהמודל.** עברית מגיעה לארגומנט של כלי כרצף בריחה של שישה תווים לאות, והמודל טועה בהם. גוף ההתראה נחתך מהמשפט עצמו; אם אי אפשר לחתוך אותו — ההתראה לא נשלחת, במקום לשלוח ג'יבריש.
- **ישות בלי אזור לא נגישה בפקודת חדר.** זו התנהגות של Home Assistant, לא באג כאן, אבל היא מפתיעה.
- **עברית בלבד.** האינטגרציה מצהירה `he` ולא `MATCH_ALL`, כדי ש‑Assist לא ינתב אליה אנגלית.

## פרטיות

- אין שום קריאת רשת בזמן שיחה. המשפטים שלך לא יוצאים מהמכונה.
- הדבר היחיד שיורד מהאינטרנט הוא ספריית המנוע, פעם אחת, בהתקנה הראשונה, מ‑Hugging Face. אפשר גם להעתיק אותה ידנית ולהישאר מנותק לגמרי — ראה `engine_lib.py`.
- אין טלמטריה, אין דיווח שימוש, אין מפתחות API.

## בעיות

`Settings ← Devices & services ← Needle Assist ← ⋮ ← Download diagnostics` מייצר דוח עם מה שהאינטגרציה רואה: אילו חדרים היא זיהתה, כמה ישויות הן ברות‑פנייה, ומצב השערים. זה המקום להתחיל.

## תודות ורישוי

- מנוע ומודל בסיס: [Needle 2](https://github.com/cactus-compute/needle) מאת Cactus Compute. משקולות הבסיס: [Cactus-Compute/needle2](https://huggingface.co/Cactus-Compute/needle2), רישיון Apache‑2.0.
- הקוד כאן: MIT. המודל המצורף: יצירה נגזרת של needle2 תחת Apache‑2.0.
- פירוט מלא: [`NOTICE`](NOTICE).

---
---

# Needle Assist — a Hebrew voice assistant for Home Assistant

Control your smart home in Hebrew — colloquial, not just formal — **with no cloud, no subscription, and nothing you say leaving the house.**

The model, the engine and the logic all run on the machine running Home Assistant. There is no network call during a conversation. The only thing downloaded from the internet is the engine library (~14MB), once, at first setup.

## What it does

- **41 tools**, every one a real Home Assistant service: lights, covers, climate, locks, media, switches, vacuum, cameras, scenes, scripts, automations, timers, notifications and state queries.
- **Knows your rooms**, not a fixed list. A house with a "הול" or a "חדר כביסה" works, aliases included.
- **Handles spoken Hebrew**: prefix particles, misplaced final letters, words glued together by speech-to-text, and typos.
- **A question cannot actuate.** Asking the state of a light answers; it does not switch it.
- **Refuses** what is not about the house, and refuses negated commands.

## Requirements

Home Assistant 2024.11+, x86_64 or aarch64, glibc or musl, ~40MB of disk. No GPU. Verified on Home Assistant OS 2026.8 on a Home Assistant Green (aarch64 + musl).

## Install

Add `https://github.com/Yosef-Chai/needle-assist-he` as a HACS custom repository (category: Integration), install, restart, then **Settings → Devices & services → Add integration → Needle Assist (Hebrew) → Submit**. Nothing to fill in — the tuned model ships with the integration. Finally pick it as your conversation agent under **Settings → Voice assistants**.

**Assign an area to every entity.** An entity with no area cannot be reached by any room command. That is Home Assistant's own targeting, but it surprises people.

## Measured

On 400 unseen held-out rows, and across all 18,806 generated rows:

| | |
|---|---|
| correct tool | 63.5% |
| tool **and** every argument | 42.5% |
| room resolution (in code, not in the model) | **99.0%** |
| room resolution, synonyms never seen in training | **100%** |
| tool shortlist recall | 98.5% |
| questions given read-only tools only, so they cannot actuate | 82.0% |
| real commands wrongly classified as questions | **0 of 15,709** |
| off-topic correctly refused | 73.9% |
| real commands wrongly refused | 2.54% |

Room resolution beats the model's own accuracy because it is not done by the model. The model knows twelve fixed English room slugs; a real house does not partition into twelve. So the room, the device name, the scene name and the notification text are read out of the sentence by deterministic code against Home Assistant's own registries. The same choice was made for tool selection and for refusal: **anything that can be decided with certainty is decided outside the model.**

See [`docs/HOW-IT-WORKS.md`](docs/HOW-IT-WORKS.md).

## Known limitations

- **Two commands in one sentence do not work** — 0% on multi-command utterances. Say them separately.
- **A question with no interrogative** ("the blind in the living room [is] open") is not always recognised as one. It cannot actuate by accident, but it may not answer.
- **The model itself never refuses anything** — 0.0% correct refusal measured. Refusal is done in code and catches 73.9% of off-topic utterances.
- **The confidence threshold is unusable.** Fine-tuning does not update the engine's confidence head and correct non-English calls measure 0.0, so any threshold above 0 rejects valid commands. Hence the default of 0.
- **Free-text arguments come from the sentence, not the model.** Hebrew reaches a tool argument as six-character escapes and the model gets them wrong.
- **Hebrew only.** The integration declares `he` rather than `MATCH_ALL`.

## Licence

Code MIT. The bundled model is a fine-tuned derivative of [Cactus-Compute/needle2](https://huggingface.co/Cactus-Compute/needle2), Apache-2.0. Engine by [Cactus Compute](https://github.com/cactus-compute/needle), MIT. Full detail in [`NOTICE`](NOTICE).
