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
"תכבה את האור בסלון וסגור את התריסים"         → שתי פעולות במשפט אחד
"תנגן לי את אם ננעלו של עומר אדם"             → music_assistant.play_media
"מה המצב של המזגן במטבח"                      → get_state   ← לא יפעיל כלום
"שים את המזגן על 23 בחדר שינה"                 → climate.set_temperature
"תפעיל את מצב לילה"                           → scene.turn_on
"אל תדליק את האור"                            → סירוב
"מה מזג האוויר בפריז"                          → סירוב
```

---

## מה זה עושה

- **42 כלים**, כולם שירותים אמיתיים של Home Assistant — תאורה, תריסים, מיזוג, מנעולים, מדיה, מוזיקה, שקעים, שואב, מצלמות, סצנות, סקריפטים, אוטומציות, טיימרים, התראות ושאילתות מצב.
- **מזהה את החדרים של הבית שלך**, לא רשימה קבועה מראש. אם יש לך "הול", "חדר כביסה" או "פינת קפה" — הם יעבדו, כולל בכינויים שהגדרת ב‑Home Assistant.
- **מבין עברית מדוברת**: תחיליות (בסלון, ולמטבח, שבמקלחת), אותיות סופיות במקום הלא נכון, מילים דבוקות מזיהוי דיבור, וטעויות הקלדה.
- **כמה פעולות במשפט אחד.** "תכבה את האור במטבח וסגור את התריסים בחדר שינה" זה שתי פקודות, והן מבוצעות שתיהן. המשפט נחתך לפני שהמודל רואה אותו, ורק במקום שבו באמת מתחילה פקודה חדשה — "תדליק את האור בסלון ובמטבח" נשאר פקודה אחת שפועלת על שני החדרים.
- **מנגן מוזיקה בשם** דרך [Music Assistant](https://www.music-assistant.io/): "תנגן לי את האלבום שבלול של כוורת", "שים פלייליסט רגוע בסלון", "תנגן רדיו גלגלצ". שם השיר נקרא מהמשפט, לא מומצא על ידי המודל. בלי Music Assistant מותקן זה פשוט ממשיך נגינה על הרמקול שבחדר.
- **שאלה לא יכולה להפעיל כלום.** "מה המצב של האור במטבח" מחזיר תשובה — הוא לא מדליק את האור.
- **הפועל בעברית קובע את הכיוון.** "נעל את הדלת" נועל, "תנמיך את המזגן" מוריד. מודל טועה בכיוון לפעמים, וזו הטעות הגרועה ביותר שעוזר בית יכול לעשות — אז המילים גוברות עליו. נמדד על כל קבוצת המבחן: **1337 פעמים המילים מסכימות עם התווית, 0 פעמים הן סותרות אותה**.
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

**כמה פקודות במשפט אחד עובדות.** "תכבה את האור בסלון ותנעל את הדלת" נחתך לשתי פקודות ומורץ אחת אחרי השנייה — 95.3% בחירת כלי נכונה על שורות כאלה. התקרה היא ארבע פקודות במשפט, כי כל אחת עולה בערך שתי שניות וחצי.

## אפשרויות

**Settings ← Devices & services ← Needle Assist ← Configure**

שתי הגדרות, ולא יותר:

| הגדרה | ברירת מחדל | מה זה |
|---|---|---|
| **רמקול למוזיקה** | ריק | על איזה רמקול של Music Assistant לנגן כשלא נאמר חדר. מיותר אם יש רמקול אחד, או אם תמיד אומרים איפה |
| **מקסימום טוקנים לאמירה** | 192 | קריאת כלי היא כ‑25 טוקנים; השאר נועד למשפטים עם כמה פקודות. העלאה לא משפרת את איכות התשובה — נמדד: 14 מתוך 28 כישלונות ב‑192, ואותם 14 ב‑768 |

**מה שכבר לא ניתן לשינוי, בכוונה.** עד גרסה 1.2 היו כאן עוד שתי הגדרות — סירוב לאמירות שאינן קשורות לבית, ורף ביטחון מינימלי. לשתיהן יש ערך אחד נכון שנמדד, ולכן הן הפכו לקבועות בקוד:

- **הסירוב פעיל תמיד.** המודל עצמו מסרב ב‑0.5% מהמקרים ומפעיל מכשיר ב‑99.5% מהשאלות שאינן קשורות לבית. הכיבוי היחיד שההגדרה אפשרה היה לתת לשאלה על כדורגל להדליק אור.
- **רף הביטחון הוא 0.** הכיוונון לא מעדכן את ראש הביטחון של המנוע, וקריאות עברית נכונות נמדדות שם ב‑0.0. כל רף מעל אפס דחה פקודות תקינות.

**נתיב למודל משלך.** אם אימנת מודל בעצמך: `⋮ ← הגדרה מחדש`, ושם נתיב לקובץ ‎.cact. השדה ריק פירושו המודל שמגיע עם האינטגרציה — וזה גם הדרך לחזור אליו.

## למה זה טוב

- **לדבר לבית בעברית, בלי ענן.** מיקרופון מקומי, מנוע מקומי — אף משפט לא יוצא מהבית. זה מה שמאפשר להתקין את זה בבית עם ילדים בלי לחשוב על זה פעמיים.
- **עברית מדוברת, לא רק כתובה.** "תכבס", "שים על שקט", "תוריד קצת" — כולל מילים דבוקות ואותיות סופיות במקום הלא נכון, כמו שזיהוי דיבור מייצר.
- **כמה פעולות במשפט אחד.** "תכבה את האור בסלון ותנעל את הדלת".
- **מוזיקה בשם.** "תנגן את האלבום שבלול בסלון".
- **לענות על שאלות בלי להזיז כלום.** "מה המצב של האור במטבח".
- **מהיר.** כשתי שניות וחצי לפקודה על Home Assistant Green, בלי GPU ובלי חיבור לאינטרנט.

## מה אפשר להפעיל

אין כאן מכשירים משלה: היא מפעילה מה שכבר מותקן אצלך ב‑Home Assistant, דרך השירותים הרגילים. **42 כלים**, על התחומים האלה:

| תחום | מה אפשר לומר |
|---|---|
| `light` | להדליק, לכבות, בהירות, אחוזים, למעלה ולמטה, צבע בשם |
| `switch` | להדליק, לכבות |
| `climate` | להדליק, לכבות, טמפרטורה מוחלטת, מצב, עוצמת מאוורר |
| `cover` | לפתוח, לסגור, לעצור, אחוז פתיחה |
| `lock` | לנעול, לשחרר |
| `media_player` | לנגן, להשהות, ווליום, השתקה, שיר הבא, מקור |
| `music_assistant` | לנגן אלבום/פלייליסט/רדיו/אמן בשם |
| `vacuum` | להתחיל, לעצור, לחזור לתחנה, עוצמת שאיבה |
| `fan` | להדליק, לכבות, סיבוב |
| `camera` | לצלם תמונה |
| `scene` / `script` / `automation` | להפעיל לפי שם |
| `timer` | להפעיל, לבטל |
| `input_boolean` / `input_number` | להדליק, לכבות, לקבוע ערך |
| `notify` | לשלוח הודעה |
| שאלות | מצב של ישות, טמפרטורה, מזג אוויר |

**מה שלא נתמך בכוונה:** כל שירות שלא קיים ב‑Home Assistant. הדקדוק של המנוע בנוי מרשימת הכלים המוצהרת, כך שקריאה לכלי שלא הוצהר היא לא־ייצוגית — המודל לא *יכול* להמציא שירות.

## דוגמאות

**להריץ פקודה מאוטומציה** — למשל כפתור פיזי שאומר משפט:

```yaml
action:
  - action: conversation.process
    data:
      agent_id: conversation.needle_assist
      text: "תכבה את כל האורות ותנעל את הדלת"
```

**לבדוק מה האינטגרציה מבינה, בלי להפעיל כלום:**

```yaml
action:
  - action: conversation.process
    data:
      agent_id: conversation.needle_assist
      text: "מה המצב של האור במטבח"
    response_variable: answer
  - action: notify.persistent_notification
    data:
      message: "{{ answer.response.speech.plain.speech }}"
```

**להשתמש בה כעוזר הקולי של רמקול:** `Settings ← Voice assistants ← Add assistant`, ולבחור **Needle Assist** בתור Conversation agent. שפת ה‑pipeline צריכה להיות עברית.

## איך זה מתעדכן

אין כאן סקרים ואין רענון תקופתי. האינטגרציה קוראת את מצב הבית מהרישומים של Home Assistant **ברגע שמדברים אליה**, ולא לפני כן:

- **חדרים וישויות** נקראים מרישום האזורים ומרישום הישויות בכל אמירה. חדר שהוספת או כינוי ששינית תופסים מיד — האינדקס נבנה מחדש באירוע השינוי.
- **מצב מכשירים** נקרא מ‑`hass.states` באותו רגע, לצורך שאלות ולצורך פקודות יחסיות ("תעלה קצת" = המצב הנוכחי ועוד).
- **המודל** נטען פעם אחת, בהפעלת האינטגרציה. שינוי הגדרות טוען אותו מחדש.
- **ספריית המנוע** יורדת פעם אחת לכל גרסת מנוע, בהתקנה הראשונה. אחר כך אין שום קריאת רשת.

## הסרה

`Settings ← Devices & services ← Needle Assist ← ⋮ ← Delete`. אם התקנת דרך HACS, אפשר גם להסיר משם את המאגר.

מה שנשאר אחרי מחיקה, ואפשר למחוק ידנית:

- `/config/custom_components/needle_assist/` — האינטגרציה עצמה (HACS מוחקת אותה בשבילך).
- `/config/needle_assist_engine/` — ספריית המנוע שירדה. אפשר למחוק; היא תרד שוב בהתקנה הבאה.
- `/config/needle_he.cact` — רק אם העתקת לשם מודל בעצמך.

## מה נמדד

על **כל 2,379 השורות** של קבוצת בדיקה שלא נראתה באימון, מתוך 22,520 שורות שנוצרו:

| | |
|---|---|
| בחירת הכלי הנכון | **76.4%** |
| כלי **וגם** כל הארגומנטים | **61.7%** |
| משפט עם כמה פקודות — בחירת הכלים | **95.3%** (היה 0.0%) |
| משפט עם כמה פקודות — כלים וארגומנטים | **91.7%** |
| תיקון עצמי באמצע משפט ("לא לא, תעשה...") | **100%** |
| "בכל הבית" | **98.7%** |
| מהירות שאיבה, מצב מאוורר וצבע — נקראים מהמשפט | **808 / 0** |
| בקשות מוזיקה — הכלי הנכון | **88.6%** |
| בקשות מוזיקה — הכלי, השם והסוג | **72.4%** |
| נעילה ופתיחה — הכיוון הנכון | **95.9%** |
| הכיוון נקרא מהמילים: מסכים / סותר את התווית | **1337 / 0** |
| הסימן של שינוי יחסי: מסכים / סותר | **1222 / 1** |
| זיהוי החדר (בקוד, לא במודל) | **99.5%** |
| זיהוי חדר במילים נרדפות שלא נראו באימון | **100%** |
| ניתוב לכלי הנכון (רשימה מקוצרת) | 99.1% |
| שאלות שקיבלו כלים לקריאה בלבד, ולכן לא יכלו להפעיל כלום | **84.4%** |
| פקודות אמיתיות שסווגו בטעות כשאלה | **0 מתוך 19,158** |
| סירוב נכון לאמירות מחוץ לתחום | 75.0% |
| פקודות אמיתיות שנדחו בטעות | 1.87% |

**וכמה מזה הוא המודל?** אותם משקלים בדיוק, אותן שאלות, בלי השכבה
הדטרמיניסטית סביבם: **51.1%** בחירת כלי במקום 76.4%, **28.9%** התאמה מלאה
במקום 61.7%, ו‑**0.5%** סירוב נכון במקום 75.0%. קרוב לחצי ממה שהמערכת עושה
נכון, היא עושה מחוץ למודל.

**למה זיהוי החדר גבוה בהרבה מדיוק המודל?** כי הוא לא נעשה במודל. המודל מכיר 12 קודי חדר קבועים באנגלית; בית אמיתי לא מתחלק ל‑12. אז החדר, שם המכשיר, שם הסצנה ותוכן ההתראה נקראים מהמשפט בקוד דטרמיניסטי מול הרישומים האמיתיים של Home Assistant. אותה בחירה נעשתה גם לגבי בחירת הכלי ולגבי הסירוב: **כל מה שאפשר להכריע בוודאות, מוכרע מחוץ למודל.**

הסבר מלא: [`docs/HOW-IT-WORKS.md`](docs/HOW-IT-WORKS.md).

## מגבלות ידועות

הרשימה הזו כתובה במלואה בכוונה. עדיף לדעת מראש.

- **כל פקודה במשפט עולה זמן.** המשפט נחתך לפי פקודות והמודל רץ פעם אחת לכל אחת — כשתי שניות וחצי לפקודה על חומרת ARM. לכן יש תקרה של ארבע פקודות במשפט אחד.
- **חיתוך המשפט מסתמך על פועל.** "תדליק את האור בסלון, ובמטבח תסגור" — פועל שבא אחרי החדר במקום לפניו — לא תמיד ייחתך נכון. הניסוח הרגיל, פועל בתחילת כל פקודה, כן.
- **מוזיקה בשם דורשת [Music Assistant](https://www.music-assistant.io/).** בלעדיו "תנגן לי כוורת" פשוט ימשיך נגינה על הרמקול שבחדר, כי אין ספרייה לחפש בה. שם השיר נקרא מהמשפט — המודל לא ממציא שמות ולא "יודע" באיזה אלבום שיר נמצא; החיפוש עצמו נעשה על ידי Music Assistant.
- **שאלה בלי מילת שאלה לא תמיד מזוהה ככזו.** "התריס בסלון פתוח" בלי סימן שאלה עמום גם בעברית; כ‑18% מהשאלות נופלות לשם. הן לא יפעילו כלום בטעות, אבל הן עלולות לא לענות.
- **המודל עצמו כמעט לא מסרב** — שיעור הסירוב הנכון שלו נמדד ב‑0.5%, והוא מפעיל מכשיר על 99.5% מהשאלות שאינן קשורות לבית. הסירוב כולו נעשה בקוד, ותופס 75.0% מהאמירות שמחוץ לתחום; השאר יגיעו למודל.
- **רף הביטחון לא שמיש.** הכיוונון לא מעדכן את ראש הביטחון של המנוע, וקריאות נכונות שאינן באנגלית נמדדו עם ביטחון 0.0. כל רף מעל 0 ידחה פקודות תקינות — ולכן זה קבוע בקוד על 0 ולא הגדרה שאפשר לשנות.
- **טקסט חופשי בארגומנט נלקח מהמשפט, לא מהמודל.** עברית מגיעה לארגומנט של כלי כרצף בריחה של שישה תווים לאות, והמודל טועה בהם. גוף ההתראה נחתך מהמשפט עצמו; אם אי אפשר לחתוך אותו — ההתראה לא נשלחת, במקום לשלוח ג'יבריש.
- **ישות בלי אזור לא נגישה בפקודת חדר.** זו התנהגות של Home Assistant, לא באג כאן, אבל היא מפתיעה.
- **עברית בלבד.** האינטגרציה מצהירה `he` ולא `MATCH_ALL`, כדי ש‑Assist לא ינתב אליה אנגלית.

## פרטיות

- אין שום קריאת רשת בזמן שיחה. המשפטים שלך לא יוצאים מהמכונה.
- הדבר היחיד שיורד מהאינטרנט הוא ספריית המנוע, פעם אחת, בהתקנה הראשונה, מ‑Hugging Face. אפשר גם להעתיק אותה ידנית ולהישאר מנותק לגמרי — ראה `engine_lib.py`.
- אין טלמטריה, אין דיווח שימוש, אין מפתחות API.

## בעיות

`Settings ← Devices & services ← Needle Assist ← ⋮ ← Download diagnostics` מייצר דוח עם מה שהאינטגרציה רואה: איזה מודל נטען, אילו חדרים היא זיהתה ובאילו מילים אפשר לפנות לכל אחד, כמה ישויות ברות‑פנייה, ומצב השערים. הדוח לא מכיל אף משפט שאמרת. זה המקום להתחיל.

| מה קורה | למה, בדרך כלל |
|---|---|
| **"לא הבנתי" על כל דבר** | ה‑pipeline לא מוגדר לעברית, או שנבחר עוזר שיחה אחר. `Settings ← Voice assistants` |
| **פקודה תקינה לא מוצאת כלום** | לישות אין אזור. `Settings ← Devices & services ← Entities`, לסנן לפי ישויות בלי אזור |
| **חדר מסוים אף פעם לא נתפס** | השם שאתם אומרים אינו שם האזור. להוסיף אותו כ‑alias לאזור; בדוח האבחון יש `reachable_by` לכל חדר, שמראה בדיוק אילו מילים מגיעות אליו |
| **"תנגן X" רק ממשיך נגינה** | אין Music Assistant מותקן, או שהרמקול שלו אינו באזור שנאמר |
| **האינטגרציה לא עולה אחרי עדכון** | דוח תיקון בשם "לא מוצא את קובץ המודל" — הרשומה מצביעה על קובץ שנמחק. `⋮ ← הגדרה מחדש`, ולרוקן את השדה |
| **הפעלה ראשונה נכשלת בלי רשת** | ספריית המנוע יורדת פעם אחת. הודעת השגיאה כוללת את הכתובת ואת הנתיב, כדי להתקין ידנית |
| **תשובה איטית מאוד** | כל פקודה במשפט היא ריצה נפרדת של המודל. משפט עם ארבע פקודות לוקח פי ארבעה |

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

- **42 tools**, every one a real Home Assistant service: lights, covers, climate, locks, media, music, switches, vacuum, cameras, scenes, scripts, automations, timers, notifications and state queries.
- **Knows your rooms**, not a fixed list. A house with a "הול" or a "חדר כביסה" works, aliases included.
- **Handles spoken Hebrew**: prefix particles, misplaced final letters, words glued together by speech-to-text, and typos.
- **Several orders in one sentence.** "Turn off the kitchen light and close the bedroom blinds" is two commands and both run. The sentence is cut before the model sees it, and only where a new order really begins — "turn on the light in the living room and the kitchen" stays one order over two rooms.
- **Plays music by name** through [Music Assistant](https://www.music-assistant.io/): the title, the artist and the kind are read out of the sentence, never invented by the model. Without Music Assistant installed it simply resumes playback on the room's speaker.
- **A question cannot actuate.** Asking the state of a light answers; it does not switch it.
- **The Hebrew verb decides the direction.** "נעל את הדלת" locks and "תנמיך את המזגן" lowers. Every model inverts a command sometimes, and that is the worst mistake a home assistant can make, so the words overrule it. Measured over the whole held-out set: **1337 times the words agree with the label, 0 times they contradict it.**
- **Refuses** what is not about the house, and refuses negated commands.

## Requirements

Home Assistant 2024.11+, x86_64 or aarch64, glibc or musl, ~40MB of disk. No GPU. Verified on Home Assistant OS 2026.8 on a Home Assistant Green (aarch64 + musl).

## Install

Add `https://github.com/Yosef-Chai/needle-assist-he` as a HACS custom repository (category: Integration), install, restart, then **Settings → Devices & services → Add integration → Needle Assist (Hebrew) → Submit**. Nothing to fill in — the tuned model ships with the integration. Finally pick it as your conversation agent under **Settings → Voice assistants**.

**Assign an area to every entity.** An entity with no area cannot be reached by any room command. That is Home Assistant's own targeting, but it surprises people.

## Use cases

- **Speak Hebrew to the house, with nothing in the cloud.** Local microphone, local engine; no sentence leaves the building.
- **Spoken Hebrew, not only written Hebrew** - glued words and misplaced final letters, the way speech-to-text produces them.
- **Several orders in one sentence** - "turn off the light in the living room and lock the door".
- **Music by name** - "play the album Shablul in the living room".
- **Questions that move nothing** - "is the light on in the kitchen".
- **Fast**: about 2.5 s per order on a Home Assistant Green, with no GPU and no network.

## Supported devices

The integration owns no devices. It drives what Home Assistant already has, through the ordinary services - **42 tools** across these domains:

`light`, `switch`, `climate`, `cover`, `lock`, `media_player`, `music_assistant`, `vacuum`, `fan`, `camera`, `scene`, `script`, `automation`, `timer`, `input_boolean`, `input_number`, `notify`, plus read-only questions about entity state, temperature and weather.

Anything Home Assistant can do that is not in that list is *not* reachable, and cannot be reached by accident: the engine's grammar is compiled from the declared tool schemas, so a service that was never declared is not representable in the model's output.

## Supported functions

| Domain | What can be said |
|---|---|
| `light` | on, off, brightness, percentages, up and down, colour by name |
| `climate` | on, off, absolute temperature, mode, fan speed |
| `cover` | open, close, stop, position |
| `media_player` | play, pause, volume, mute, next track, source |
| `music_assistant` | play an album, playlist, radio station or artist by name |
| `vacuum` | start, stop, return to dock, suction level |
| `scene` / `script` / `automation` | run by name |
| `timer` | start, cancel |
| `notify` | send a message, whose text is taken from the sentence |

## Examples

Run a sentence from an automation - a physical button that speaks one, for instance:

```yaml
action:
  - action: conversation.process
    data:
      agent_id: conversation.needle_assist
      text: "תכבה את כל האורות ותנעל את הדלת"
```

Ask it something and use the answer, without actuating anything:

```yaml
action:
  - action: conversation.process
    data:
      agent_id: conversation.needle_assist
      text: "מה המצב של האור במטבח"
    response_variable: answer
  - action: notify.persistent_notification
    data:
      message: "{{ answer.response.speech.plain.speech }}"
```

To use it as a voice satellite's assistant, pick **Needle Assist** as the conversation agent under **Settings → Voice assistants**, with the pipeline language set to Hebrew.

## How data is updated

Nothing is polled and nothing is refreshed on a schedule. The house is read when the agent is spoken to and not before:

- **Rooms and entities** come from the area and entity registries on each utterance; the compiled phrase index is invalidated by the registry-updated events, so a renamed room or a new alias takes effect immediately.
- **Device state** is read from `hass.states` at that moment - for questions, and for relative orders, which are the current reading plus a step.
- **The model** is loaded once at setup, and reloaded when an option changes.
- **The engine's native library** is fetched once per engine version, on the first run, through Home Assistant's shared HTTP session. After that there is no network call at all.

## Removal

**Settings → Devices & services → Needle Assist → ⋮ → Delete**, then remove the repository from HACS if it was installed that way.

Deleting the entry leaves three things on disk, all safe to remove by hand:

- `/config/custom_components/needle_assist/` - the integration (HACS removes this for you).
- `/config/needle_assist_engine/` - the downloaded engine; it is fetched again if the integration is reinstalled.
- `/config/needle_he.cact` - only if you copied a model there yourself.

## Troubleshooting

**Settings → Devices & services → Needle Assist → ⋮ → Download diagnostics** reports which model loaded, every area with the exact phrases that reach it, how many entities are addressable, and the gate thresholds. It contains no utterance of yours. Start there.

| Symptom | Usually |
|---|---|
| Everything answers "I did not understand" | The pipeline is not set to Hebrew, or another conversation agent is selected |
| A valid order finds nothing | The entity has no area. Filter for area-less entities under Settings → Devices & services → Entities |
| One room is never recognised | The word you say is not the area's name. Add it as an area alias; `reachable_by` in the diagnostics shows exactly which words reach each room |
| "Play X" only resumes playback | Music Assistant is not installed, or its player is not in the room that was named |
| It stops loading after an update | A repair issue named "cannot find its model file" - the entry points at a file that was deleted. Use ⋮ → Reconfigure and clear the field |
| First start fails with no network | The engine library is fetched once; the error names the URL and the path so it can be installed by hand |
| Very slow answers | Each order in a sentence is a separate run of the model; four orders take four times as long |

## Measured

On **all 2,379 rows** of the held-out test set, out of 22,520 generated:

| | |
|---|---|
| correct tool | **76.4%** |
| tool **and** every argument | **61.7%** |
| multi-order sentences, correct tool set | **95.3%** (was 0.0%) |
| multi-order sentences, tools and arguments | **91.7%** |
| mid-sentence self-correction ("no no, do...") | **100%** |
| "the whole house" | **98.7%** |
| fan speed, fan mode and colour, read from the sentence | **808 / 0** |
| music requests, correct tool | **88.6%** |
| music requests, tool, title and kind | **72.4%** |
| lock and unlock, the right direction | **95.9%** |
| direction read from the words: agrees / contradicts the label | **1337 / 0** |
| sign of a relative change: agrees / contradicts | **1222 / 1** |
| room resolution (in code, not in the model) | **99.5%** |
| room resolution, synonyms never seen in training | **100%** |
| tool shortlist recall | 99.1% |
| questions given read-only tools only, so they cannot actuate | **84.4%** |
| real commands wrongly classified as questions | **0 of 19,158** |
| off-topic correctly refused | 75.0% |
| real commands wrongly refused | 1.87% |

**And how much of that is the model?** The same weights, the same questions,
without the deterministic layer around them: **51.1%** correct tool instead of
76.4%, **28.9%** exact instead of 61.7%, and **0.5%** correct refusal instead
of 75.0%. Nearly half of what this system gets right, it gets right outside the
model.

Room resolution beats the model's own accuracy because it is not done by the model. The model knows twelve fixed English room slugs; a real house does not partition into twelve. So the room, the device name, the scene name and the notification text are read out of the sentence by deterministic code against Home Assistant's own registries. The same choice was made for tool selection and for refusal: **anything that can be decided with certainty is decided outside the model.**

See [`docs/HOW-IT-WORKS.md`](docs/HOW-IT-WORKS.md).

## Known limitations

- **Each order in a sentence costs time.** The sentence is cut into orders and the model runs once per order — about 2.5 s each on ARM — so there is a cap of four.
- **Cutting the sentence relies on a verb.** An order whose verb comes after the room rather than before it may not be separated. The ordinary phrasing, a verb at the head of each order, is.
- **Playing music by name needs [Music Assistant](https://www.music-assistant.io/).** Without it, a request to play something resumes the room's speaker instead, because there is no library to search. The title comes out of the sentence; nothing here tries to *know* which album a song is on — Music Assistant does the lookup.
- **A question with no interrogative** ("the blind in the living room [is] open") is not always recognised as one. It cannot actuate by accident, but it may not answer.
- **The model itself almost never refuses anything** — 0.5% correct refusal measured, and 99.5% false actuation. Refusal is done in code and catches 75.0% of off-topic utterances; the rest reach the model.
- **The confidence threshold is unusable.** Fine-tuning does not update the engine's confidence head and correct non-English calls measure 0.0, so any threshold above 0 rejects valid commands. It is a constant of 0 rather than a setting, for that reason.
- **Free-text arguments come from the sentence, not the model.** Hebrew reaches a tool argument as six-character escapes and the model gets them wrong.
- **Hebrew only.** The integration declares `he` rather than `MATCH_ALL`.

## Licence

Code MIT. The bundled model is a fine-tuned derivative of [Cactus-Compute/needle2](https://huggingface.co/Cactus-Compute/needle2), Apache-2.0. Engine by [Cactus Compute](https://github.com/cactus-compute/needle), MIT. Full detail in [`NOTICE`](NOTICE).
