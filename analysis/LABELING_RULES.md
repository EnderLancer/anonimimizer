# Ukrainian court-decision depersonalization: rules for ОСОБА / АДРЕСА / НОМЕР / ІНФОРМАЦІЯ

The labels in this challenge are the ones used by the **Єдиний державний реєстр судових рішень (ЄДРСР)**. Below are (1) the legal sources that define them, (2) the official rule for each label, and (3) what the 240 gold documents actually do, with counts, including every exception found.

**How well each statement is verified.** The cloud environment's network policy blocks zakon.rada.gov.ua, court.gov.ua, protocol.ua, kodeksy.com.ua and ligazakon, so primary texts could not be opened directly. Legal statements come from search-engine excerpts of those pages and are marked:
- ✅ quoted in an excerpt of the primary text;
- 🟡 reported by a secondary source (news, commentary), or the excerpts disagree;
- ❓ not verified.

Statements about the gold data (📊) are measured on `data/test/court_cases.jsonl` and are exact.

---

## 1. Legal sources

| # | Act | What it says about masking | Status |
|---|---|---|---|
| 1 | **Закон України «Про доступ до судових рішень» № 3262-IV (22.12.2005), ст. 7** «Відомості, що не можуть бути розголошені в текстах судових рішень, відкритих для загального доступу» | The list of what must be hidden. Hidden data **«замінюються літерними або цифровими позначеннями»** | ✅ |
| 1a | ст. 7 ч. 1 (current wording) | must not be disclosed: (1) **place of residence or stay of natural persons with the address, phone numbers or other means of communication, e-mail, РНОКПП (taxpayer number), details of identity documents, unique record numbers in the Єдиний державний демографічний реєстр**; (2) **vehicle registration numbers**; (3) **bank account and payment card numbers**; (4) **information protected by a closed (закрите) court hearing** | ✅ (items 1–3 quoted; item 4 in two excerpts) |
| 1b | ст. 7 (older/other wording, cited by commentaries) | «відомості, що дають можливість ідентифікувати фізичну особу»: **names (ім'я, по батькові, прізвище) of natural persons**, address/contacts/IDs, vehicle numbers, **«інші відомості, що дають можливість ідентифікувати фізичну особу»** | 🟡 Excerpts disagree on whether names are still in the current list. A 2017 amendment reworded the article (DEJURE: «з грудня 2017 року в ст. 7 відсутній п. 4»). Check the current edition on zakon.rada.gov.ua |
| 1c | ст. 7, exceptions (not treated as protected) | **surnames and initials of the judges who made the decision**; **names of officials and civil servants acting in their official capacity**; **names of the parties to a case decided by an international court (e.g. ЄСПЛ) that the Ukrainian decision cites** (added by Закон № 4779-VI of 17.05.2012) | ✅ judges and international-court parties; 🟡 officials (cited by Пленум ВАСУ, постанова 29.09.2016 № 10) |
| 1d | ст. 7 ч. 2 | in criminal cases, for safety reasons, the names of judges and other participants **may be omitted** | 🟡 |
| 2 | **Порядок ведення ЄДРСР**, рішення ВРП від 19.04.2018 № 1200/0/15-18 (amended, e.g. № 2364/0/15-21). It replaced постанова КМУ № 740 of 25.05.2006, which lost force on 01.10.2018 | **знеособлення** = masking non-disclosable data in the electronic copy «за допомогою спеціалізованого програмного забезпечення з подальшою вибірковою візуальною перевіркою». Documents are masked automatically within 3 working days and published by the next working day | ✅ |
| 3 | **Кодифікатор відомостей, що не можуть бути розголошені в текстах судових рішень** (наказ ДСА від 13.06.2006 № 59; copies on ki.court.gov.ua/tu11/14/36, lg.court.gov.ua/tu13/inshe/10/1) | defines the **four label words and their numbering**, plus an annex listing the participants it applies to | ✅ for the rules quoted in §2; the annex was not readable |
| 4 | **Інструкція … копій судових рішень, що підлягають внесенню до ЄДРСР** (наказ ДСА від 14.05.2008 № 37), п. 1.10 | the substitution is done by the registry's technical administrator (now ДП «Інформаційні судові системи») | ✅ |
| 5 | Practice: **ДП «ІСС»** | Since the end of 2017 masking has been done by ІСС's own **AI module**, with 30–60 thousand documents a day and **selective** human checking. The previous (Russian-developed) module was estimated at 20–30% effectiveness. Press reports describe the module masking even the names of Grand Chamber judges in dissenting opinions | 🟡 |
| 6 | Draft law **№ 7033-д** (passed at first reading 23.05.2024) | would let judges remove more information and **automatically remove names and addresses of public institutions**. Status after first reading unknown | 🟡 not law |

**Consequence for this dataset.** The `pseudonymized` field of the test set is ЄДРСР output, so the gold labels are the output of ІСС's automatic masker with spot checks. They are not hand-made by one annotator following the codifier. This explains why the gold sometimes breaks the official rules (judges masked, e-mail as ІНФОРМАЦІЯ, state bodies masked in some documents only; see §3). The `original` field looks reconstructed: the "real" entities are synthetic (ISO dates «2017-10-28», «Військова частина А1234», «UA2889999803…», «DF987654»).

---

## 2. Official rule per label (codifier and law)

All four labels share one format: **`ЛЕЙБЛ_N`**, where **N «відповідає черговості появи цієї відомості у тексті судового рішення»** (order of first appearance) ✅. The same piece of information gets the same number on every repeat (codifier example: «Іванов Іван Іванович» → «ОСОБА_1») ✅.

| Label | Official scope | Codifier example | Status |
|---|---|---|---|
| **ОСОБА_N** | name (ім'я, по батькові, прізвище) of a natural person who is a participant listed in the codifier annex (позивач, відповідач, третя особа, заявник, цивільний позивач, представник, свідок, поручитель, потерпілий, …) | «Іванов Іван Іванович» → «ОСОБА_1» | ✅ example; 🟡 annex |
| **АДРЕСА_N** | «адреса місця проживання або перебування фізичних осіб» | street, building, flat, city → «АДРЕСА_1» | ✅ |
| **НОМЕР_N** | phone number, **e-mail address**, vehicle registration number, identification number (РНОКПП), identity-document number, **bank account / card number**, any document number that identifies a natural person | phone «222-22-22» → «НОМЕР_1» | ✅ |
| **ІНФОРМАЦІЯ_N** | «інша інформація, що дає можливість ідентифікувати фізичну особу» | author of a work «ТВІР» → «ІНФОРМАЦІЯ_1» | ✅ |

**Not masked, according to the law:**
- the judges who made the decision (surname + initials);
- officials and civil servants acting in office (prosecutors, state bodies' representatives);
- parties of cited international-court cases (ЄСПЛ);
- **legal entities and their codes/addresses**, because ст. 7 protects natural persons only (🟡: consistent across sources, no explicit quote);
- case numbers, court names, dates of procedural acts.

---

## 3. What the gold actually does (📊), label by label

### ОСОБА: 2 826 mentions, ≈862 people

**Forms that are masked:**

| form | count | example |
|---|---|---|
| full ПІБ in any case form | 2 412 | «Попова Сергія Олексійовича» |
| Прізвище І. П. | 385 | «Писаковська О. І.», «Ангелін І.І.» |
| І. П. Прізвище | 18 | «О. В. Павлов», «А.Ясагової», «В. Зуєв» |
| surname only | 6 | «Шевченко», «Шидерової» |
| two words (name + surname, foreign) | 4 | «Кудсіє Філіпп», «Аліни Степанової» |

**Who is masked, against the law's exceptions:**

| role (by the word before the name) | masked | left open |
|---|---|---|
| суддя / головуючий / склад колегії | **67** | 0 |
| адвокат | 82 | 0 |
| нотаріус | 48 | 0 |
| державний / приватний виконавець | 14 | 0 |
| представник | 6 | 0 |

**Exception 1. Judges are masked.** This breaks ст. 7 (judges' surnames and initials are not protected) but is what the ЄДРСР masker does. Predict them as ОСОБА.

**Exception 2. What is never masked:**
- ЄСПЛ case names («Пономарьов проти України», «Brualla Gomez de la Torre v. Spain»). This follows ст. 7 / Закон 4779-VI;
- honorific names inside institution titles («імені … полковника Євгенія Пікуса»);
- «Верховна Рада» and other bodies;
- street names («вул. Тараса Шевченка» stays inside АДРЕСА).

**Span boundaries:**
- role words are outside the span («Головуючий [В. Зуєв]», «адвокат [Мастістий І. А.]»). There is one gold exception (116234574, which counts as a leak);
- an initial's dot is inside the span, except where the gold drops a dot that also ends the sentence (36 vs 1, per README). The gold is inconsistent at line breaks: «Фінагеєва В. О.\n» keeps the dot.

**Numbering:**
- one number per person across all inflections («Корзуна / Корзуном / Корзуну Ігорю» → ОСОБА_1);
- across full name and initials: 79 of 563 multi-mention persons mix «Прізвище Ім'я По-батькові» and «Прізвище І. П.» under one number;
- order: **not strictly by appearance in this text.** In 131 of 240 documents the numbers rise in order of first appearance; 19 more fit if ordered by first full-name mention; 90 do not fit either way (e.g. 116175256 starts at ОСОБА_3). The likely reason is that numbering was done on the full original decision, and the excerpt we get is shorter or reordered. The metric ignores the numbers (only B³ consistency counts), so only the grouping matters.

**Known gold errors** (from `analysis/README.md`):
- full names labelled ІНФОРМАЦІЯ (116186753, 116158320);
- one label for two people (116206025 Тимченко / Тимошенко);
- two labels for one person (116234778 Гварамадзе, 116205945 Підлубний).

### АДРЕСА: 233 mentions

- **Rule.** Mask an address at street level («вул./просп./пров./в'їзд …, буд., кв.») or village/settlement level for land plots («село Лука-Мелешківська, Вінницький район, Вінницька область»). **A bare city or region is never masked** (standalone «м. Київ» never appears as a gold span).
- **Composition:**
  - with flat number 108;
  - with postal index 62;
  - with neither 82;
  - the span runs from the street (or index) to the end of the address, including city, district, oblast and index.
- **Contexts:** «за адресою(:/ –)» 86, objects of property («квартира / житловий будинок / земельна ділянка / машиномісце / паркінг по …») ≈ 100, residence («проживає / зареєстрований»).
- **Not only natural persons' residence** (wider than the codifier):
  - addresses of property under dispute are masked;
  - addresses of **courts** («Верховного Суду за адресою: м. Київ, вул. …») and **public institutions / legal entities with ЄДРПОУ** are not masked.
- **Exceptions and inconsistencies:**
  - a leading preposition «по / на» is inside the span 17 times and outside 29 times;
  - two addresses are labelled ІНФОРМАЦІЯ (116234614);
  - one crime-scene address is left open (116258697).

### НОМЕР: 447 mentions

| kind | count | example | note |
|---|---|---|---|
| **military unit number** | **314** | «А4412», «В0540», «ВЧ40238», «А10257» | **the largest group.** Not in the codifier (not a natural person's data), but always masked. «в/ч» / «військова частина» before it is usually outside the span (included 7 times as «в/ч А7219») |
| enforcement-proceeding (виконавче провадження) number | ≈45 | «ОК/3072», «№ 62354178», «543/2022/05» | «№» usually outside (inside 5 times) |
| IBAN / bank account | 16 | «UA4732212800…» | the **treasury / institution IBAN is left open** (masked 1 of 15) |
| РНОКПП / ІПН / «ідентифікаційний номер» | 13 | «8154027698» | |
| identity documents and certificates: passport, ID card, birth/marriage certificate, diploma, workbook, veteran ID | ≈12 | «КВ123456», «FP874563», «МН122214», «СА-452581», «248765432» | |
| vehicles: plate, body/chassis VIN, factory number | ≈14 | «AB-8462-CH», «Y6LRT72E08A123456», «НА394668» | VIN and factory numbers are masked too |
| real-estate registration number, land plot | 3 | «420644751101», «ділянка № 384551» | |
| **legal-entity codes** (ЄДРПОУ / «ідентифікаційний код юридичної особи», «(код …)» of a military unit) | 3 | «41862436», «20635749», «09206351» | against the natural-persons-only rule |
| weapon serial numbers | 5 | «НГ457213», «МС167432» | |
| garage / parking space number | 3 | «ВН8412АН», «АН8619КА», «№ 341» | short numbers like «машино-місце № 213» elsewhere are left open |

**Never masked:**
- case numbers («№ 756/7462/21»);
- decision / order numbers of courts and bodies («№476», «№ 65/2022»);
- dates;
- amounts;
- article numbers;
- court fee payment IDs.

**Contradicts the codifier:** **e-mail is never НОМЕР in the gold** (5 of 5 are ІНФОРМАЦІЯ; see below).

### ІНФОРМАЦІЯ: 180 mentions, the least consistent label

| kind | count | rule in the gold | example |
|---|---|---|---|
| **date of birth** | ≈53 | right after a person's name, with or without «року народження» (the words are inside the span when present) | «Сітник Олег Леонідович, [1981-03-20]», «[12 січня 1982 року народження]» |
| **date of death** | ≈24 | after «помер/померла/померлого», «після смерті …» | «яка померла [2015-03-18]», «[12 серпня 2023 року] Мельник Ірина Борисівна померла» |
| **state bodies and military formations** | 72 | masked **when they are a party** (позивач / відповідач / скаржник), and **only in some documents**: e.g. «Міністерство оборони України» 25 of 33, ГУ Нацполіції, ГУ ДПС, ТЦК та СП, ВЛК, «Військовий навчальний центр "Січ"». The same kind of body is left open in other documents (64 such FP in our run) | «до [Міністерства оборони України]» |
| **private companies** | 12 | masked in **2 documents only** («ТОВ "Комтех Логістік"», «ПрАТ "Дельта-Пром"», «Експрес-Сервіс»); open everywhere else | |
| **e-mail** | 5 | always ІНФОРМАЦІЯ (codifier says НОМЕР) | «natalia.solovyova@gmail.com» |
| «Військова частина А1234» as a whole | 4 | normally only «А1234» as НОМЕР | |
| gold errors | 12 | full names (10), addresses (2) | |

**Not masked as ІНФОРМАЦІЯ:**
- dates of procedural events (decision dates, «19 червня 2023 року ухвалою …»);
- a second date in the same sentence is sometimes left open («яка народилася [1952-11-12], та померла 2022-05-08»).

---

## 4. Quick decision table for the pipeline

| text | label | confidence |
|---|---|---|
| ПІБ / initials of any natural person, **including judges, lawyers, notaries, enforcement officers** | ОСОБА | high |
| ЄСПЛ case names, names in institution titles, historical names in street names | — | high |
| residence or property address down to street / village | АДРЕСА | high |
| court address, institution address, bare city | — | high |
| military unit code (А/В/ВЧ + digits) | НОМЕР (without «в/ч») | high |
| РНОКПП, passport, certificates, plates, VIN, private IBAN, enforcement-proceeding number, real-estate registration number | НОМЕР | high |
| treasury / institution IBAN, case numbers, decision numbers | — | high |
| e-mail | **ІНФОРМАЦІЯ** | high |
| date right after a person's name, or after «помер/народився» | ІНФОРМАЦІЯ | high |
| state body as a party | ІНФОРМАЦІЯ | **coin flip** (54 TP vs 64 FP in our run) |
| private company name | — (masked in 2 of ~41 documents) | medium |

## Sources

- Закон «Про доступ до судових рішень», ст. 7: [zakon.rada.gov.ua/go/3262-15](https://zakon.rada.gov.ua/go/3262-15), [protocol.ua](https://protocol.ua/ua/pro_dostup_do_sudovih_rishen_stattya_7/), [kodeksy.com.ua](https://kodeksy.com.ua/pro_dostup_do_sudovih_rishen/7.htm), [cn.court.gov.ua](https://cn.court.gov.ua/tu25/7/321)
- Кодифікатор відомостей (наказ ДСА № 59 від 13.06.2006): [ki.court.gov.ua/tu11/14/36](https://ki.court.gov.ua/tu11/14/36), [lg.court.gov.ua/tu13/inshe/10/1](https://lg.court.gov.ua/tu13/inshe/10/1)
- Інструкція (наказ ДСА № 37 від 14.05.2008): [zakon.rada.gov.ua/rada/show/v0037750-08](https://zakon.rada.gov.ua/rada/show/v0037750-08), [cn.court.gov.ua/tu25/7/326](https://cn.court.gov.ua/tu25/7/326)
- Порядок ведення ЄДРСР (рішення ВРП № 1200/0/15-18): [zakon.rada.gov.ua/go/v1200910-18](https://zakon.rada.gov.ua/go/v1200910-18); amendments [v2364910-21](https://zakon.rada.gov.ua/go/v2364910-21); old КМУ № 740: [zakon.rada.gov.ua/laws/show/740-2006-п](https://zakon.rada.gov.ua/laws/show/740-2006-%D0%BF)
- Names of international-court parties (Закон 4779-VI): [jurliga.ligazakon.net](https://jurliga.ligazakon.net/news/66309_znyata-zaborona-na-vkazvku-men-storn-v-tekstakh-rshen-mzhnarodnikh-sudv), [court.gov.ua/archive/16231](https://court.gov.ua/archive/16231/)
- ДП «ІСС» automatic masking: [court.gov.ua/press/news/798803](https://court.gov.ua/press/news/798803/)
- Judges not confidential (Пленум ВАСУ № 10, 2016): [nads.gov.ua PDF](https://nads.gov.ua/storage/app/sites/5/Dostup%20do%20publichnoi%20informacii/metod%20materiali/sudova-praktika.pdf); DEJURE on ст. 7: [dejure.foundation](https://dejure.foundation/chomu-znykayut-sudovi-rishennia/)
- Draft law 7033-д: [zmina.ua PDF](https://zmina.ua/wp-content/uploads/sites/2/2024/11/7033-d-dopovnennya-argumentacziyi-red.pdf)
