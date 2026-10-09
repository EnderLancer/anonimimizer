# Ukrainian court-decision depersonalization: rules for ОСОБА / АДРЕСА / НОМЕР / ІНФОРМАЦІЯ

The labels in this challenge are the ones used by the **Єдиний державний реєстр судових рішень (ЄДРСР)**. Below are (1) the legal sources that define them, quoted from the primary texts, (2) the official rule for each label, and (3) what the 240 gold documents actually do, with counts, including every exception found.

**How well each statement is verified.**
- ✅ quoted from the primary text (downloaded 08.10.2026; verbatim copies in [`legal/`](legal/));
- 🟡 secondary source only (news, commentary);
- 📊 measured on `data/test/court_cases.jsonl` (exact).

---

## 1. Legal sources (all ✅ unless marked)

| # | Act | Status on 08.10.2026 | What it contributes |
|---|---|---|---|
| 1 | **Закон «Про доступ до судових рішень» № 3262-IV, ст. 7.** Wording of Закон № 2147-VIII (03.10.2017, in force 15.12.2017); ч. 3 added by Закон № 361-IX (06.12.2019). Text: [`legal/ARTICLE_7_CURRENT.txt`](legal/ARTICLE_7_CURRENT.txt) | in force | what must be hidden (see below) |
| 2 | **Same article, 2012–2017 wording.** Text: [`legal/ARTICLE_7_2012_2017.txt`](legal/ARTICLE_7_2012_2017.txt) | replaced 15.12.2017 | the origin of the name masking and of the judge / official / international-court exceptions |
| 3 | **Кодифікатор відомостей, що не можуть бути розголошені в текстах судових рішень** (наказ ДСА від 13.06.2006 № 59). Text: [`legal/CODIFIER_DSA_59_2006.txt`](legal/CODIFIER_DSA_59_2006.txt) | **in force** («поточна редакція» on zakon.rada.gov.ua) | **defines the four labels, the numbering, the official-person exception, and the 59 participant roles whose data is masked** |
| 4 | **Порядок ведення ЄДРСР** (рішення ВРП від 19.04.2018 № 1200/0/15-18; last amended 03.09.2026, № 1807/0/15-26, about access, not masking). Excerpts: [`legal/PORYADOK_VRP_1200_2018_excerpts.txt`](legal/PORYADOK_VRP_1200_2018_excerpts.txt), from protocol.ua's copy of the 2018 text (the Rada site hosts only the cover decision) | in force | how masking is done in practice |
| 5 | Інструкція … копій судових рішень (наказ ДСА 14.05.2008 № 37), п. 2.8–2.9 | **cancelled** 10.02.2009 (наказ № 17) | repeated the codifier word for word; history only |
| 6 | 🟡 Practice: **ДП «Інформаційні судові системи» (ІСС)** | — | Since the end of 2017, masking has been done by ІСС's own **AI module** (30–60 thousand documents a day). The previous module was rated at 20–30% effectiveness. Press reports say it masks even Grand Chamber judges' names in dissenting opinions |
| 7 | 🟡 Draft law № 7033-д (first reading 23.05.2024) | not law | would also strip names and addresses of public institutions |

### 1.1 ст. 7, current wording (verbatim)

> 1. У текстах судових рішень, що відкриті для загального доступу, не можуть бути розголошені такі відомості:
> 1) місце проживання або перебування фізичних осіб із зазначенням адреси, номери телефонів чи інших засобів зв'язку, адреси електронної пошти, реєстраційні номери облікової картки платника податків, реквізити документів, що посвідчують особу, унікальні номери запису в Єдиному державному демографічному реєстрі;
> 2) реєстраційні номери транспортних засобів;
> 3) номери банківських рахунків, номери платіжних карток;
> 4) інформація, для забезпечення захисту якої розгляд справи або вчинення окремих процесуальних дій відбувалися у закритому судовому засіданні;
> 5) інші відомості, що дають можливість ідентифікувати фізичну особу.
> Такі відомості замінюються літерними або цифровими позначеннями.
> 2. З мотивів безпеки у текстах судових рішень у кримінальних справах … можуть не зазначатися відомості щодо імен (ім'я, по батькові, прізвище) суддів, які розглядали судову справу, та учасників судового процесу.
> 3. … у текстах ухвал про тимчасовий доступ до речей та документів у кримінальних справах … не зазначаються відомості, що дають можливість ідентифікувати осіб, які зазначені в таких ухвалах як володільці речей або документів.

What the 2017 rewrite changed compared with the 2012–2017 wording:
- **Names are no longer listed explicitly.** Before, item 1 was «імена (ім'я, по батькові, прізвище) фізичних осіб». Now names are covered only by the catch-all «5) інші відомості, що дають можливість ідентифікувати фізичну особу».
- **Bank accounts and payment cards were added** (item 3). So was the closed-hearing item, which was a separate part before.
- **All three exceptions were removed from the law:**
  - «прізвища та ініціали суддів, які ухвалили судове рішення»;
  - «імена посадових чи службових осіб, які, виконуючи свої повноваження, беруть участь у … справах»;
  - «імена сторін у справі, що розглядалася міжнародною судовою чи іншою міжнародною установою, на рішення якої міститься посилання» (the ЄСПЛ exception, added 2012).

  Only the officials exception survives, and only in the codifier (below).

### 1.2 The codifier (in force; verbatim, abridged)

> Відомості, що дають можливість ідентифікувати фізичну особу … замінюються літерним або цифровим позначенням …
> 1. Перелік фізичних осіб, відомості щодо яких не можуть бути розголошені … визначено в додатку 1.
> 2. **Імена (прізвище, ім'я, по батькові) фізичних осіб, за винятком імен посадових чи службових осіб, які, виконуючи свої повноваження, беруть участь у цивільній, господарській, адміністративній чи кримінальній справах, справах про адміністративні правопорушення (проступки)**, замінюються літерним позначенням "ОСОБА" та цифровим позначенням, що відповідає черговості появи цієї відомості у тексті судового рішення. Наприклад: "Іванов Іван Іванович" замінюється на: "ОСОБА_1".
> 3. **Адреса місця проживання або перебування фізичних осіб** замінюється … "АДРЕСА" … Наприклад: "Вул. Київська, кв. 1, буд. 1, м. Київ, 01001" замінюється на: "АДРЕСА_1".
> 4. **Номер телефону чи іншого засобу зв'язку, адреса електронної пошти, реєстраційний номер транспортного засобу, ідентифікаційний номер або номер документа, що дає можливість ідентифікувати фізичну особу**, замінюються … "НОМЕР" … Наприклад: Телефонний номер "222-22-22" замінюється на: телефонний номер "НОМЕР_1".
> 5. **Інша інформація, що дає можливість ідентифікувати фізичну особу**, замінюється … "ІНФОРМАЦІЯ" … Наприклад: Автор твору "ТВІР" замінюється на: автор твору "ІНФОРМАЦІЯ_1".

**Annex 1: whose data is masked (59 roles).**
- **Criminal:** підозрюваний, обвинувачений, підсудний, засуджений, виправданий, and their законні представники; потерпілий and his представник / близький родич / законний представник; цивільний позивач / відповідач and their представники; **свідок** and the законний представник of an underage witness; поручитель, заставодавець; the person who understands a mute or deaf witness; minors under educational coercion; persons under compulsory medical measures; the person named in an окрема ухвала.
- **Civil / administrative / commercial:** позивач, відповідач, both kinds of третя особа, заявник, інша заінтересована особа — each with представник and законний представник; **№ 46 «Особа, яка надає правову допомогу»** (i.e. **lawyers and advocates are masked**); піклувальник, опікун, батьки (усиновлювачі), усиновлювана особа; persons in psychiatric or TB compulsory-treatment cases and their representatives or defenders; **кредитор, боржник**; persons under administrative liability and their representatives.
- **№ 59 «Інша фізична особа, відомості щодо якої містяться у тексті судового рішення»**: the catch-all, so effectively **every natural person except officials acting in office**.

### 1.3 Порядок ВРП 2018: how it is done

- «**знеособлення** – процедура маскування … відомостей, що не можуть бути розголошені … **за допомогою спеціалізованого програмного забезпечення з подальшою вибірковою візуальною перевіркою** результатів такого маскування».
- Documents are masked «в автоматичному режимі» within 3 working days of registration and opened to the public by the next working day.
- **The court itself pre-tags the data.** When registering a decision, the court enters in its case-management system:
  - the procedural status of every participant (II.3.15);
  - a flag to remove judges' and participants' names for safety in criminal cases (II.3.16);
  - «наявні в тексті … місця проживання …, адреси, номери телефонів …, адреси електронної пошти, РНОКПП, реквізити документів …, унікальні номери … в ЄДДР, **а також інші відомості, що дають можливість ідентифікувати фізичну особу – учасника справи**» (II.3.17).

  So what gets masked depends on what the court clerk tagged, plus what the software finds. That is a second, human source of inconsistency.

**Consequence for this dataset.** The `pseudonymized` field is ЄДРСР output: court tags + ІСС's automatic masker + spot checks. It is not a careful annotation of the codifier. That is why the gold breaks the official rules in places (judges and enforcement officers masked, e-mail as ІНФОРМАЦІЯ, state bodies masked in some documents only; see §3). The `original` field looks reconstructed: the "real" entities are synthetic (ISO dates «2017-10-28», «Військова частина А1234», «UA2889999803…», «DF987654»).

---

## 2. Official rule per label (codifier + current law), ✅

All four labels: **`ЛЕЙБЛ_N`**, where N «відповідає черговості появи цієї відомості у тексті судового рішення» (order of first appearance). The same piece of information keeps the same number.

| Label | Codifier scope | Added by the current law | Codifier example |
|---|---|---|---|
| **ОСОБА_N** | name (прізвище, ім'я, по батькові) of any natural person in Annex 1 (59 roles, including lawyers and the catch-all «інша фізична особа»), **except officials acting in office** | names only via «інші відомості»; judges' and officials' exceptions removed from the law | «Іванов Іван Іванович» → «ОСОБА_1» |
| **АДРЕСА_N** | address of residence or stay of a natural person | same | «Вул. Київська, кв. 1, буд. 1, м. Київ, 01001» → «АДРЕСА_1» (whole address, city and index included) |
| **НОМЕР_N** | phone or other contact number, **e-mail**, vehicle registration number, identification number, any document number identifying a natural person | **bank account and payment card numbers**, РНОКПП, ID-document details, ЄДДР record number | phone «222-22-22» → «НОМЕР_1» |
| **ІНФОРМАЦІЯ_N** | any other information identifying a natural person | closed-hearing information | author of «ТВІР» → «ІНФОРМАЦІЯ_1» |

**Not masked under the official rules:**
- **Officials acting in office.** Codifier п. 2: judges, prosecutors, court secretaries, investigators, enforcement officers, state bodies' representatives.
- **Legal entities, their names, codes and addresses.** Every rule speaks of «фізична особа» only.
- Case numbers, court names, dates of procedural acts.
- ЄСПЛ parties were explicitly exempt in the 2012–2017 law. Since 2017 there is no explicit exemption, but they are not participants of the Ukrainian case.

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

**Who is masked, compared with the codifier:**

| role (by the word before the name) | masked | left open | codifier |
|---|---|---|---|
| суддя / головуючий / склад колегії | **67** | 0 | **not masked** (official acting in office) |
| державний / приватний виконавець | 14 | 0 | state enforcement officer is an official, so **not masked**; a private one is debatable |
| нотаріус | 48 | 0 | debatable (a private notary is not a civil servant) |
| адвокат | 82 | 0 | **masked**: Annex 1 № 46 «особа, яка надає правову допомогу» |
| представник | 6 | 0 | **masked**: Annex 1 (представник позивача / відповідача / …) |

**Exception 1. Judges and enforcement officers are masked, contrary to codifier п. 2.** The current law no longer exempts judges, so the masker treats them like anyone else. Predict them as ОСОБА.

**Exception 2. What is never masked:**
- ЄСПЛ case names («Пономарьов проти України», «Brualla Gomez de la Torre v. Spain»). This was an explicit exception in 2012–2017 (Закон 4779-VI) and is still the practice;
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

Primary texts (downloaded 08.10.2026; verbatim copies in `analysis/legal/`):
- Закон «Про доступ до судових рішень», ст. 7: current — [zakon.rada.gov.ua/laws/show/3262-15](https://zakon.rada.gov.ua/laws/show/3262-15); 2012–2017 wording — [ed20150212](https://zakon.rada.gov.ua/laws/show/3262-15/ed20150212)
- Кодифікатор (наказ ДСА № 59 від 13.06.2006): status — [zakon.rada.gov.ua/go/v0059750-06](https://zakon.rada.gov.ua/go/v0059750-06); full text with Annex 1 — [lg.court.gov.ua/tu13/inshe/10/1](https://lg.court.gov.ua/tu13/inshe/10/1) (the copy on [ki.court.gov.ua/tu11/14/36](https://ki.court.gov.ua/tu11/14/36) stops at role 55)
- Порядок ведення ЄДРСР (рішення ВРП № 1200/0/15-18): [zakon.rada.gov.ua/go/v1200910-18](https://zakon.rada.gov.ua/go/v1200910-18) (cover decision only); text — [protocol.ua](https://protocol.ua/ua/noviy_poryadok_vedennya_edinogo_dergavnogo_reestru_sudovih_rishen/); latest amendment — [v1807910-26](https://zakon.rada.gov.ua/go/v1807910-26)
- Інструкція (наказ ДСА № 37 від 14.05.2008) — [v0037750-08](https://zakon.rada.gov.ua/rada/show/v0037750-08), cancelled by [v0017750-09](https://zakon.rada.gov.ua/go/v0017750-09)

Secondary:
- ДП «ІСС» automatic masking: [court.gov.ua/press/news/798803](https://court.gov.ua/press/news/798803/)
- Names of international-court parties (Закон 4779-VI): [jurliga.ligazakon.net](https://jurliga.ligazakon.net/news/66309_znyata-zaborona-na-vkazvku-men-storn-v-tekstakh-rshen-mzhnarodnikh-sudv)
- Draft law 7033-д: [zmina.ua PDF](https://zmina.ua/wp-content/uploads/sites/2/2024/11/7033-d-dopovnennya-argumentacziyi-red.pdf)
