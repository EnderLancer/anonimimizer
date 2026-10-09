"""Values for ІНФОРМАЦІЯ placeholders: birth / death / derived dates, e-mails, web addresses, accounts, identifiers,
unit / institution / company names.

The context rules (silver.fill.context) give a sub-type and hints for every placeholder; this module turns them into
a value of the right shape. Whether the value is recorded as a label or as a mask is the fill step's business
(silver.fill.run): the generator writes a natural value either way.
"""
from __future__ import annotations

import datetime as dt
import random
import re
import string
from dataclasses import dataclass, field

from silver.fill.orgs import Org, OrgGenerator, Seen
from silver.fill.persons import Person, PersonGenerator, translit

MONTHS_GEN = ["січня", "лютого", "березня", "квітня", "травня", "червня", "липня", "серпня", "вересня", "жовтня",
              "листопада", "грудня"]
# Format of birth / death dates, one per document. The test generator writes dates as ISO (A. Ruda's thesis); the
# shares are coarse and follow the outside report, which measured them on the test set.
DATE_STYLES, DATE_WEIGHTS = ("iso", "numeric", "words", "words_roku", "words_narodzh"), (45, 25, 8, 12, 10)
REF_YEAR = 2024  # the decisions are from 2024
MAIL_DOMAINS = ["gmail.com", "gmail.com", "gmail.com", "ukr.net", "ukr.net", "i.ua", "meta.ua", "outlook.com", "icloud.com", "email.ua"]
# a date computed from a child's birth date: the age the text names («до повноліття», «до 23 років», «до трьох років») and,
# when the decision has no generated birth date of a child, the years after REF_YEAR in which such a date may fall
DERIVED_AGE = {"majority": 18, "age23": 23, "age3": 3}
DERIVED_YEARS = {"majority": (1, 17), "age23": (1, 5), "age3": (0, 3)}
# the age (lowest, highest) of the person a derived date is computed from, when the fill step found that person's
# birth date: a minor for «до повноліття», a pupil or a student for «до 23 років», an infant for «до трьох років»
DERIVED_CHILD_AGE = {"majority": (1, 17), "age23": (12, 22), "age3": (0, 2)}
WAR_START = dt.date(2022, 2, 24)
# «малолітнього ОСОБА_3, », «неповнолітньої дитини ОСОБА_5, »: the birth date of a minor, whatever the other mentions of
# the placeholder say (the rules give the hints of the last one); 38 of 38 read
_MINOR_LEFT = re.compile(r"(?<![а-яіїє])(?:малолітн|неповнолітн)\w+\s+(?:\w+\s+){0,2}?ОСОБА_\d+\s*,?\s*$", re.I)
# «засудженого ОСОБА_1, », «обвинуваченням ОСОБА_2, », «військовослужбовця ОСОБА_3, »: an adult by the role
_ADULT_ROLE = re.compile(r"(?<![а-яіїє])(?:засуджен|обвинувач|підозрюван|військовослужбов|громадян|позивач|відповідач|боржник|"
                         r"стягувач|заявни[кц]|скаржни[кц]|померл|загибл|спадкодав|дружин|чоловік|батьк|матер|матір|мати\b|вдов|"
                         r"подружж|пенсіонер|працівник|директор|водія|водій)", re.I)
# «який помер X у період дії воєнного стану», «загинув X під час виконання бойового завдання»: a death of the full-scale war
_WAR_CUE = re.compile(r"воєнн\w+\s+стан|бойов\w+\s+(?:дій|завдан)|захист\w*\s+Батьківщини|збройн\w+\s+агресі|(?:зник\w*|зникл\w+)\s+безвісти|"
                      r"безвісно\s+(?:відсутн|зник)|військовослужбов", re.I)
# «ОСОБА_3, ІНФОРМАЦІЯ_2, який помер »: the placeholder before ours is the birth date of the person who died
_DOB_JUST_BEFORE = re.compile(r"ІНФОРМАЦІЯ_(\d+)[^.;]{0,40}$")
# «ОСОБА_3, 【X】, який помер ІНФОРМАЦІЯ_4»: the same pair seen from the birth date, when the death date was filled earlier
_DEATH_JUST_AFTER = re.compile(r"^[^.;]{0,30}(?:помер|загин)\w*\s+ІНФОРМАЦІЯ_(\d+)")
# what the text says about a web address: the network («у мережі Facebook», «Telegram-каналі») …
_PLATFORMS = (("facebook", re.compile(r"facebook|фейсбу[кц]|(?<![a-z])fb(?![a-z])", re.I)), ("instagram", re.compile(r"instagram|інстаграм", re.I)),
              ("telegram", re.compile(r"telegram|телеграм", re.I)), ("youtube", re.compile(r"you\s?tube|ютуб", re.I)),
              ("tiktok", re.compile(r"tik\s?tok|тік\s?ток", re.I)))
# … a video («відеоролики за посиланнями», «у відеоматеріалі») …
_VIDEO_CUE = re.compile(r"відео|ролик|ефір|трансляці", re.I)
# … a court service («сервісу «Електронний суд»», «на сайті Ізмаїльського міськрайонного суду», «ЄСІТС», «відеоконференції») …
_COURT_CUE = re.compile(r"(?<![а-яіїє])суд(?:у|ом|і|ів)?(?![а-яіїє])|ЄСІТС|відео-?\s?конференц|відоконференц|ВКЗ|EasyCon|судов\w+\s+влад", re.I)
# … or the site by its name: «на веб-сайті ZN.UA за адресою:» -> zn.ua
_NAMED_DOMAIN = re.compile(r"(?<![\w@.])((?:[A-Za-z0-9-]+\.)+(?:ua|com|net|org|info|media|tv))(?![\w.]*@)(?!\w)", re.I)
CONTEXT = 70  # characters before a web address in which these cues count
COURT_PREFIXES = ["od", "lv", "ki", "hr", "dp", "vn", "zp", "pl", "cn", "zt", "if", "km", "ck", "te", "rv", "vl", "sm", "mk"]
# web addresses by what the text says about them: a site, a page of a site (an article, a video), a profile
# no court or government domain: the decisions name court.gov.ua and the like in the open (852 times in the decisions that have an
# ІНФОРМАЦІЯ placeholder), a labelled one would contradict them
SITE_DOMAINS = ["setam.net.ua", "olx.ua", "rozetka.com.ua", "prom.ua", "auto.ria.com", "dom.ria.com", "work.ua", "rabota.ua", "epicentrk.ua",
                "privatbank.ua", "monobank.ua", "nova-poshta.ua", "ukrposhta.ua", "booking.com", "hotline.ua"]
NEWS_DOMAINS = ["pravda.com.ua", "ukrinform.ua", "unian.ua", "tsn.ua", "censor.net", "zaxid.net", "suspilne.media", "nv.ua",
                "radiosvoboda.org", "lb.ua", "obozrevatel.com", "glavcom.ua", "dumskaya.net", "0352.ua", "056.ua", "depo.ua"]
SITE_WORDS = ["novyny", "pravo", "misto", "region", "info", "portal", "visnyk", "holos", "ohliad", "fakty", "torg", "market",
              "bud", "agro", "avto", "dim", "shop", "service", "media", "kray"]
SITE_ZONES = ["ua", "com.ua", "com.ua", "org.ua", "in.ua", "net.ua", "kiev.ua", "lviv.ua", "od.ua", "com", "info"]
PAGE_SECTIONS = ["news", "novyny", "articles", "publikatsii", "suspilstvo", "kryminal", "polityka", "blogs", "ukraine", "regions"]
PAGE_WORDS = ["sud", "sprava", "rishennia", "deputat", "posadovets", "skandal", "rozsliduvannia", "khabar", "zemlia", "biudzhet",
              "tender", "zaiava", "politsiia", "prokuratura", "zvilnennia", "konflikt", "vybory", "korruptsiia", "mer", "holova"]
_VIDEO_ID = string.ascii_letters + string.digits + "-_"
# surnames for a name that stands in the quotes of the text, when no person generator was given
SURNAMES = ["Шевченко", "Коваленко", "Бондаренко", "Ткаченко", "Кравченко", "Мельник", "Олійник", "Лисенко", "Мороз", "Савчук"]
KNOWN_SUBTYPES = ("dob", "death", "derived", "date_event", "email", "url", "nick", "ident", "military_name", "org", "quoted",
                  "place", "other")


@dataclass
class DocState:
    """What the fill step knows about one decision while it fills it."""
    seen: Seen = field(default_factory=Seen)   # the names the decision has: its text and the bodies drawn so far
    child_dob: dt.date | None = None           # the last generated birth date of a child: derived dates follow it
    last_dob: dt.date | None = None            # the last generated birth date of anybody: a death next to it comes after
    births: list[tuple[dt.date, bool]] = field(default_factory=list)  # every generated birth date (and: a child's?), in text order
    birth_of: dict[int, dt.date] = field(default_factory=dict)  # … and by the index of its placeholder
    # what the fill step read from the text before filling (silver.fill.run.link_derived): the placeholder that holds the
    # birth date a derived date is computed from, and the age range of the person born on such a date
    derived_from: dict[int, int] = field(default_factory=dict)
    child_age: dict[int, tuple[int, int]] = field(default_factory=dict)
    derived_used: set[tuple[dt.date, str]] = field(default_factory=set)  # (birth date, age) pairs that already gave a derived date
    death_of: dict[int, dt.date] = field(default_factory=dict)  # generated death dates by the index of their placeholder


class InfoGenerator:
    def __init__(self, rng: random.Random, persons: PersonGenerator | None = None):
        """`persons`: the generator of the fill step, for a surname in quotes; without it a short list is used."""
        self.rng = rng
        self.org_gen = OrgGenerator(rng)
        self.persons = persons

    def _date(self, y0: int, y1: int) -> dt.date:
        y = self.rng.randint(y0, y1)
        return dt.date(y, self.rng.randint(1, 12), self.rng.randint(1, 28))

    def _birth(self, child: bool, adult: bool = False, ages: tuple[int, int] | None = None) -> dt.date:
        """Minors are 1–17 years old; adults are mostly of working age (25–50), the rest 18–80. One birth date in ten
        without a child cue is a minor's all the same (the cue is often missing), unless the text gives the person an
        adult's role (`adult`). `ages`: the lowest and highest age, when the text fixes them."""
        r = self.rng
        if ages:
            age = r.randint(*ages)
        elif child or (not adult and r.random() < 0.1):
            age = r.randint(1, 17)
        else:
            age = r.randint(25, 50) if r.random() < 0.7 else r.randint(18, 80)
        return self._date(REF_YEAR - age, REF_YEAR - age)

    def _death(self, born: dt.date | None, war: bool) -> dt.date:
        """A date of the last 21 years; from 24.02.2022 on after a cue of the full-scale war; after `born`, the birth
        date of the person who died, when the text gives it right before."""
        first, last = dt.date(REF_YEAR - 21, 1, 1), dt.date(REF_YEAR, 12, 28)
        if war:
            first = WAR_START
        if born:
            first = max(first, min(born + dt.timedelta(days=366), last))
        d = first + dt.timedelta(days=self.rng.randint(0, (last - first).days))
        return d.replace(day=min(d.day, 28))

    def doc_date_style(self) -> str:
        return self.rng.choices(DATE_STYLES, DATE_WEIGHTS)[0]

    def render_date(self, d: dt.date, style: str) -> str:
        if style == "numeric":
            return d.strftime("%d.%m.%Y")
        if style == "iso":
            return d.isoformat()
        if style == "words":
            return f"{d.day} {MONTHS_GEN[d.month - 1]} {d.year}"
        if style == "words_roku":
            return f"{d.day} {MONTHS_GEN[d.month - 1]} {d.year} року"
        return f"{d.day} {MONTHS_GEN[d.month - 1]} {d.year} року народження"

    def _style(self, date_style: str | None, hints: dict, birth: bool) -> str:
        """The document's date format, narrowed by the text around the placeholder; only a birth date may end with
        «року народження»."""
        style = date_style or self.doc_date_style()
        if hints.get("style") == "numeric":  # «… р.н.» follows
            return "numeric"
        if hints.get("no_roku"):  # the text itself continues with «року …»
            return style if style in ("iso", "numeric") else "words"
        return "words_roku" if style == "words_narodzh" and not birth else style

    def derived_date(self, until: str, child_dob: dt.date | None) -> dt.date:
        """The day a child reaches the age the text names («до досягнення повноліття, тобто до …»): the child's
        generated birth date plus that age, or - when the decision has none - a day of the years it may fall in."""
        until = until if until in DERIVED_AGE else "majority"
        if child_dob is not None:
            return child_dob.replace(year=child_dob.year + DERIVED_AGE[until])  # the day is 1 … 28: no 29 February
        y0, y1 = DERIVED_YEARS[until]
        return self._date(REF_YEAR + y0, REF_YEAR + y1)

    def _derived(self, until: str, idx: int | None, doc: DocState) -> dt.date:
        """The derived date of placeholder `idx`: from the birth date the fill step linked it to (generated here when
        that placeholder stands later in the text); else from the nearest earlier birth date - a child's first - that
        puts the day in REF_YEAR or later and gave no such date yet (two children of one decision do not come of age on one day);
        else a day of the years it may fall in."""
        until = until if until in DERIVED_AGE else "majority"
        src = doc.derived_from.get(idx) if idx is not None else None
        if src is not None:
            if src not in doc.birth_of:
                doc.birth_of[src] = self._birth(True, ages=doc.child_age.get(src))
            born = doc.birth_of[src]
        else:
            age = DERIVED_AGE[until]
            free = [(b, child) for b, child in reversed(doc.births) if (b, until) not in doc.derived_used]
            # a child's date first; any other only when it leaves the day clearly ahead (not an adult of eighteen)
            born = next((b for b, child in free if child and b.year + age >= REF_YEAR), None) or next(
                (b for b, _ in free if b.year + age > REF_YEAR), None)
        if born is not None and born.year + DERIVED_AGE[until] < REF_YEAR:
            born = None  # the linked date is an adult's after all: no date in the past
        if born is not None:
            doc.derived_used.add((born, until))
        return self.derived_date(until, born)

    def _surname(self) -> str:
        return self.persons.new_person().surname if self.persons else self.rng.choice(SURNAMES)

    @staticmethod
    def _platform(left: str) -> str | None:
        """The network the text names last before the address: «у мережі Facebook за посиланням» -> facebook."""
        found = [(m.end(), name) for name, rx in _PLATFORMS for m in rx.finditer(left[-CONTEXT:])]
        return max(found)[1] if found else None

    @staticmethod
    def _named_domain(left: str) -> str | None:
        """«на веб-сайті ZN.UA за адресою:» -> zn.ua: the site the text names right before its address."""
        found = _NAMED_DOMAIN.findall(left[-CONTEXT:])
        return found[-1].lower() if found else None

    def _court_host(self) -> str:
        """The site of a court, as the court portal names them: «iz.od.court.gov.ua»-like hosts are not in the open text."""
        r = self.rng
        return f"{''.join(r.choice(string.ascii_lowercase) for _ in range(2))}.{r.choice(COURT_PREFIXES)}.court.gov.ua"

    def _site(self, left: str = "") -> str:
        r = self.rng
        named = self._named_domain(left)
        if named:
            domain = named
        elif _COURT_CUE.search(left[-CONTEXT:]):
            domain = self._court_host()
        elif r.random() < 0.6:
            domain = r.choice(SITE_DOMAINS + NEWS_DOMAINS)
        else:
            one, two = r.sample(SITE_WORDS, 2)
            domain = f"{one}{r.choice(['', '-', ''])}{two}.{r.choice(SITE_ZONES)}"
        return f"{r.choice(['https://', 'https://', 'https://www.', 'http://', 'www.', ''])}{domain}"

    def _page(self, left: str = "") -> str:
        """An article, or a video where the text speaks of one; a page of the site or the network the text names."""
        r = self.rng
        platform, named = self._platform(left), self._named_domain(left)
        video = platform == "youtube" or (bool(_VIDEO_CUE.search(left[-CONTEXT:])) and not named and r.random() < 0.7)
        if platform in ("facebook", "instagram", "telegram", "tiktok"):
            return self._post(platform)
        if video or (not left and r.random() < 0.35):
            vid = "".join(r.choice(_VIDEO_ID) for _ in range(11))
            return r.choice([f"https://www.youtube.com/watch?v={vid}", f"https://www.youtube.com/watch?v={vid}", f"https://youtu.be/{vid}"])
        if not named and _COURT_CUE.search(left[-CONTEXT:]):
            court = f"https://{self._court_host()}/sud{r.randint(100, 2700):04d}"
            return r.choice([f"{court}/gromadyanam/{r.choice(['csz', 'online', 'rozklad', 'videokonf'])}/", f"{court}/",
                             f"{court}/pres-centr/news/{r.randint(10**5, 10**7)}/"])
        slug = "-".join(r.sample(PAGE_WORDS, r.randint(2, 4)))
        if named:
            return f"https://{named}/{r.choice(PAGE_SECTIONS)}/{slug}-{r.randint(10**4, 10**6)}.html"
        path = r.choice([f"{r.choice(PAGE_SECTIONS)}/{r.randint(2019, REF_YEAR)}/{r.randint(1, 12):02d}/{r.randint(1, 28):02d}/{r.randint(10**5, 10**7)}/",
                         f"{r.choice(PAGE_SECTIONS)}/{slug}-{r.randint(10**4, 10**6)}.html", f"{r.choice(PAGE_SECTIONS)}/{slug}",
                         f"{r.choice(PAGE_SECTIONS)}/{r.randint(10**5, 10**7)}-{slug}.html"])
        return f"https://{r.choice(['', '', 'www.'])}{r.choice(NEWS_DOMAINS)}/{path}"

    def _post(self, platform: str) -> str:
        """One publication of a network: «у публікації на сторінці Facebook за посиланням …»."""
        r = self.rng
        name = "_".join(r.sample(SITE_WORDS, 2))
        if platform == "facebook":
            return r.choice([f"https://www.facebook.com/{name.replace('_', '.')}/posts/{r.randint(10**14, 10**16)}",
                             f"https://www.facebook.com/permalink.php?story_fbid={r.randint(10**14, 10**16)}&id={r.randint(10**9, 10**10)}"])
        if platform == "instagram":
            return f"https://www.instagram.com/p/{''.join(r.choice(_VIDEO_ID) for _ in range(11))}/"
        if platform == "telegram":
            return f"https://t.me/{name}/{r.randint(10, 99999)}"
        return f"https://www.tiktok.com/@{name}/video/{r.randint(10**18, 10**19)}"

    def _profile(self, person: Person | None, left: str = "") -> str:
        """A person's profile, of the network the text names when it names one."""
        r = self.rng
        links = {"facebook": f"https://www.facebook.com/profile.php?id={r.randint(10**9, 10**10)}",
                 "telegram": f"https://t.me/{translit(person.surname) if person else 'user'}{r.randint(1, 999)}",
                 "instagram": f"https://instagram.com/{translit(person.first) if person else 'user'}_{r.randint(1, 9999)}",
                 "youtube": f"https://www.youtube.com/@{translit(person.surname) if person else 'user'}{r.randint(1, 999)}",
                 "tiktok": f"https://www.tiktok.com/@{translit(person.surname) if person else 'user'}{r.randint(1, 999)}"}
        return links.get(self._platform(left)) or r.choice([links["facebook"], links["telegram"], links["instagram"]])

    def _account_page(self, person: Person | None, left: str = "") -> str:
        """A page or a channel of a social network: «на сторінці …», «у Telegram-каналі …»; of the network the text names."""
        r = self.rng
        base = translit(person.surname) if person and r.random() < 0.5 else "_".join(r.sample(SITE_WORDS, 2))
        name = r.choice([base, f"{base}{r.randint(1, 99)}", f"{base}.{r.choice(['ua', 'official', 'news', 'live'])}"])
        links = {"telegram": [f"https://t.me/{name.replace('.', '_')}", f"t.me/{name.replace('.', '_')}"],
                 "facebook": [f"https://www.facebook.com/{name}", f"https://www.facebook.com/groups/{r.randint(10**11, 10**15)}"],
                 "instagram": [f"https://www.instagram.com/{name}/"], "tiktok": [f"https://www.tiktok.com/@{name}"],
                 "youtube": [f"https://www.youtube.com/@{name}",
                             f"https://www.youtube.com/channel/UC{''.join(r.choice(_VIDEO_ID) for _ in range(22))}"]}
        platform = self._platform(left)
        return r.choice(links[platform] if platform else [link for group in links.values() for link in group])

    def _ident(self, person: Person | None) -> str:
        """A login or an identifier of an information system: «логін …», «ідентифікатор …», «ID …»."""
        r = self.rng
        base = translit(person.surname) if person else "user"
        return r.choice([f"{base}{r.randint(1, 9999)}", f"{base}_{r.randint(1960, 2010)}", f"{base[:1]}.{base}" if person else f"user_{r.randint(100, 99999)}",
                         f"id{r.randint(10**6, 10**9)}", str(r.randint(10**7, 10**10)),
                         "".join(r.choice(string.ascii_lowercase + string.digits) for _ in range(r.randint(8, 12))),
                         f"{''.join(r.choice(string.ascii_uppercase) for _ in range(2))}{r.randint(10**4, 10**6)}"])

    def generate(self, subtype: str, hints: dict, person: Person | None, date_style: str | None = None,
                 org_ctx: dict | None = None, doc: DocState | None = None, left: str = "", right: str = "",
                 idx: int | None = None) -> str | Org:
        """`date_style` is the document's date format (doc_date_style); the text around a placeholder may narrow it.
        `org_ctx` says what the decision is about (military, commercial, social) and picks the kind of institution.
        `doc` carries what the decision already has: its names (a body that stands there is not drawn) and the birth
        date of a child. `left` and `right` are the text around the placeholder: they tell which bank / court / hospital
        kind of name a slot of a known kind holds, which network a web address is of, whether a birth date is a minor's
        and a death one of the war. `idx` is the index of the placeholder, by which a derived date finds the birth date
        it is computed from. An institution comes back as an Org, which the caller renders in the case of each mention.
        A sub-type the generator does not know is filled like «other»: with an institution."""
        r = self.rng
        org_ctx = org_ctx or {}
        doc = doc or DocState()
        if subtype == "dob":
            ages = doc.child_age.get(idx) if idx is not None else None
            child = bool(hints.get("child")) or bool(ages) or bool(_MINOR_LEFT.search(left))
            d = doc.birth_of.get(idx) if idx is not None else None  # a derived date earlier in the text already fixed it
            if d is None:
                d = self._birth(child, adult=bool(_ADULT_ROLE.search(left[-CONTEXT:])), ages=ages)
                died = _DEATH_JUST_AFTER.search(right)
                died = doc.death_of.get(int(died.group(1))) if died else None
                if died and d >= died - dt.timedelta(days=366):  # the person was born before the death the text goes on with
                    d = d.replace(year=died.year - r.randint(20, 80))
            if child:
                doc.child_dob = d
            doc.last_dob = d
            doc.births.append((d, child))
            if idx is not None:
                doc.birth_of[idx] = d
            return self.render_date(d, self._style(date_style, hints, birth=True))
        if subtype == "death":
            before = _DOB_JUST_BEFORE.search(left[-60:])
            d = self._death(doc.birth_of.get(int(before.group(1))) if before else None,
                            war=bool(_WAR_CUE.search(left[-CONTEXT:] + " " + right[:CONTEXT])))
            if idx is not None:
                doc.death_of[idx] = d
            return self.render_date(d, self._style(date_style, hints, birth=False))
        if subtype == "derived":  # «до досягнення дитиною повноліття, тобто до …»
            return self.render_date(self._derived(hints.get("until", "majority"), idx, doc), self._style(date_style, hints, birth=False))
        if subtype == "date_event":  # a recent date that is neither birth nor death
            return self.render_date(self._date(REF_YEAR - 6, REF_YEAR), self._style(date_style, hints, birth=False))
        if subtype == "email":
            if person and r.random() < 0.7:
                local = r.choice([f"{translit(person.first)}.{translit(person.surname)}",
                                  f"{translit(person.surname)}{r.randint(1960, 2005)}",
                                  f"{translit(person.first)[0]}{translit(person.surname)}{r.randint(1, 99)}"])
            else:
                local = f"user{r.randint(1000, 99999)}"
            return f"{local}@{r.choice(MAIL_DOMAINS)}"
        if subtype == "url":
            kind = hints.get("url_kind", "profile")
            return self._site(left) if kind == "site" else self._page(left) if kind == "page" else self._profile(person, left)
        if subtype == "nick":
            if hints.get("nick_kind") == "page":
                return self._account_page(person, left)
            base = translit(person.surname) if person else "user"
            return r.choice([f"@{base}{r.randint(1, 999)}", f"{base}_{r.randint(10, 99)}", f"@{base}.official"])
        if subtype == "ident":
            return self._ident(person)
        if subtype == "military_name":
            return self.org_gen.unit_name(seen=doc.seen)
        if subtype == "quoted":  # the quotes stand in the text: «ІНФОРМАЦІЯ_3» -> «Комтех Логістік»
            kind = hints.get("quote_kind", "company")
            if kind == "person":
                return self._surname()
            return self.org_gen.quoted_name(kind, bool(org_ctx.get("commercial")), seen=doc.seen, left=left)
        # «org», and «other» or an unknown sub-type: ЄДРСР marked personal data here, most likely an institution
        org_kind = hints.get("org_kind", "party") if subtype == "org" else "party"
        return self.org_gen.institution(**org_ctx, org_kind=org_kind, seen=doc.seen, left=left, right=right,
                                        count_as=None if subtype == "org" else "other")
