import pytest

from silver.fill.orgs import Org, OrgGenerator, inflect_phrase
from silver.fill.run import Filler, has_org_info
from silver.normalize import find_placeholders, normalize


@pytest.mark.parametrize("head, case, expected", [
    ("Міністерство", "gent", "Міністерства"), ("Міністерство", "datv", "Міністерству"), ("Міністерство", "ablt", "Міністерством"),
    ("Головне управління", "gent", "Головного управління"), ("Головне управління", "datv", "Головному управлінню"),
    ("Головне управління", "loct", "Головному управлінні"), ("Головне управління", "accs", "Головне управління"),
    ("Військова частина", "gent", "Військової частини"), ("Військова частина", "accs", "Військову частину"),
    ("Військово-лікарська комісія", "gent", "Військово-лікарської комісії"), ("Військово-лікарська комісія", "accs", "Військово-лікарську комісію"),
    ("Деснянський районний територіальний центр", "gent", "Деснянського районного територіального центру"),
    ("Деснянський районний територіальний центр", "accs", "Деснянський районний територіальний центр"),
    ("Головний центр", "gent", "Головного центру"), ("Квартирно-експлуатаційний відділ", "gent", "Квартирно-експлуатаційного відділу"),
    ("Департамент", "datv", "Департаменту"), ("Генеральний штаб", "datv", "Генеральному штабу"),
    ("Приватне акціонерне товариство", "gent", "Приватного акціонерного товариства"), ("Товариство", "ablt", "Товариством"),
    ("24 окрема механізована бригада", "gent", "24 окремої механізованої бригади"), ("Командування", "gent", "Командування"),
    ("ТЦК та СП", "gent", "ТЦК та СП"), ("Стрийська міська рада", "gent", "Стрийської міської ради"),
    ("Павлоградська районна державна адміністрація", "gent", "Павлоградської районної державної адміністрації"),
])
def test_inflect_phrase(head, case, expected):
    assert inflect_phrase(head, case) == expected


def test_org_keeps_its_tail_and_fixed_forms():
    assert Org("Міністерство", "оборони України").render("gent") == "Міністерства оборони України"
    assert Org("Товариство", "з обмеженою відповідальністю «Комтех Логістік»").render("datv") == "Товариству з обмеженою відповідальністю «Комтех Логістік»"
    assert Org("в/ч", "А1234", fixed=True).render("gent") == "в/ч А1234"


def test_an_institution_is_one_entity_declined_per_mention():
    raw = ("ОСОБА_1 звернувся з позовом до ІНФОРМАЦІЯ_1 (далі - відповідач) про стягнення грошового забезпечення; просив зобов'язати "
           "ІНФОРМАЦІЯ_1 виплатити кошти. ІНФОРМАЦІЯ_1 подало касаційну скаргу. Повернути ІНФОРМАЦІЯ_1 касаційну скаргу.")
    text = normalize(raw)
    doc = {"id": "org", "text": text, "placeholders": find_placeholders(text), "kind": "ВС"}
    assert has_org_info(doc)
    out = Filler(seed=3).fill(doc)
    mentions = [out["text"][l["start"]:l["end"]] for l in out["labels"] if l["pseudonym"] == "ІНФОРМАЦІЯ_1"]
    assert len(mentions) == 4
    gent, accs, nomn, datv = mentions
    # the subject form is the nominative: the genitive and the dative are its declined forms, not the other way round
    assert inflect_phrase(nomn, "gent") == gent and inflect_phrase(nomn, "datv") == datv and accs == nomn
    assert len({m.split()[0][:4] for m in mentions}) == 1  # the same institution every time
    assert not (gent == accs == nomn == datv and not mentions[0].startswith(("ТЦК", "в/ч")))  # at least one case form differs


def test_company_in_quotes_is_a_bare_name():
    text = normalize("Товариство з обмеженою відповідальністю « ІНФОРМАЦІЯ_1 » (далі - ТОВ « ІНФОРМАЦІЯ_1 ») подало позов.")
    doc = {"id": "q", "text": text, "placeholders": find_placeholders(text), "kind": "ГПК"}
    out = Filler(seed=3).fill(doc)
    for l in out["labels"]:
        name = out["text"][l["start"]:l["end"]]
        assert "«" not in name and name[0].isupper()
        assert out["text"][l["start"] - 1] == "«" and out["text"][l["end"]] == "»"


def test_institution_kind_follows_the_decision():
    g = OrgGenerator(__import__("random").Random(1))
    military = [g.institution(military=True).render() for _ in range(200)]
    commercial = [g.institution(commercial=True).render() for _ in range(200)]
    military_words = ("оборони", "частина", "ТЦК", "в/ч", "комплектування", "Збройних Сил", "військов", "гвардії", "прикордонн",
                      "полк", "бригада", "батальйон", "загін", "госпітал", "Служба безпеки", "Генеральн", "Командування", "квартирно-експлуатац")
    assert sum(any(w in x for w in military_words) for x in military) > 120  # the rest: pension / police bodies of servicemen's cases
    assert sum(("«" in x or '"' in x) for x in commercial) > 150
