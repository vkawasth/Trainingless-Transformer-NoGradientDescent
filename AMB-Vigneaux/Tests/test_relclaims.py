import pytest
spacy = pytest.importorskip("spacy")
try:
    spacy.load("en_core_web_sm")
except OSError:
    pytest.skip("en_core_web_sm not installed", allow_module_level=True)
from amb_vigneaux.relclaims import extract, detect

CASES = {
    "polarity (said)": {"A": "Smith said the tax cut will reduce the deficit sharply next year.",
                        "B": "Smith denied that the tax cut will reduce the deficit sharply next year."},
    "speaker": {"A": "Senator Jones said the bill protects rural hospitals from closure.",
                "B": "Governor Brown said the bill protects rural hospitals from closure."},
    "polarity (cause)": {"A": "The storm caused the blackout.", "B": "The storm did not cause the blackout."},
    "reversal": {"A": "The protests led to the curfew.", "B": "The curfew sparked the protests."},
    "cycle": {"A": "Inflation caused the strike.", "B": "The strike triggered the shortage.", "C": "The shortage fueled inflation."},
    "NEG paraphrase": {"A": "Smith said the plan would lower costs for families.",
                       "B": "According to Smith, the plan would lower costs for families."},
    "NEG chain": {"A": "Heavy rain caused the flooding.", "B": "The flooding caused evacuations."},
    "NEG passive agree": {"A": "The blackout was caused by the storm.", "B": "The storm caused the blackout."},
    "NEG role": {"A": "Mr. Goelman said the decision should trouble all Americans.",
                 "B": "The attorney said the decision should trouble all Americans."},
    "NEG nested": {"A": '"Trump says he wants to run the nation like his business," Mr. Bloomberg said.',
                   "B": "Mr. Bloomberg said Trump says he wants to run the nation like his business."},
}


@pytest.mark.parametrize("name", list(CASES))
def test_planted(name):
    cl = [c for s, t in CASES[name].items() for c in extract(s, t)]
    kinds = [x.kind for x in detect(cl)]
    if name.startswith("NEG"):
        assert kinds == []
    else:
        assert kinds and kinds[0].split("/")[0] == name.split(" ")[0]
