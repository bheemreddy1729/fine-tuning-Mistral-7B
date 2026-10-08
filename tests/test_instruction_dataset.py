"""Run: python tests/test_instruction_dataset.py  (no framework, fails on first broken assert)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import instruction_dataset as d  # noqa: E402

D = d.DISCLAIMER
CHUNK = {"text": "WHO says give 7 mg/kg of primaquine for 14 days to prevent relapse in vivax malaria patients after G6PD testing."}


def test_sentences():
    assert d.count_sentences("Per WHO, give 0.5 mg/kg daily. Avoid it in P. vivax cases, e.g. in pregnancy.") == 2
    assert d.count_sentences("One. Two. Three. Four.") == 4


def test_check_draft():
    good = {"instruction": "How is relapse prevented?",
            "response": "Per WHO guidance, vivax relapse is prevented with a 7 mg/kg primaquine course spread over 14 days, "
                        "which protects patients while the national programme decides on supervised dosing, "
                        "and clinicians should confirm the G6PD status. " + D}
    assert d.check_draft(good, CHUNK) == [], d.check_draft(good, CHUNK)
    assert d.check_draft({**good, "response": good["response"][:-5]}, CHUNK)  # disclaimer broken
    assert any("numbers" in e for e in d.check_draft({**good, "response": good["response"].replace("14", "21")}, CHUNK))
    assert any("verbatim" in e for e in d.check_draft(
        {**good, "response": "Per WHO, " + CHUNK["text"][8:] + " " + D}, CHUNK))


def test_leakage():
    fixed = d.BASELINE_PROMPTS[0]["instruction"]
    assert d.check_leakage(fixed) is not None
    assert d.check_leakage("What is azithromycin?") is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
