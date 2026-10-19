"""Headless tests of the Streamlit app using streamlit.testing.AppTest."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _run():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def _text(at) -> str:
    return " ".join(m.value for m in at.markdown)


def test_app_renders_all_tabs():
    at = _run()
    assert len(at.tabs) == 4
    assert "Estimated survival probability" in _text(at)


def test_presets_change_estimate():
    at = _run()
    at.button(key="preset_0").click().run()          # 1st-class woman
    assert not at.exception
    high = _text(at)
    at.button(key="preset_1").click().run()          # 3rd-class man
    assert not at.exception
    assert "leans towards survival" in high
    assert "leans towards not surviving" in _text(at)


def test_unknown_age_and_custom_fare():
    at = _run()
    at.checkbox(key="age_unknown").check().run()
    assert not at.exception
    assert any("Age unknown" in i.value for i in at.info)
    at.checkbox(key="typical_fare").uncheck().run()
    at.number_input(key="fare").set_value(900.0).run()
    assert not at.exception
    assert any("Outside the training data" in w.value for w in at.warning)
