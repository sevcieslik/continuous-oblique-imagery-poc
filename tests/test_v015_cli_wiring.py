"""Static wiring checks for main_v015: a typo in an args/config key would otherwise only crash hours
into a full-circuit run, after sequence building and LiDAR indexing."""
import argparse
import ast
from pathlib import Path

import pytest

import continuous_oblique_v015 as co

SRC = Path(co.__file__).read_text(encoding="utf-8")
TREE = ast.parse(SRC)


def _func(name):
    return next(n for n in ast.walk(TREE) if isinstance(n, ast.FunctionDef) and n.name == name)


class _Parsed(Exception):
    pass


def _parsed_namespace(monkeypatch):
    orig = argparse.ArgumentParser.parse_args

    def capture(self, args=None, namespace=None):
        raise _Parsed(orig(self, args, namespace))

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture)
    with pytest.raises(_Parsed) as exc:
        co.main_v015(["D:/nonexistent_project"])
    return exc.value.args[0]


def test_every_args_attribute_used_in_main_is_defined(monkeypatch):
    ns = _parsed_namespace(monkeypatch)
    used = {n.attr for n in ast.walk(_func("main_v015"))
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "args"}
    missing = sorted(a for a in used if not hasattr(ns, a))
    assert not missing, f"args attributes used but not defined by the parser: {missing}"


def test_det_cfg_provides_every_key_the_detector_reads():
    main = _func("main_v015")
    det_cfg = next(n.value for n in ast.walk(main)
                   if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "det_cfg" for t in n.targets))
    provided = {k.value for k in det_cfg.keys}
    render = _func("render_candidate_products_and_crops_v15")
    read = {n.slice.value for n in ast.walk(render)
            if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id == "c"
            and isinstance(n.slice, ast.Constant)}
    assert read <= provided, f"missing det_cfg keys: {sorted(read - provided)}"


def test_defaults_and_version(monkeypatch):
    ns = _parsed_namespace(monkeypatch)
    assert ns.tower_min_ownership == 0.55 and ns.registration_min_matched == 2
    assert ns.crop_qa_review_limit == 200 and ns.competitor_structures is None
    assert 'summary["poc_version"] = "0.15"' in SRC
    assert "raise SystemExit(main_v015())" in SRC
