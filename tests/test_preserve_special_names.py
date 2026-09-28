import importlib.util
import sys
import types
from pathlib import Path


def load_preserve_special_names():
    here = (
        Path(__file__).resolve().parents[1]
        / "models/codec/dualcodec/dualcodec/utils/normalization"
    )
    package = types.ModuleType("amphion_norm")
    package.__path__ = [str(here)]
    sys.modules["amphion_norm"] = package

    def load(name):
        spec = importlib.util.spec_from_file_location(
            f"amphion_norm.{name}", here / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        module.__package__ = "amphion_norm"
        sys.modules[f"amphion_norm.{name}"] = module
        spec.loader.exec_module(module)
        return module

    load("global_punct")
    return load("en").preserve_special_names


def test_possessive_acronym_keeps_its_last_letter():
    preserve_special_names = load_preserve_special_names()
    assert preserve_special_names("NASA's") == "N A S A's"
    assert preserve_special_names("AI's") == "A I's"
    assert preserve_special_names("the USA") == "the U S A"
    assert preserve_special_names("John's") == "john's"
    assert preserve_special_names("don't") == "don't"
