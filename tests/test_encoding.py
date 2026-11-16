"""Guards against the Windows UnicodeDecodeError (see docs/postmortems.md).

Windows defaults to cp1252 when no encoding is given, so reading UTF-8 files with
read_text() crashed there but not on Linux. These tests do not depend on the OS default.
"""
import ast
import pathlib
import tempfile

from ragassist.eval.run_eval import load_golden, load_index

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
ENCODING_CALLS = {"read_text", "write_text"}


def test_no_text_io_without_explicit_encoding():
    """Fails if any read_text/write_text/open(text mode) call in src/ omits encoding=."""
    problems = []
    for path in SRC.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            is_text_io = name in ENCODING_CALLS or name == "open"
            if is_text_io and not any(k.arg == "encoding" for k in node.keywords):
                # open() with a binary mode is fine
                modes = [a.value for a in node.args[1:2] if isinstance(a, ast.Constant)]
                if name == "open" and any("b" in str(m) for m in modes):
                    continue
                problems.append(f"{path.name}:{node.lineno} {name}()")
    assert not problems, "missing encoding='utf-8': " + ", ".join(problems)


def test_non_ascii_corpus_and_golden_roundtrip():
    text = "# Atmosphere\nStandard temperature is 59 °F. “Quoted” text — with dashes."
    with tempfile.TemporaryDirectory() as d:
        d = pathlib.Path(d)
        (d / "doc.md").write_text(text, encoding="utf-8")
        (d / "g.jsonl").write_text(
            '{"question": "temp?", "answer": "59 °F", "sources": ["doc"]}\n', encoding="utf-8"
        )
        idx = load_index(d)
        assert any("°F" in c.text for c in idx.chunks)
        assert load_golden(d / "g.jsonl")[0]["answer"] == "59 °F"
