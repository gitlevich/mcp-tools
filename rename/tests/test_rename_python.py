import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from rename_python import RenameResult, find_symbol_offset, rename_python


def test_find_symbol_offset_simple():
    source = "def calculate(x):\n    return x * 2\n"
    offset = find_symbol_offset(source, "calculate")
    assert source[offset : offset + len("calculate")] == "calculate"


def test_find_symbol_offset_not_found():
    source = "def foo():\n    pass\n"
    with pytest.raises(ValueError, match="not found"):
        find_symbol_offset(source, "bar")


def test_find_symbol_offset_avoids_substring():
    source = "def recalculate():\n    pass\n\ndef calculate():\n    pass\n"
    offset = find_symbol_offset(source, "calculate")
    # Should find "calculate" (the standalone function), not inside "recalculate"
    assert source[offset : offset + len("calculate")] == "calculate"
    # Verify it's the standalone one by checking the char before is not a letter
    assert offset == 0 or not source[offset - 1].isalnum()


def test_rename_function_single_file(tmp_path: Path):
    f = tmp_path / "math_utils.py"
    f.write_text("def calculate(x):\n    return x * 2\n\nresult = calculate(5)\n")

    result = rename_python(str(f), "calculate", "compute", str(tmp_path))

    assert result.old_name == "calculate"
    assert result.new_name == "compute"
    assert len(result.files_changed) >= 1

    content = f.read_text()
    assert "def compute(x):" in content
    assert "result = compute(5)" in content
    assert "calculate" not in content


def test_rename_updates_imports(tmp_path: Path):
    lib = tmp_path / "lib.py"
    lib.write_text("def foo():\n    return 42\n")

    main = tmp_path / "main.py"
    main.write_text("from lib import foo\n\nprint(foo())\n")

    result = rename_python(str(lib), "foo", "bar", str(tmp_path))

    assert len(result.files_changed) == 2

    assert "def bar():" in lib.read_text()
    main_content = main.read_text()
    assert "from lib import bar" in main_content
    assert "print(bar())" in main_content


def test_rename_class(tmp_path: Path):
    f = tmp_path / "models.py"
    f.write_text(
        "class UserProfile:\n"
        "    def __init__(self, name):\n"
        "        self.name = name\n"
        "\n"
        "p = UserProfile('alice')\n"
    )

    result = rename_python(str(f), "UserProfile", "Account", str(tmp_path))

    content = f.read_text()
    assert "class Account:" in content
    assert "p = Account(" in content
    assert "UserProfile" not in content


def test_rename_reports_changed_files(tmp_path: Path):
    f = tmp_path / "app.py"
    f.write_text("x = 1\n")

    result = rename_python(str(f), "x", "y", str(tmp_path))

    assert isinstance(result, RenameResult)
    assert isinstance(result.files_changed, list)
    assert all(isinstance(p, str) for p in result.files_changed)
