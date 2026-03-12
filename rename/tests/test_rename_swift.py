import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from rename_swift import RenameResult, find_project_root, find_symbol_position, rename_swift


# ---------------------------------------------------------------------------
# Unit tests for helpers
# ---------------------------------------------------------------------------

def test_find_symbol_position_simple():
    source = "func calculate(x: Int) -> Int {\n    return x * 2\n}\n"
    line, char = find_symbol_position(source, "calculate")
    assert line == 0
    assert char == 5  # "func " is 5 chars


def test_find_symbol_position_second_line():
    source = "import Foundation\n\nclass Atlas {\n}\n"
    line, char = find_symbol_position(source, "Atlas")
    assert line == 2
    assert char == 6  # "class " is 6 chars


def test_find_symbol_position_not_found():
    source = "func foo() {}\n"
    with pytest.raises(ValueError, match="not found"):
        find_symbol_position(source, "bar")


def test_find_symbol_position_avoids_substring():
    source = "func recalculate() {}\n\nfunc calculate() {}\n"
    line, char = find_symbol_position(source, "calculate")
    assert line == 2
    assert char == 5


def test_find_project_root_spm(tmp_path: Path):
    pkg = tmp_path / "Package.swift"
    pkg.write_text("// swift-tools-version:5.9\n")
    nested = tmp_path / "Sources" / "Lib"
    nested.mkdir(parents=True)
    swift_file = nested / "Foo.swift"
    swift_file.write_text("struct Foo {}\n")

    root = find_project_root(str(swift_file))
    assert root == tmp_path


def test_find_project_root_xcodeproj(tmp_path: Path):
    proj = tmp_path / "App.xcodeproj"
    proj.mkdir()
    swift_file = tmp_path / "App" / "Main.swift"
    swift_file.parent.mkdir()
    swift_file.write_text("import Foundation\n")

    root = find_project_root(str(swift_file))
    assert root == tmp_path


def test_find_project_root_not_found(tmp_path: Path):
    lonely = tmp_path / "orphan.swift"
    lonely.write_text("let x = 1\n")
    with pytest.raises(ValueError, match="No Package.swift"):
        find_project_root(str(lonely))


# ---------------------------------------------------------------------------
# Integration tests — require sourcekit-lsp (ships with Xcode)
# ---------------------------------------------------------------------------

def _make_spm_package(tmp_path: Path, files: dict[str, str]) -> Path:
    """Create a minimal SPM package with the given source files."""
    pkg = tmp_path / "Package.swift"
    pkg.write_text(
        '// swift-tools-version:5.9\n'
        'import PackageDescription\n'
        'let package = Package(\n'
        '    name: "TestPkg",\n'
        '    targets: [\n'
        '        .target(name: "TestPkg", path: "Sources"),\n'
        '    ]\n'
        ')\n'
    )
    sources = tmp_path / "Sources"
    sources.mkdir()
    for name, content in files.items():
        (sources / name).write_text(content)
    return tmp_path


def _sourcekit_available() -> bool:
    import shutil
    return shutil.which("sourcekit-lsp") is not None


skip_no_sourcekit = pytest.mark.skipif(
    not _sourcekit_available(),
    reason="sourcekit-lsp not found",
)


@skip_no_sourcekit
def test_rename_function_single_file(tmp_path: Path):
    _make_spm_package(tmp_path, {
        "Math.swift": (
            "func calculate(_ x: Int) -> Int {\n"
            "    return x * 2\n"
            "}\n"
            "\n"
            "let result = calculate(5)\n"
        ),
    })

    result = rename_swift(
        str(tmp_path / "Sources" / "Math.swift"),
        "calculate",
        "compute",
    )

    assert result.old_name == "calculate"
    assert result.new_name == "compute"
    assert len(result.files_changed) >= 1

    content = (tmp_path / "Sources" / "Math.swift").read_text()
    assert "func compute(" in content
    assert "compute(5)" in content
    assert "calculate" not in content


@skip_no_sourcekit
def test_rename_struct(tmp_path: Path):
    _make_spm_package(tmp_path, {
        "Models.swift": (
            "struct UserProfile {\n"
            "    let name: String\n"
            "}\n"
            "\n"
            "let p = UserProfile(name: \"alice\")\n"
        ),
    })

    result = rename_swift(
        str(tmp_path / "Sources" / "Models.swift"),
        "UserProfile",
        "Account",
    )

    content = (tmp_path / "Sources" / "Models.swift").read_text()
    assert "struct Account" in content
    assert "Account(name:" in content
    assert "UserProfile" not in content


@skip_no_sourcekit
def test_rename_cross_file(tmp_path: Path):
    _make_spm_package(tmp_path, {
        "Lib.swift": "public func foo() -> Int { return 42 }\n",
        "Main.swift": "let x = foo()\n",
    })

    result = rename_swift(
        str(tmp_path / "Sources" / "Lib.swift"),
        "foo",
        "bar",
    )

    lib = (tmp_path / "Sources" / "Lib.swift").read_text()
    assert "func bar()" in lib

    main = (tmp_path / "Sources" / "Main.swift").read_text()
    assert "bar()" in main
    assert "foo" not in main

    assert len(result.files_changed) == 2


@skip_no_sourcekit
def test_rename_symbol_not_found(tmp_path: Path):
    _make_spm_package(tmp_path, {
        "Empty.swift": "let x = 1\n",
    })

    with pytest.raises(ValueError, match="not found"):
        rename_swift(
            str(tmp_path / "Sources" / "Empty.swift"),
            "nonexistent",
            "whatever",
        )


@skip_no_sourcekit
def test_rename_reports_result(tmp_path: Path):
    _make_spm_package(tmp_path, {
        "App.swift": "let count = 1\n",
    })

    result = rename_swift(
        str(tmp_path / "Sources" / "App.swift"),
        "count",
        "total",
    )

    assert isinstance(result, RenameResult)
    assert isinstance(result.files_changed, list)
    assert all(isinstance(p, str) for p in result.files_changed)
