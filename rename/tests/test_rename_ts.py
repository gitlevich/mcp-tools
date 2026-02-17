import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from rename_ts import RenameResult, rename_ts


def _make_ts_project(tmp_path: Path, files: dict[str, str]) -> Path:
    """Create a minimal TS project with tsconfig.json and given files."""
    src = tmp_path / "src"
    src.mkdir()
    tsconfig = tmp_path / "tsconfig.json"
    tsconfig.write_text(
        json.dumps(
            {
                "compilerOptions": {
                    "target": "ES2020",
                    "module": "commonjs",
                    "strict": True,
                    "baseUrl": ".",
                    "paths": {"*": ["src/*"]},
                },
                "include": ["src/**/*.ts"],
            }
        )
    )
    for name, content in files.items():
        (src / name).write_text(content)
    return tsconfig


def test_rename_function_single_file(tmp_path: Path):
    tsconfig = _make_ts_project(
        tmp_path,
        {
            "math.ts": (
                "export function calculate(x: number): number {\n"
                "  return x * 2;\n"
                "}\n"
                "\n"
                "const result = calculate(5);\n"
            ),
        },
    )

    result = rename_ts(
        str(tmp_path / "src" / "math.ts"),
        "calculate",
        "compute",
        str(tsconfig),
    )

    assert result.old_name == "calculate"
    assert result.new_name == "compute"
    assert len(result.files_changed) >= 1

    content = (tmp_path / "src" / "math.ts").read_text()
    assert "function compute(" in content
    assert "calculate" not in content


def test_rename_updates_imports(tmp_path: Path):
    tsconfig = _make_ts_project(
        tmp_path,
        {
            "lib.ts": "export function foo(): number { return 42; }\n",
            "main.ts": (
                'import { foo } from "./lib";\n'
                "\n"
                "console.log(foo());\n"
            ),
        },
    )

    result = rename_ts(
        str(tmp_path / "src" / "lib.ts"),
        "foo",
        "bar",
        str(tsconfig),
    )

    assert len(result.files_changed) == 2

    lib_content = (tmp_path / "src" / "lib.ts").read_text()
    assert "function bar(" in lib_content

    main_content = (tmp_path / "src" / "main.ts").read_text()
    assert "import { bar }" in main_content
    assert "bar()" in main_content


def test_rename_interface(tmp_path: Path):
    tsconfig = _make_ts_project(
        tmp_path,
        {
            "types.ts": (
                "export interface UserProfile {\n"
                "  name: string;\n"
                "}\n"
                "\n"
                "const p: UserProfile = { name: 'alice' };\n"
            ),
        },
    )

    result = rename_ts(
        str(tmp_path / "src" / "types.ts"),
        "UserProfile",
        "Account",
        str(tsconfig),
    )

    content = (tmp_path / "src" / "types.ts").read_text()
    assert "interface Account" in content
    assert "const p: Account" in content
    assert "UserProfile" not in content


def test_rename_symbol_not_found(tmp_path: Path):
    tsconfig = _make_ts_project(
        tmp_path,
        {"empty.ts": "export const x = 1;\n"},
    )

    with pytest.raises(RuntimeError, match="not found"):
        rename_ts(
            str(tmp_path / "src" / "empty.ts"),
            "nonexistent",
            "whatever",
            str(tsconfig),
        )


def test_rename_reports_result(tmp_path: Path):
    tsconfig = _make_ts_project(
        tmp_path,
        {"app.ts": "export const count = 1;\n"},
    )

    result = rename_ts(
        str(tmp_path / "src" / "app.ts"),
        "count",
        "total",
        str(tsconfig),
    )

    assert isinstance(result, RenameResult)
    assert isinstance(result.files_changed, list)
    assert all(isinstance(p, str) for p in result.files_changed)
