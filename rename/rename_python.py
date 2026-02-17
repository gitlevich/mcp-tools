import logging
import re
from dataclasses import dataclass

from rope.base.project import Project
from rope.refactor.rename import Rename

logger = logging.getLogger(__name__)


@dataclass
class RenameResult:
    old_name: str
    new_name: str
    files_changed: list[str]


def find_symbol_offset(source: str, name: str) -> int:
    """Find byte offset of first whole-word occurrence of name in source."""
    pattern = re.compile(r"\b" + re.escape(name) + r"\b")
    match = pattern.search(source)
    if match is None:
        raise ValueError(f"Symbol '{name}' not found in source")
    return match.start()


def rename_python(
    file_path: str,
    old_name: str,
    new_name: str,
    project_root: str,
) -> RenameResult:
    """Rename a Python symbol across the project using Rope. Applies to disk."""
    project = Project(project_root, ropefolder=None)
    try:
        rel_path = str(
            __import__("pathlib").Path(file_path).relative_to(project_root)
        )
        resource = project.get_file(rel_path)
        source = resource.read()
        offset = find_symbol_offset(source, old_name)

        renamer = Rename(project, resource, offset)
        changes = renamer.get_changes(new_name)

        changed_files = [change.resource.path for change in changes.changes]

        project.do(changes)

        return RenameResult(
            old_name=old_name,
            new_name=new_name,
            files_changed=changed_files,
        )
    finally:
        project.close()
