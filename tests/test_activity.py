import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from activity import ActivityReporter


def test_creates_activity_file(tmp_path):
    reporter = ActivityReporter("test-server", activity_dir=tmp_path)
    f = tmp_path / "test-server.json"
    assert f.exists()
    data = json.loads(f.read_text())
    assert data["server"] == "test-server"
    assert data["state"] == "idle"
    assert data["pid"] == os.getpid()


def test_report_sets_busy_then_idle(tmp_path):
    reporter = ActivityReporter("test-server", activity_dir=tmp_path)
    f = tmp_path / "test-server.json"

    with reporter.report("Testing operation"):
        data = json.loads(f.read_text())
        assert data["state"] == "busy"
        assert data["operation"] == "Testing operation"
        assert data["started_at"] is not None

    data = json.loads(f.read_text())
    assert data["state"] == "idle"
    assert data["operation"] is None


def test_recent_entries(tmp_path):
    reporter = ActivityReporter("test-server", activity_dir=tmp_path)
    f = tmp_path / "test-server.json"

    with reporter.report("Op 1"):
        pass
    with reporter.report("Op 2"):
        pass

    data = json.loads(f.read_text())
    assert len(data["recent"]) == 2
    assert data["recent"][0]["operation"] == "Op 2"
    assert data["recent"][1]["operation"] == "Op 1"
    assert data["recent"][0]["duration_s"] >= 0


def test_recent_capped_at_max(tmp_path):
    reporter = ActivityReporter("test-server", activity_dir=tmp_path)
    for i in range(25):
        with reporter.report(f"Op {i}"):
            pass
    data = json.loads((tmp_path / "test-server.json").read_text())
    assert len(data["recent"]) == 20


def test_progress(tmp_path):
    reporter = ActivityReporter("test-server", activity_dir=tmp_path)
    f = tmp_path / "test-server.json"

    with reporter.report("Indexing"):
        reporter.set_progress(5, 100, "batch 1")
        data = json.loads(f.read_text())
        assert data["progress"]["current"] == 5
        assert data["progress"]["total"] == 100
        assert data["progress"]["detail"] == "batch 1"

    data = json.loads(f.read_text())
    assert data["progress"] is None


def test_cleanup(tmp_path):
    reporter = ActivityReporter("test-server", activity_dir=tmp_path)
    f = tmp_path / "test-server.json"
    assert f.exists()
    reporter.cleanup()
    assert not f.exists()


def test_exception_preserves_idle_state(tmp_path):
    reporter = ActivityReporter("test-server", activity_dir=tmp_path)
    f = tmp_path / "test-server.json"

    try:
        with reporter.report("Failing op"):
            raise ValueError("boom")
    except ValueError:
        pass

    data = json.loads(f.read_text())
    assert data["state"] == "idle"
    assert len(data["recent"]) == 1
    assert data["recent"][0]["operation"] == "Failing op"


def test_second_instance_becomes_noop_if_owner_alive(tmp_path):
    """A new reporter backs off if the file is owned by a different live process."""
    f = tmp_path / "test-server.json"
    # Write a file owned by PID 1 (launchd, always alive on macOS)
    fake_state = {
        "server": "test-server",
        "pid": 1,
        "state": "idle",
        "operation": None,
        "started_at": None,
        "progress": None,
        "recent": [],
        "updated_at": 0,
    }
    f.write_text(json.dumps(fake_state))

    reporter = ActivityReporter("test-server", activity_dir=tmp_path)
    assert not reporter._active

    # File still has PID 1
    data = json.loads(f.read_text())
    assert data["pid"] == 1

    # Cleanup on inactive instance is a no-op
    reporter.cleanup()
    assert f.exists()


def test_takes_over_from_dead_process(tmp_path):
    """A new reporter takes over a file left by a dead process."""
    f = tmp_path / "test-server.json"
    # Write a fake file with a PID that doesn't exist
    fake_state = {
        "server": "test-server",
        "pid": 999999999,
        "state": "idle",
        "operation": None,
        "started_at": None,
        "progress": None,
        "recent": [],
        "updated_at": 0,
    }
    f.write_text(json.dumps(fake_state))

    reporter = ActivityReporter("test-server", activity_dir=tmp_path)
    assert reporter._active
    data = json.loads(f.read_text())
    assert data["pid"] == os.getpid()
