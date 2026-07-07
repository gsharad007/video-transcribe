"""Model-free checks for the TUI's pure logic: catalog argv-building, the
carriage-return/newline stream demux, and the environment doctor.

The Textual UI itself is exercised by an optional pilot test at the bottom,
skipped when the 'tui' extra isn't installed.

Run: uv run python tests/tui_check.py
"""

from __future__ import annotations

from video_transcribe.tui_catalog import (
    CATALOG,
    CATEGORY_ORDER,
    ValidationError,
    build_tokens,
    grouped_catalog,
    split_paths,
)
from video_transcribe.tui_doctor import check_environment
from video_transcribe.tui_stream import LineDemux


def _events_from_bytes(*chunks: bytes) -> list[tuple[str, str]]:
    demux = LineDemux()
    out: list[tuple[str, str]] = []
    for c in chunks:
        out += demux.feed_bytes(c)
    out += demux.finish()
    return out


def test_catalog_wellformed():
    assert CATALOG, "catalog is empty"
    for key, task in CATALOG.items():
        assert task.key == key
        assert task.category in CATEGORY_ORDER, f"{key}: unknown category {task.category}"
        assert task.argv_prefix[0] == "-m", f"{key}: prefix must invoke a module"
        assert task.argv_prefix[1].startswith("video_transcribe"), f"{key}: {task.argv_prefix}"
        # unique arg names within a task (they become widget ids)
        names = [a.name for a in task.args]
        assert len(names) == len(set(names)), f"{key}: duplicate arg names {names}"
        for arg in task.args:
            if arg.choices and not arg.repeatable and not arg.required:
                assert str(arg.default) in arg.choices or str(arg.default) == "", \
                    f"{key}.{arg.name}: default {arg.default!r} not in choices"
    print(f"[ok] catalog: {len(CATALOG)} tasks, all well-formed")


def test_grouped_order():
    grouped = grouped_catalog()
    seen = list(grouped)
    ordered = [c for c in CATEGORY_ORDER if c in seen]
    assert seen == ordered, seen
    # every task appears exactly once across the groups
    flat = [t.key for tasks in grouped.values() for t in tasks]
    assert sorted(flat) == sorted(CATALOG), flat
    print(f"[ok] grouping: categories in canonical order, {len(flat)} tasks placed")


def test_build_basic_transcribe():
    task = CATALOG["transcribe"]
    tokens = build_tokens(task, {"inputs": "a.mp4\nb.mp4"})
    assert tokens[:2] == ["-m", "video_transcribe"]
    # defaults (model=large-v3, device=cpu, ...) are dropped; only the files remain
    assert tokens[-2:] == ["a.mp4", "b.mp4"], tokens
    assert "--model" not in tokens, "unchanged default should be omitted"
    print("[ok] build: basic transcribe drops defaults, appends positional files")


def test_build_changed_values_and_flags():
    task = CATALOG["transcribe"]
    tokens = build_tokens(task, {
        "inputs": "clip.mp4",
        "model": "base",
        "format": "txt,srt",
        "no_vad": True,
        "quiet": False,
        "language": "en",
    })
    assert "--model" in tokens and tokens[tokens.index("--model") + 1] == "base"
    assert tokens.count("--format") == 2
    assert "--no-vad" in tokens
    assert "--quiet" not in tokens  # False bool omitted
    # flags come before the trailing positional
    assert tokens[-1] == "clip.mp4"
    print("[ok] build: changed value / repeatable / bool flags, positional last")


def test_build_prefix_flags_preserved():
    # diarize and list-tracks bake a flag into the prefix
    assert "--diarize" in build_tokens(CATALOG["diarize"], {"inputs": "m.mp4"})
    assert "--list-tracks" in build_tokens(CATALOG["list-tracks"], {"inputs": "m.mp4"})
    # subcommand shape: voiceprint enroll
    tokens = build_tokens(CATALOG["voiceprint-enroll"],
                          {"transcript": "t.json", "media": "m.mp4", "store": "vp.json"})
    assert tokens[:3] == ["-m", "video_transcribe.voiceprint", "enroll"]
    assert tokens[-2:] == ["t.json", "m.mp4"], tokens  # positionals in order, last
    assert "--store" in tokens
    print("[ok] build: prefix flags + subcommand positionals order correctly")


def test_build_required_and_numeric_validation():
    task = CATALOG["transcribe"]
    try:
        build_tokens(task, {"inputs": ""})
    except ValidationError as e:
        assert "inputs" in str(e)
    else:
        raise AssertionError("missing required inputs should raise")

    try:
        build_tokens(CATALOG["diarize"], {"inputs": "a.mp4", "speakers": "two"})
    except ValidationError as e:
        assert "speakers" in str(e) and "integer" in str(e)
    else:
        raise AssertionError("non-numeric --speakers should raise")

    try:
        build_tokens(task, {"inputs": "a.mp4", "format": "txt,pdf"})
    except ValidationError as e:
        assert "pdf" in str(e)
    else:
        raise AssertionError("bad format choice should raise")
    print("[ok] build: required / integer / choice validation")


def test_examples_valid():
    total = 0
    for key, task in CATALOG.items():
        arg_names = {a.name for a in task.args}
        for ex in task.examples:
            total += 1
            unknown = set(ex.values) - arg_names
            assert not unknown, f"{key}: example references unknown args {unknown}"
            # every example must build a complete, valid command
            tokens = build_tokens(task, dict(ex.values))
            assert tokens[:2] == list(task.argv_prefix[:2]), (key, tokens)
    assert total >= 15, f"expected a healthy set of examples, got {total}"
    print(f"[ok] examples: {total} runs, all reference real args and build cleanly")


def test_split_paths():
    assert split_paths("a.mp4\nb.mp4\n") == ["a.mp4", "b.mp4"]
    assert split_paths("  C:\\My Videos\\talk.mp4  ") == ["C:\\My Videos\\talk.mp4"]
    # a single line with quoted paths splits on the quotes, keeping backslashes
    assert split_paths('"C:\\a b\\x.mp4" "C:\\c\\y.mp4"') == ["C:\\a b\\x.mp4", "C:\\c\\y.mp4"]
    assert split_paths("") == []
    print("[ok] split_paths: newline list, spaced path, quoted CLI-style list")


def test_demux_committed_lines_crlf():
    # ordinary Windows output: CRLF terminated lines
    events = _events_from_bytes(b"hello\r\nworld\r\n")
    assert events == [("line", "hello"), ("line", "world")], events
    print("[ok] demux: CRLF lines commit as clean lines")


def test_demux_transient_progress():
    # a progress bar repaints with bare '\r', then a final CRLF frame
    stream = b"\r  10%\r  40%\r  100%\r\n"
    events = _events_from_bytes(stream)
    kinds = [k for k, _ in events]
    assert kinds.count("transient") == 3  # empty leading + 10% + 40%
    assert ("line", "  100%") in events, events  # final frame committed
    print("[ok] demux: '\\r' frames are transient, final frame commits")


def test_demux_split_crlf_across_chunks():
    # the '\r' and '\n' of a CRLF split across two reads must still be one line
    events = _events_from_bytes(b"line one\r", b"\nline two\r\n")
    assert events == [("line", "line one"), ("line", "line two")], events
    print("[ok] demux: CRLF split across chunk boundary is one newline")


def test_demux_utf8_split_across_chunks():
    # a multi-byte char split across reads is decoded without mojibake
    text = "café\n".encode("utf-8")
    events = _events_from_bytes(text[:4], text[4:])
    assert events == [("line", "café")], events
    print("[ok] demux: multi-byte UTF-8 split across chunks decodes cleanly")


def test_doctor_runs():
    checks = check_environment()
    names = {c.name for c in checks}
    assert {"python", "ffmpeg", "ffprobe", "faster-whisper"} <= names
    # every failing check offers a hint the user can act on
    for c in checks:
        if not c.ok:
            assert c.hint, f"{c.name}: failing check with no hint"
    print(f"[ok] doctor: {len(checks)} checks, failing ones all carry hints")


def test_tui_app_pilot():
    try:
        import textual  # noqa: F401
    except ImportError:
        print("[skip] tui pilot: textual not installed (uv sync --extra tui)")
        return

    import asyncio

    from video_transcribe.tui_catalog import CATALOG as _CAT
    from video_transcribe.tui_catalog import build_tokens as _build
    from video_transcribe.tui import TranscribeTUI

    async def _drive() -> None:
        app = TranscribeTUI()
        async with app.run_test() as pilot:
            await pilot.pause()
            # a task is auto-selected and its option widgets are mounted
            assert app._selected is not None
            task = _CAT[app._selected]
            inputs = app.query("#arg-inputs")
            assert inputs, "form widgets did not mount"
            # clicking an example's Load button fills the form; read it back
            assert app.query("#ex-0"), "example Load button did not mount"
            await pilot.click("#ex-0")
            await pilot.pause()
            loaded = _build(task, app._collect(task))
            expected = _build(task, dict(task.examples[0].values))
            assert loaded == expected, (loaded, expected)
            # and a manual edit still reads back through _collect
            inputs.first().load_text("clip.mp4")
            await pilot.pause()
            tokens = _build(task, app._collect(task))
            assert tokens[:2] == ["-m", "video_transcribe"], tokens
            assert tokens[-1] == "clip.mp4", tokens
            # filtering narrows the list to a known task and positions the cursor
            app.query_one("#filter").value = "mux"
            await pilot.pause()
            assert app._first_task_index() is not None
            assert app.query("#task-mux"), "filter did not surface the mux task"

            # end-to-end: run the fast, dependency-free doctor task as a real
            # subprocess and confirm output streamed + a final state was recorded
            app.query_one("#filter").value = ""
            await pilot.pause()
            await app._select_task("doctor")
            await pilot.pause()
            app._maybe_run()
            for _ in range(200):  # up to ~10s; doctor takes well under 1s
                await pilot.pause(0.05)
                if "doctor" not in app._runs and app._state.get("doctor") not in (None, "running"):
                    break
            assert app._state.get("doctor") in ("succeeded", "warned", "failed"), app._state
            assert any("environment" in ln.lower() for ln in app._log_buffer), \
                "doctor output did not reach the log"

    asyncio.run(_drive())
    print("[ok] tui pilot: mounts, reads back, filters, runs a subprocess end-to-end")


if __name__ == "__main__":
    test_catalog_wellformed()
    test_grouped_order()
    test_build_basic_transcribe()
    test_build_changed_values_and_flags()
    test_build_prefix_flags_preserved()
    test_build_required_and_numeric_validation()
    test_examples_valid()
    test_split_paths()
    test_demux_committed_lines_crlf()
    test_demux_transient_progress()
    test_demux_split_crlf_across_chunks()
    test_demux_utf8_split_across_chunks()
    test_doctor_runs()
    test_tui_app_pilot()
    print("\nALL TUI CHECKS PASSED")
