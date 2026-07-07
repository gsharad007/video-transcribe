"""Terminal UI for every video-transcribe command.

A Textual app driven by :data:`video_transcribe.tui_catalog.CATALOG`: pick a
task on the left, fill in its options on the right (every CLI flag rendered as a
widget), watch the exact command build live, then run and monitor it -- several
at once if you like. Progress bars animate on a single live line while the
scrollback log keeps the full, highlighted output.

Run it with::

    uv run video-transcribe-tui
    # or
    uv run python -m video_transcribe.tui

Needs the 'tui' extra (``uv sync --extra tui``). The commands it launches are the
same ``python -m video_transcribe ...`` invocations you'd type by hand, so the
tool's own extras still gate what each job can do -- run the Environment doctor
task if something won't start.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Final, Literal

from video_transcribe.tui_catalog import (
    CATALOG,
    CATEGORY_LABELS,
    Arg,
    Task,
    ValidationError,
    build_tokens,
    grouped_catalog,
)
from video_transcribe.tui_highlight import highlight_line, is_warning_line
from video_transcribe.tui_stream import LineDemux

try:
    import psutil
    from textual import events
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, Vertical, VerticalScroll
    from textual.css.query import NoMatches
    from textual.fuzzy import Matcher
    from textual.widget import Widget
    from textual.widgets import (
        Button,
        Footer,
        Header,
        Input,
        Label,
        ListItem,
        ListView,
        RichLog,
        Select,
        Static,
        Switch,
        TextArea,
    )
except ModuleNotFoundError as e:  # pragma: no cover - exercised only without the extra
    raise SystemExit(
        "the TUI needs the 'tui' extra -- run:  uv sync --extra tui\n"
        f"(missing module: {e.name})"
    ) from e

__all__ = ("main",)

_FILTER_ID: Final = "filter"
_LOG_BUFFER_MAX: Final = 20_000
_TERMINATE_TIMEOUT_SECONDS: Final = 2.0
_LOG_DIR = Path.home() / ".video-transcribe" / "tui-logs"

# Keys that drive control flow; every other printable key typed while the task
# list is focused is forwarded to the filter (search-as-you-type).
_RESERVED_KEYS: Final[frozenset[str]] = frozenset({"q", "r", "s", "c", "y", "w", "/"})

TaskState = Literal["idle", "running", "succeeded", "failed", "warned", "stopped"]

_STATE_MARKER: Final[dict[TaskState, str]] = {
    "idle": "   ", "running": ">> ", "succeeded": "ok ",
    "failed": "!! ", "warned": "~~ ", "stopped": "// ",
}
_STATE_COLOR: Final[dict[TaskState, str]] = {
    "idle": "", "running": "bold #FFD862", "succeeded": "bold #7AE582",
    "failed": "bold #FF8A8A", "warned": "bold #E09BFF", "stopped": "bold #7AE0E5",
}

FormValues = dict[str, "str | bool"]


@dataclass(slots=True)
class TaskRun:
    """One in-flight subprocess: the process, when it started, and whether the
    user asked it to stop. Kept together so they can't drift apart."""

    proc: asyncio.subprocess.Process
    start_time: float
    stop_requested: bool = False

    def duration(self) -> float:
        return time.monotonic() - self.start_time


def _final_state(exit_code: int, warnings: int) -> TaskState:
    if exit_code != 0:
        return "failed"
    return "warned" if warnings > 0 else "succeeded"


def _task_label(task: Task, state: TaskState) -> str:
    marker = _STATE_MARKER[state]
    color = _STATE_COLOR[state]
    return f"[{color}]{marker}{task.label}[/]" if color else f"{marker}{task.label}"


def _matches_filter(task: Task, query: str) -> bool:
    if not query:
        return True
    haystack = f"{task.key} {task.label} {task.summary} {task.category} {' '.join(task.tags)}"
    return Matcher(query, case_sensitive=False).match(haystack) > 0


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining = divmod(seconds, 60)
    return f"{int(minutes)}m{remaining:04.1f}s"


def _quote(token: str) -> str:
    return f'"{token}"' if (not token or " " in token) else token


def _arg_widget(arg: Arg) -> Widget:
    wid = f"arg-{arg.name}"
    if arg.kind == "bool":
        return Switch(value=bool(arg.default), id=wid)
    if arg.kind == "choice":
        options = [(c, c) for c in arg.choices]
        return Select(options, value=str(arg.default), allow_blank=False, id=wid)
    if arg.kind == "paths":
        return TextArea(text=str(arg.default), id=wid, classes="paths-area")
    return Input(value=str(arg.default), placeholder=arg.placeholder or arg.help, id=wid)


def _read_widget(widget: Widget) -> str | bool:
    if isinstance(widget, Switch):
        return widget.value
    if isinstance(widget, Select):
        return str(widget.value)
    if isinstance(widget, TextArea):
        return widget.text
    if isinstance(widget, Input):
        return widget.value
    raise TypeError(f"unexpected widget: {type(widget).__name__}")


def _terminate_tree(pid: int, timeout: float) -> None:
    """Kill the whole process tree. proc.terminate() only reaches the immediate
    child on Windows; ffmpeg / torch workers spawned underneath survive it."""
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    members = [*parent.children(recursive=True), parent]
    for member in members:
        with contextlib.suppress(psutil.NoSuchProcess):
            member.terminate()
    _, alive = psutil.wait_procs(members, timeout=timeout)
    for member in alive:
        with contextlib.suppress(psutil.NoSuchProcess):
            member.kill()


async def _terminate(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is not None:
        return
    await asyncio.to_thread(_terminate_tree, proc.pid, _TERMINATE_TIMEOUT_SECONDS)
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(proc.wait(), timeout=_TERMINATE_TIMEOUT_SECONDS)


class TranscribeTUI(App[int]):
    TITLE = "video-transcribe"
    SUB_TITLE = "transcribe / diarize / correct — configure, run, monitor"

    CSS = """
    Screen { layout: horizontal; }
    #sidebar { width: 42; border: solid $primary; }
    #sidebar-title { padding: 0 1; color: $accent; text-style: bold; }
    #filter { margin: 0 1 1 1; height: 3; }
    #main { width: 1fr; padding: 1 2; }
    #task-summary { margin-bottom: 1; color: $secondary; height: auto; }
    #preview { color: $text-muted; background: $boost; padding: 0 1; height: auto; margin-bottom: 1; }
    #form-area { height: 1fr; min-height: 6; border: round $panel; padding: 0 1; }
    #button-row { height: 3; align: left middle; }
    #button-row Button { margin-right: 1; min-width: 12; }
    #status { color: $accent; margin: 1 0 0 0; height: auto; }
    #progress { color: $warning; height: 1; }
    #log-area { border: solid $accent; padding: 0 1; height: 1fr; }
    .category-header { color: $accent; text-style: bold; background: $boost; padding: 0 1; }
    ListView > ListItem.--highlight { background: $primary 20%; border-left: thick $accent; }
    .arg-block { height: auto; margin-bottom: 1; }
    .arg-label { color: $text; height: auto; }
    .paths-area { height: 4; border: round $panel; }
    """

    BINDINGS: ClassVar = [
        ("q", "quit", "Quit"),
        ("r", "run_task", "Run"),
        ("s", "stop_task", "Stop"),
        ("c", "clear_log", "Clear"),
        ("w", "save_log", "Save log"),
        ("y", "copy_log", "Copy log"),
        ("slash", "focus_filter", "Filter"),
        ("escape", "focus_list", "Tasks"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._selected: str | None = None
        self._runs: dict[str, TaskRun] = {}
        self._state: dict[str, TaskState] = {}
        self._filter_query = ""
        self._last_result: tuple[str, TaskState, float] | None = None
        self._log_buffer: collections.deque[str] = collections.deque(maxlen=_LOG_BUFFER_MAX)

    # -- composition ------------------------------------------------------- #

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="sidebar"):
                yield Label(f"Tasks ({len(CATALOG)})", id="sidebar-title")
                yield Input(id=_FILTER_ID, placeholder="filter tasks...")
                yield ListView(id="task-list")
            with Vertical(id="main"):
                yield Static("Pick a task on the left.", id="task-summary")
                yield Static("", id="preview")
                yield VerticalScroll(id="form-area")
                with Horizontal(id="button-row"):
                    yield Button("Run (r)", id="run-button", variant="primary", disabled=True)
                    yield Button("Stop (s)", id="stop-button", variant="error", disabled=True)
                    yield Button("Clear (c)", id="clear-button")
                yield Static("idle", id="status")
                yield Static("", id="progress")
                yield RichLog(id="log-area", highlight=False, markup=False, wrap=True,
                              max_lines=_LOG_BUFFER_MAX)
        yield Footer()

    async def on_mount(self) -> None:
        await self._render_task_list()
        list_view = self.query_one("#task-list", ListView)
        list_view.focus()
        first = next(iter(CATALOG), None)
        if first is not None:
            list_view.index = 1  # row 0 is a category header
            self._select_task(first)

    async def on_unmount(self) -> None:
        for run in list(self._runs.values()):
            with contextlib.suppress(Exception):
                await _terminate(run.proc)

    # -- sidebar / task list ---------------------------------------------- #

    def _state_of(self, key: str) -> TaskState:
        return self._state.get(key, "idle")

    async def _render_task_list(self) -> None:
        """Rebuild the filtered, category-grouped sidebar. Awaits clear() before
        appending -- otherwise the append races the pending removal and Textual
        raises DuplicateIds on the next filter keystroke."""
        list_view = self.query_one("#task-list", ListView)
        await list_view.clear()
        for category, tasks in grouped_catalog().items():
            matching = [t for t in tasks if _matches_filter(t, self._filter_query)]
            if not matching:
                continue
            header = CATEGORY_LABELS.get(category, category).upper()
            list_view.append(ListItem(Label(f"  {header}"), id=f"header-{category}",
                                       classes="category-header"))
            for task in matching:
                list_view.append(ListItem(Label(_task_label(task, self._state_of(task.key))),
                                          id=f"task-{task.key}"))

    def _first_task_index(self) -> int | None:
        list_view = self.query_one("#task-list", ListView)
        for i, item in enumerate(list_view.children):
            if isinstance(item, ListItem) and (item.id or "").startswith("task-"):
                return i
        return None

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        if event.item is None or event.item.id is None:
            return
        if event.item.id.startswith("task-"):
            self._select_task(event.item.id.removeprefix("task-"))

    def _select_task(self, key: str) -> None:
        self._selected = key
        task = CATALOG[key]
        self.query_one("#task-summary", Static).update(f"[b]{task.label}[/]\n{task.summary}")
        self._rebuild_form(task)
        self._refresh_buttons()
        self._update_preview()

    def _rebuild_form(self, task: Task) -> None:
        form = self.query_one("#form-area", VerticalScroll)
        form.remove_children()
        if not task.args:
            form.mount(Static("[dim]No options — just run it.[/]"))
            return
        for arg in task.args:
            block = Vertical(classes="arg-block")
            form.mount(block)
            star = " [b red]*[/]" if arg.required else ""
            kind = "" if arg.kind in ("str", "paths") else f" [dim]({arg.kind})[/]"
            block.mount(Static(f"[b]{arg.name}[/]{star}{kind}  [dim]{arg.help}[/]",
                               classes="arg-label"))
            block.mount(_arg_widget(arg))

    # -- form reading / preview ------------------------------------------- #

    def _collect(self, task: Task) -> FormValues:
        values: FormValues = {}
        for arg in task.args:
            try:
                widget = self.query_one(f"#arg-{arg.name}", Widget)
            except NoMatches:
                continue  # form mid-rebuild
            values[arg.name] = _read_widget(widget)
        return values

    def _update_preview(self) -> None:
        try:
            preview = self.query_one("#preview", Static)
        except NoMatches:
            return
        if self._selected is None:
            preview.update("")
            return
        task = CATALOG[self._selected]
        try:
            tokens = build_tokens(task, self._collect(task))
        except ValidationError as e:
            preview.update(f"[#FF8A8A]needs input:[/] {e}")
            return
        cmd = "python " + " ".join(_quote(t) for t in tokens)
        preview.update(f"[dim]$[/] {cmd}")

    async def on_input_changed(self, event: Input.Changed) -> None:
        # Awaited inline (not via a worker): message handlers are serialized, so
        # rapid typing can't race _render_task_list against its own pending
        # clear(). A worker with exclusive=True would also share the default
        # worker group and cancel any running task -- exactly what we must not do.
        if event.input.id == _FILTER_ID:
            self._filter_query = event.value.strip().lower()
            await self._render_task_list()
            idx = self._first_task_index()
            if idx is not None:
                self.query_one("#task-list", ListView).index = idx
        elif (event.input.id or "").startswith("arg-"):
            self._update_preview()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == _FILTER_ID:
            list_view = self.query_one("#task-list", ListView)
            list_view.focus()
            idx = self._first_task_index()
            if idx is not None:
                list_view.index = idx

    def on_select_changed(self, event: Select.Changed) -> None:
        if (event.select.id or "").startswith("arg-"):
            self._update_preview()

    def on_switch_changed(self, event: Switch.Changed) -> None:
        if (event.switch.id or "").startswith("arg-"):
            self._update_preview()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if (event.text_area.id or "").startswith("arg-"):
            self._update_preview()

    # -- keys / focus ------------------------------------------------------ #

    def action_focus_filter(self) -> None:
        self.query_one(f"#{_FILTER_ID}", Input).focus()

    def action_focus_list(self) -> None:
        self.query_one("#task-list", ListView).focus()

    def on_key(self, event: events.Key) -> None:
        char = event.character
        if not char or len(char) != 1 or not char.isprintable() or char.isspace():
            return
        if char in _RESERVED_KEYS:
            return
        try:
            list_view = self.query_one("#task-list", ListView)
            filter_input = self.query_one(f"#{_FILTER_ID}", Input)
        except NoMatches:
            return
        if not list_view.has_focus:
            return
        filter_input.focus()
        filter_input.value += char
        event.stop()

    # -- buttons / actions ------------------------------------------------- #

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "run-button":
            self._maybe_run()
        elif event.button.id == "stop-button":
            self._maybe_stop()
        elif event.button.id == "clear-button":
            self.action_clear_log()

    def action_run_task(self) -> None:
        self._maybe_run()

    def action_stop_task(self) -> None:
        self._maybe_stop()

    def action_clear_log(self) -> None:
        self._log_buffer.clear()
        self.query_one("#log-area", RichLog).clear()

    def action_save_log(self) -> None:
        if not self._log_buffer:
            self._log("[tui] log is empty; nothing to save")
            return
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = _LOG_DIR / f"tui-{time.strftime('%Y%m%d-%H%M%S')}.log"
        path.write_text("\n".join(self._log_buffer) + "\n", encoding="utf-8")
        self._log(f"[tui] saved {len(self._log_buffer)} lines to {path}")

    def action_copy_log(self) -> None:
        if not self._log_buffer:
            self._log("[tui] log is empty; nothing to copy")
            return
        self.copy_to_clipboard("\n".join(self._log_buffer))
        self._log(f"[tui] copied {len(self._log_buffer)} lines to clipboard")

    def _refresh_buttons(self) -> None:
        try:
            run_btn = self.query_one("#run-button", Button)
            stop_btn = self.query_one("#stop-button", Button)
        except NoMatches:
            return
        running = self._selected is not None and self._selected in self._runs
        run_btn.disabled = self._selected is None or running
        stop_btn.disabled = not running

    def _maybe_run(self) -> None:
        if self._selected is None:
            return
        task = CATALOG[self._selected]
        if task.key in self._runs:
            self._log(f"[{task.key}] already running; ignored")
            return
        try:
            tokens = build_tokens(task, self._collect(task))
        except ValidationError as e:
            self._log(f"[{task.key}] cannot run: {e}")
            return
        self.run_worker(self._run_subprocess(task.key, tokens))

    def _maybe_stop(self) -> None:
        if self._selected is None:
            return
        run = self._runs.get(self._selected)
        if run is None:
            self._log(f"[{self._selected}] not running")
            return
        self._log(f"[{self._selected}] stop requested...")
        run.stop_requested = True
        self.run_worker(_terminate(run.proc))

    # -- logging / status -------------------------------------------------- #

    def _log(self, line: str) -> None:
        self._log_buffer.append(line)
        with contextlib.suppress(NoMatches):
            self.query_one("#log-area", RichLog).write(highlight_line(line))

    def _set_progress(self, key: str, frame: str) -> None:
        with contextlib.suppress(NoMatches):
            self.query_one("#progress", Static).update(f"{key}: {frame.strip()}" if frame.strip() else "")

    def _set_state(self, key: str, state: TaskState) -> None:
        if self._state.get(key, "idle") == state:
            return
        self._state[key] = state
        with contextlib.suppress(NoMatches):
            item = self.query_one(f"#task-{key}", ListItem)
            item.query_one(Label).update(_task_label(CATALOG[key], state))
        if key == self._selected:
            self._refresh_buttons()

    def _refresh_status(self) -> None:
        with contextlib.suppress(NoMatches):
            status = self.query_one("#status", Static)
            running = "idle" if not self._runs else f"{len(self._runs)} running: {', '.join(sorted(self._runs))}"
            last = ""
            if self._last_result is not None:
                key, state, dur = self._last_result
                color = _STATE_COLOR[state]
                body = f"{_STATE_MARKER[state].strip() or state} {key} ({_format_duration(dur)})"
                last = f"    |    last: [{color}]{body}[/]" if color else f"    |    last: {body}"
            status.update(running + last)

    # -- run loop ---------------------------------------------------------- #

    async def _run_subprocess(self, key: str, tokens: list[str]) -> None:
        argv = [sys.executable, "-u", *tokens]
        self._log(f"[{key}] $ python " + " ".join(_quote(t) for t in tokens))
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        proc = await asyncio.create_subprocess_exec(
            *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, env=env,
        )
        run = TaskRun(proc=proc, start_time=time.monotonic())
        self._runs[key] = run
        self._set_state(key, "running")
        self._refresh_status()
        try:
            warnings = await self._pump(key, proc)
            exit_code = await proc.wait()
            dur = run.duration()
            self._log(f"[{key}] --- exit_code={exit_code} warnings={warnings} "
                      f"duration={_format_duration(dur)} ---")
            state = "stopped" if run.stop_requested else _final_state(exit_code, warnings)
            self._set_state(key, state)
            self._last_result = (key, state, dur)
        except asyncio.CancelledError:
            await _terminate(proc)
            self._set_state(key, "stopped")
            raise
        finally:
            self._runs.pop(key, None)
            self._set_progress(key, "")
            if key == self._selected:
                self._refresh_buttons()
            self._refresh_status()

    async def _pump(self, key: str, proc: asyncio.subprocess.Process) -> int:
        stdout = proc.stdout
        if stdout is None:
            raise RuntimeError("subprocess started without a stdout pipe")
        demux = LineDemux()
        warnings = 0
        while True:
            chunk = await stdout.read(4096)
            if not chunk:
                break
            for kind, content in demux.feed_bytes(chunk):
                if kind == "line":
                    self._log(f"[{key}] {content}")
                    if is_warning_line(content):
                        warnings += 1
                elif content.strip():
                    self._set_progress(key, content)
        for kind, content in demux.finish():
            if kind == "line":
                self._log(f"[{key}] {content}")
                if is_warning_line(content):
                    warnings += 1
        return warnings


def main(argv: list[str] | None = None) -> int:
    TranscribeTUI().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
