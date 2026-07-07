"""Split a subprocess text stream into committed lines vs transient progress
frames.

The CLI (and the tqdm bars inside faster-whisper / pyannote / Hugging Face
downloads) repaint progress in place with a bare carriage return -- e.g.
``\\r  transcribing [####----] 45.0%`` -- and only emit a real newline for the
final frame and for ordinary log lines. A naive line reader would either buffer
those repaints until a newline arrived or spam one log line per repaint.

:class:`LineDemux` classifies each piece as either a committed ``line`` (ended
with ``\\n``) or a ``transient`` frame (ended with a lone ``\\r``). The TUI shows
transient frames in a single live status widget it overwrites, and appends only
committed lines to the scrollback log -- so progress animates without flooding
the log, and the final frame still lands there.

``\\r\\n`` (every ordinary line on Windows) counts as one newline, including when
the pair is split across two ``feed`` calls.
"""

from __future__ import annotations

import codecs
from typing import Iterator, Literal

__all__ = ("LineDemux", "Event", "EventKind")

EventKind = Literal["line", "transient"]
Event = tuple[EventKind, str]


class LineDemux:
    """Incremental carriage-return / newline demultiplexer. One per stream."""

    __slots__ = ("_decoder", "_line", "_pending_cr")

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._line = ""
        self._pending_cr = False

    def feed_bytes(self, chunk: bytes) -> list[Event]:
        """Decode a byte chunk and return the events it completes."""
        return self.feed(self._decoder.decode(chunk))

    def feed(self, text: str) -> list[Event]:
        events: list[Event] = []
        if self._pending_cr:
            # A '\r' ended the previous chunk; only now can we tell whether it
            # was the '\r' of a '\r\n' newline or a bare progress-frame return.
            if text.startswith("\n"):
                events.append(("line", self._line))
                self._line = ""
                text = text[1:]
            else:
                events.append(("transient", self._line))
                self._line = ""
            self._pending_cr = False

        i, n = 0, len(text)
        while i < n:
            ch = text[i]
            if ch == "\n":
                events.append(("line", self._line))
                self._line = ""
            elif ch == "\r":
                if i + 1 < n:
                    if text[i + 1] == "\n":
                        events.append(("line", self._line))
                        self._line = ""
                        i += 1  # consume the paired '\n'
                    else:
                        events.append(("transient", self._line))
                        self._line = ""
                else:
                    self._pending_cr = True  # decide on the next feed
            else:
                self._line += ch
            i += 1
        return events

    def finish(self) -> list[Event]:
        """Flush at EOF. Any buffered text (a final line with no trailing
        newline, or a dangling '\\r') is committed as a line."""
        events: list[Event] = []
        tail = self._decoder.decode(b"", final=True)
        if tail:
            events += self.feed(tail)
        if self._pending_cr or self._line:
            events.append(("line", self._line))
            self._line = ""
            self._pending_cr = False
        return events


def iter_events(demux: LineDemux, chunks: Iterator[bytes]) -> Iterator[Event]:
    """Convenience wrapper: stream events from an iterator of byte chunks."""
    for chunk in chunks:
        yield from demux.feed_bytes(chunk)
    yield from demux.finish()
