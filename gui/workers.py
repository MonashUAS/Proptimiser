## Background execution, so JAX work never blocks the Qt event loop

# External Imports
import sys
import threading
import traceback

from PySide6.QtCore import QObject, QTimer, Signal


class Task(QObject):
    '''
    Runs fn(report) on a daemon thread. report(value) emits `progress`; the return value arrives through
    `done`, an exception's traceback through `failed`, and `finished` follows either one. Signals are
    delivered on the Qt thread, so connected slots may touch widgets.

    Daemon threads let the application quit even while a long JIT compile is running.
    '''
    progress = Signal(object)
    done = Signal(object)
    failed = Signal(str)
    finished = Signal()

    _running = set()        # keeps started tasks alive until they finish

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self.fn = fn
        self.finished.connect(self._release)

    def start(self):
        Task._running.add(self)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            result = self.fn(self.progress.emit)
        except BaseException:
            text = traceback.format_exc()
            sys.stderr.write(text)
            self.failed.emit(text)
        else:
            self.done.emit(result)
        finally:
            self.finished.emit()

    def _release(self):
        Task._running.discard(self)


class LogStream(QObject):
    '''
    File-like tee for sys.stdout / sys.stderr: text still reaches the original stream, and is also emitted
    through `text` on the Qt thread in batches (writes may come from any thread, including JAX callbacks).
    '''
    text = Signal(str)
    encoding = "utf-8"
    errors = "replace"

    def __init__(self, original, parent=None):
        super().__init__(parent)
        self.original = original
        self._buffer = []
        self._lock = threading.Lock()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._emit)
        self._timer.start(100)

    def write(self, s):
        if self.original is not None:
            try:
                self.original.write(s)
            except (UnicodeEncodeError, ValueError):
                pass
        with self._lock:
            self._buffer.append(s)
        return len(s)

    def flush(self):
        if self.original is not None:
            self.original.flush()

    def isatty(self):
        return False

    def fileno(self):
        # Some libraries probe fileno(); report the original stream's descriptor when it has one
        if self.original is None:
            raise OSError("LogStream has no file descriptor")
        return self.original.fileno()

    def _emit(self):
        with self._lock:
            text = "".join(self._buffer)
            self._buffer.clear()
        if text:
            self.text.emit(text)
