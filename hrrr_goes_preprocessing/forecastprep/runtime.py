"""Atomic artifacts, process locks, and graceful scheduler interruption."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import tempfile

from filelock import FileLock, Timeout
from .core import write_json


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


@contextmanager
def locked(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(directory / '.writer.lock'), timeout=1):
            yield
    except Timeout as exc:
        raise RuntimeError(f'Another process is writing to {directory}') from exc


def atomic_torch_save(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix='.partial')
    os.close(handle)
    try:
        torch.save(value, temporary)
        with open(temporary, 'rb') as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class StopRequest:
    """Signal handler only sets a flag. Main loops commit at safe boundaries."""
    def __init__(self):
        self.requested = False
        self.previous = {}

    def __enter__(self):
        for name in ('SIGTERM', 'SIGINT', 'SIGUSR1'):
            if hasattr(signal, name):
                number = getattr(signal, name)
                self.previous[number] = signal.signal(number, self._handle)
        return self

    def _handle(self, number, frame):
        self.requested = True

    def __exit__(self, *args):
        for number, handler in self.previous.items():
            signal.signal(number, handler)


def status(directory, state, **details):
    from datetime import datetime, timezone
    write_json(Path(directory) / 'status.json', {
        'state': state, 'updated_utc': datetime.now(timezone.utc).isoformat(), **details})


def source_id():
    """Bind experiment resumes to the implementation that created them."""
    from .core import digest
    package = Path(__file__).parent
    return digest({path.name: sha256(path) for path in sorted(package.glob('*.py'))})
