"""Allow one training, evaluation or demo controller per simulation workspace."""
from contextlib import contextmanager
import os
from pathlib import Path


@contextmanager
def exclusive_control(name, path=None):
    import fcntl
    path = Path(path or Path(__file__).parents[2]/"experiments/.controller.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            stream.seek(0)
            owner = stream.read().strip()
            raise RuntimeError(f"Simulation already controlled by {owner}. Stop that controller first.") from error
        try:
            stream.seek(0)
            stream.truncate()
            stream.write(f"{name} (process {os.getpid()})")
            stream.flush()
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
