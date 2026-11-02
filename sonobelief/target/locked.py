"""Exclusive lock for read-modify-write of shared summary CSVs (several generation runs may write them at once)."""
import contextlib
import fcntl


@contextlib.contextmanager
def locked(path):
    with open(path + ".lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield
