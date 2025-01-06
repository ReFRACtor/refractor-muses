from contextlib import contextmanager
import os


@contextmanager
def in_dir(p):
    if p in {"", "."}:
        yield
    else:
        cwd = os.getcwd()
        try:
            os.chdir(p)
            yield
        finally:
            os.chdir(cwd)
