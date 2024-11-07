import hashlib
from itertools import islice
from secrets import token_hex
import subprocess
from typing import Any, Iterable, Callable

__all__ = [
    "default_arg",
    "md5",
    "random_string",
    "shell",
    "take",
]


def default_arg[T](
    v: T,
    default: T | None = None,
    default_factory: Callable[[], T] | None = None,
):
    """Populate default parameters."""
    if v is not None:
        return v

    if default is None and default_factory is not None:
        return default_factory()

    return default


def random_string(length: int | None = None) -> str:
    """Generate random string of specified length."""
    # Defaults
    length = default_arg(length, lambda: 8)

    return token_hex(length // 2 + 1)[0:length]


def shell(command: str) -> str:
    """Run shell command."""
    result = subprocess.run(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise Exception(f"Command failed with error: {result.stderr}")

    return result.stdout.strip()


def md5(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()


def take(n: int, iterable: Iterable[Any]):
    """Process items n at a time."""
    it = iter(iterable)
    while True:
        chunk = tuple(islice(it, n))
        if not chunk:
            break
        yield chunk
