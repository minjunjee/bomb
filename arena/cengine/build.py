"""Build libbomb.so (next to bomb.c) with gcc.

    python -m arena.cengine.build            # build if needed
    python -m arena.cengine.build --force    # rebuild

`import arena.cengine` calls ensure_built(): it builds when libbomb.so is missing, older than bomb.c / bomb.h, or
was built from other sources / flags / on another host (-march=native is host specific; the build id compiled into
the library says what it was built from). Safe when many processes import at once: the compile runs under an
exclusive lock file (.build.lock), goes to a temporary file in the same folder and is moved into place with one
atomic rename, so a reader sees either the old complete library or the new complete one.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import platform
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCES = (HERE / "bomb.c", HERE / "bomb.h")
LIB = HERE / "libbomb.so"
LOCK = HERE / ".build.lock"
CC = os.environ.get("CC", "gcc")
# No -ffast-math and no FMA contraction: bomb_features must reproduce the reference float32 bit patterns.
CFLAGS = ("-O3", "-march=native", "-fPIC", "-shared", "-fopenmp", "-std=c11", "-fno-fast-math", "-ffp-contract=off",
          "-Wall", "-Wextra")
_MARK = b"BOMB_BUILD_ID="


def build_id() -> str:
    """What a library built now would be built from: the sources, the flags, the compiler name and the host."""
    h = hashlib.sha256()
    for src in SOURCES:
        h.update(src.read_bytes())
    h.update(" ".join((CC, *CFLAGS, platform.node(), platform.machine())).encode())
    return h.hexdigest()[:20]


def _lib_id() -> str | None:
    """The build id compiled into the existing libbomb.so (None: no library or no id in it)."""
    try:
        data = LIB.read_bytes()
    except OSError:
        return None
    i = data.find(_MARK)
    if i < 0:
        return None
    j = data.find(b";", i)
    return data[i + len(_MARK):j].decode("ascii", "replace") if j > 0 else None


def needs_build() -> bool:
    try:
        lib_mtime = LIB.stat().st_mtime_ns
    except OSError:
        return True  # missing
    if any(src.stat().st_mtime_ns > lib_mtime for src in SOURCES):
        return True  # older than the sources
    return _lib_id() != build_id()  # other sources / flags / host


def build(force: bool = False, verbose: bool = False) -> Path:
    """Compile libbomb.so if needed (or always with force). Returns its path."""
    with open(LOCK, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # one builder at a time; the others wait here and then see a fresh library
        try:
            if not force and not needs_build():
                return LIB
            bid = build_id()
            tmp = HERE / f".libbomb.{os.getpid()}.{bid}.tmp"
            cmd = [CC, *CFLAGS, f'-DBOMB_BUILD_ID="{bid}"', "-o", str(tmp), str(SOURCES[0])]
            if verbose:
                print(" ".join(cmd))
            try:
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode != 0:
                    raise RuntimeError(f"building libbomb.so failed:\n{' '.join(cmd)}\n{r.stdout}{r.stderr}")
                if verbose and r.stderr:
                    print(r.stderr, file=sys.stderr)
                os.replace(tmp, LIB)  # atomic: processes that already loaded the old file keep their mapping
            finally:
                if tmp.exists():
                    tmp.unlink()
            return LIB
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def ensure_built() -> Path:
    """The path of an up-to-date libbomb.so (built now if needed)."""
    return build() if needs_build() else LIB


if __name__ == "__main__":
    was_stale = needs_build()
    path = build(force="--force" in sys.argv[1:], verbose=True)
    print(f"{path}  ({'built' if was_stale or '--force' in sys.argv[1:] else 'up to date'}, build id {_lib_id()})")
