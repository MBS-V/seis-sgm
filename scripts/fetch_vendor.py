"""Fetch pinned shallow copies of the authors' repos into ./vendor (git-ignored).
Usage:  python scripts/fetch_vendor.py [IPS-RNA] [VI-Non-IID]     (no args = all)
Stdlib only, so it runs the same on Windows and Colab."""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor"


def git(*args, cwd=None):
    return subprocess.run(["git", *args], cwd=cwd, check=True,
                          capture_output=True, text=True).stdout.strip()


def main():
    want = set(sys.argv[1:])
    for line in (ROOT / "vendor.lock").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, url, sha = line.split()
        if want and name not in want:
            continue
        dest = VENDOR / name
        if (dest / ".git").exists() and git("rev-parse", "HEAD", cwd=dest) == sha:
            print(f"[ok] {name} already at {sha[:8]}")
            continue
        dest.mkdir(parents=True, exist_ok=True)
        if not (dest / ".git").exists():
            git("init", "-q", cwd=dest)
            git("remote", "add", "origin", url, cwd=dest)
        print(f"[..] fetching {name} @ {sha[:8]} (shallow)")
        git("fetch", "--depth", "1", "origin", sha, cwd=dest)
        git("checkout", "-q", "FETCH_HEAD", cwd=dest)
        print(f"[ok] {name} at {git('rev-parse', 'HEAD', cwd=dest)[:8]}")


if __name__ == "__main__":
    main()
