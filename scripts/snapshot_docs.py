"""
Snapshot the built docs into the committed static site.

Copies ``docs/_build`` to ``site/<major.minor>/`` (patch releases overwrite
their minor's folder), then regenerates ``site/versions.json`` — which the
version dropdown fetches at runtime — and points the catch-all rewrite in
``vercel.json`` at the newest version. Vercel serves matching files first, so
versioned URLs are served as-is and every other path (including ``/``) is
rewritten into the latest version without changing the URL. There must be no
``site/index.html``: it would shadow the rewrite at the root.

Usage:
    uv run python scripts/snapshot_docs.py                  # snapshot
    uv run python scripts/snapshot_docs.py --print-version  # print target folder name
"""

import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "docs" / "_build"
SITE = ROOT / "site"
VERCEL = ROOT / "vercel.json"


def current_minor() -> str:
    text = (ROOT / "pyproject.toml").read_text()
    full = re.search(r'^version = "([^"]+)"', text, re.M).group(1)
    return ".".join(full.split(".")[:2])


def main() -> None:
    minor = current_minor()
    if "--print-version" in sys.argv:
        print(minor)
        return

    if not (BUILD / "index.html").exists():
        sys.exit("docs/_build/index.html not found — run `make build-docs` first.")

    dest = SITE / minor
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(BUILD, dest, ignore=shutil.ignore_patterns(".doctrees", ".buildinfo"))

    versions = sorted(
        (d.name for d in SITE.iterdir() if d.is_dir() and re.fullmatch(r"\d+\.\d+", d.name)),
        key=lambda v: tuple(int(p) for p in v.split(".")),
        reverse=True,
    )
    latest = versions[0]

    payload = {"latest": latest, "versions": [{"version": v, "url": f"/{v}/"} for v in versions]}
    (SITE / "versions.json").write_text(json.dumps(payload, indent=2) + "\n")

    config = json.loads(VERCEL.read_text())
    config["rewrites"] = [
        # The root needs an explicit file target: Vercel's rewrite layer does
        # not resolve directory destinations to their index.html.
        {"source": "/", "destination": f"/{latest}/index.html"},
        {"source": "/:path*", "destination": f"/{latest}/:path*"},
    ]
    VERCEL.write_text(json.dumps(config, indent=2) + "\n")

    print(f"Snapshotted docs to site/{minor}/  (latest: {latest}; all: {', '.join(versions)})")


if __name__ == "__main__":
    main()
