#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate a Markdown report of the packages installed in the current environment.

Usage:
  # with the virtual environment ACTIVATED
  python report_env_md.py
  python report_env_md.py --output deps.md

  # without activating the virtual environment (calling the Python executable inside .venv)
  .venv/bin/python report_env_md.py
  .venv/bin/python report_env_md.py --output deps.md
"""
from __future__ import annotations
import argparse
import platform
import sys
from datetime import datetime

# Python 3.8+ provides importlib.metadata; fall back to pkg_resources if needed
try:
    from importlib.metadata import distributions  # type: ignore[attr-defined]
    def iter_pkgs():
        for d in distributions():
            name = getattr(d, "metadata", {}).get("Name") if hasattr(d, "metadata") else None
            if not name and hasattr(d, "metadata"):  # py3.11+
                name = d.metadata["Name"]
            yield (name or d.metadata["Name"], d.version)
except Exception:
    import pkg_resources  # type: ignore
    def iter_pkgs():
        for dist in pkg_resources.working_set:
            yield (dist.project_name, dist.version)

def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a Markdown table of installed packages")
    ap.add_argument("--output", "-o", default="packages_report.md", help="Output file (.md)")
    args = ap.parse_args()

    # collect environment data
    rows = sorted(((n or "").strip(), (v or "").strip()) for n, v in iter_pkgs())
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pyver = platform.python_version()
    sysinfo = f"{platform.system()} {platform.release()}"
    venv_hint = sys.prefix

    # build markdown output
    lines = []
    lines.append(f"# Environment Package Report\n")
    lines.append(f"- **Recorded on:** {now}\n- **Python:** {pyver}\n- **OS:** {sysinfo}\n- **prefix:** `{venv_hint}`\n")
    lines.append("\n## Installed packages\n")
    lines.append("| Package | Version |")
    lines.append("|--------|--------|")
    for name, ver in rows:
        lines.append(f"| {name} | {ver} |")

    # include an optional requirements-style section
    lines.append("\n## requirements.txt (reference)\n")
    lines.append("```txt")
    for name, ver in rows:
        if name and ver:
            lines.append(f"{name}=={ver}")
    lines.append("```")
    content = "\n".join(lines)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"OK! Report saved to {args.output}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

