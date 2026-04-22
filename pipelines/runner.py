# runner.py

from __future__ import annotations
import argparse
from argparse import RawTextHelpFormatter
import os
import sys
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional
from pathlib import Path

# === Help/Epilogue for CLI ===
HELP_EPILOG = r'''Step selection (use names exactly as defined in pipeline.yaml):

  --only STEP              Run only STEP. By default, upstream dependencies are included.
  --no-upstream            Use with --only: do not run dependencies; execute STEP in isolation.

  --from STEP              Run from STEP to the end (including required dependencies).
  --to STEP                Run from the beginning through STEP (including required dependencies).
  --after STEP             Run only steps that depend (directly or indirectly) on STEP.
  --before STEP            Run only steps that STEP depends on (upstream of STEP).

  --skip STEP [STEP...]    Remove these steps from the plan (and everything depending on them is re-evaluated).
  --keep-going             Continue the pipeline even if a step fails (collect errors at the end).
  --list                   Only print the computed execution plan and exit (do not run anything).

Precedence rules (when options are combined):
  1) --only (with or without --no-upstream) takes precedence over any other filter.
  2) Then ranges: --from/--to/--after/--before (they can be combined).
  3) Finally, --skip is applied to the resulting plan.
  4) --list prints the final plan (after applying all rules) and does not execute it.

Dependency behavior:
  - Without --no-upstream, the runner always includes the required dependencies for each selected step.
  - With --no-upstream, the runner executes the step in isolation (without guaranteeing prerequisites).

Examples:
  # 1) Run only 'preprocess' with dependencies
  python runner.py --only preprocess

  # 2) Run only 'preprocess' in isolation (without dependencies)
  python runner.py --only preprocess --no-upstream

  # 3) Run from 'train' to the end
  python runner.py --from train

  # 4) Run from the beginning through 'pio_build'
  python runner.py --to pio_build

  # 5) Run everything that occurs after 'quantize_optimizing'
  python runner.py --after quantize_optimizing

  # 6) Run only what comes before 'pio_upload'
  python runner.py --before pio_upload

  # 7) Skip some steps in the middle of the flow
  python runner.py --skip header_generator pio_pull_model

  # 8) Only list the computed plan (without executing)
  python runner.py --list

  # 9) Continue even if a step fails
  python runner.py --from train --keep-going'''


try:
    import yaml
except Exception:
    print("[runner] ERROR: PyYAML not found. Install it with:\n  python -m pip install pyyaml", file=sys.stderr)
    sys.exit(1)
# ============================
# Data model
# ============================

@dataclass
class Step:
    name: str
    script: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    cwd: Optional[str] = None
    depends_on: List[str] = field(default_factory=list)
    continue_on_error: bool = False

    def cmd(self, python_exe: str) -> List[str]:
        # Always execute with the current Python interpreter to avoid launcher issues
        return [python_exe, str(self.script), *self.args]


@dataclass
class Pipeline:
    steps: Dict[str, Step]
    order: List[str]  # declarative order for cases without dependencies

    @staticmethod
    def load(yaml_path: Path) -> "Pipeline":
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        raw_steps = data.get("steps") or data.get("pipeline") or data
        if not isinstance(raw_steps, list):
            raise ValueError("Invalid YAML file: expected 'steps: [...]'")

        steps: Dict[str, Step] = {}
        order: List[str] = []
        for item in raw_steps:
            name = item["name"]
            order.append(name)
            steps[name] = Step(
                name=name,
                script=item["script"],
                args=list(item.get("args", [])),
                env=dict(item.get("env", {})),
                cwd=item.get("cwd"),
                depends_on=list(item.get("depends_on", [])),
                continue_on_error=bool(item.get("continue_on_error", False)),
            )
        return Pipeline(steps=steps, order=order)

    def graph(self) -> Dict[str, Set[str]]:
        """Return dep->step adjacency."""
        adj: Dict[str, Set[str]] = {name: set() for name in self.steps}
        for s in self.steps.values():
            for dep in s.depends_on:
                if dep not in self.steps:
                    raise KeyError(f"Step '{s.name}' depends on '{dep}' which does not exist.")
                adj.setdefault(dep, set()).add(s.name)
        return adj

    def reverse_graph(self) -> Dict[str, Set[str]]:
        """Return step->deps adjacency (for traversing upstream dependencies)."""
        rev: Dict[str, Set[str]] = {name: set() for name in self.steps}
        for s in self.steps.values():
            rev[s.name] = set(s.depends_on)
        return rev


# ============================
# Selection / ordering algorithms
# ============================

def topo_sort(subset: Set[str], rev_graph: Dict[str, Set[str]]) -> List[str]:
    """Topological sort honoring dependencies contained in the subset."""
    indeg = {n: 0 for n in subset}
    children = {n: set() for n in subset}
    for n in subset:
        for d in rev_graph[n]:
            if d in subset:
                indeg[n] += 1
                children.setdefault(d, set()).add(n)

    ready = [n for n in subset if indeg[n] == 0]
    plan: List[str] = []
    while ready:
        n = sorted(ready)[0]
        ready.remove(n)
        plan.append(n)
        for ch in children.get(n, set()):
            indeg[ch] -= 1
            if indeg[ch] == 0:
                ready.append(ch)

    if len(plan) != len(subset):
        missing = subset - set(plan)
        raise RuntimeError(f"Cycle detected or dependencies outside the subset: {missing}")
    return plan


def collect_downstream(start: Set[str], graph: Dict[str, Set[str]]) -> Set[str]:
    """BFS in the dep->step direction (everything downstream)."""
    out = set()
    frontier = list(start)
    while frontier:
        n = frontier.pop(0)
        if n in out:
            continue
        out.add(n)
        for ch in graph.get(n, set()):
            frontier.append(ch)
    return out


def collect_upstream(targets: Set[str], rev_graph: Dict[str, Set[str]]) -> Set[str]:
    """Collect direct/indirect dependencies upstream (step->deps)."""
    out = set()
    frontier = list(targets)
    while frontier:
        n = frontier.pop(0)
        if n in out:
            continue
        out.add(n)
        for d in rev_graph.get(n, set()):
            frontier.append(d)
    return out


def select_plan(pipeline: Pipeline, args: argparse.Namespace) -> List[str]:
    names = set(pipeline.steps.keys())
    g  = pipeline.graph()
    rg = pipeline.reverse_graph()

    # Base case: if nothing is specified, run everything in topological order over the full DAG
    target_set: Set[str] = set(names)

    # Main filters
    if args.only:
        only_set = set(args.only)
        unknown = only_set - names
        if unknown:
            raise KeyError(f"--only with non-existent step(s): {sorted(unknown)}")
        if args.no_upstream:
            # Execute only the specified steps, without dependencies
            target_set = only_set
        else:
            # Include required dependencies
            target_set = collect_upstream(only_set, rg)
    elif args.from_step or args.to_step or args.after or args.before:
        target_set = set()
        if args.from_step:
            if args.from_step not in names:
                raise KeyError(f"--from: non-existent step '{args.from_step}'")
            target_set |= collect_downstream({args.from_step}, g)
        if args.to_step:
            if args.to_step not in names:
                raise KeyError(f"--to: non-existent step '{args.to_step}'")
            # everything that leads to 'to' (traverse dependencies upward) plus 'to' itself
            target_set |= collect_upstream({args.to_step}, rg)
        if args.after:
            if args.after not in names:
                raise KeyError(f"--after: non-existent step '{args.after}'")
            downstream = collect_downstream({args.after}, g)
            target_set |= (downstream - {args.after})
        if args.before:
            if args.before not in names:
                raise KeyError(f"--before: non-existent step '{args.before}'")
            # everything preceding 'before'
            target_set |= (collect_upstream({args.before}, rg) - {args.before})

        if not target_set:
            # no effective filter -> no plan
            raise ValueError("No steps were selected by the provided filters.")
    else:
        # No filters: run everything
        target_set = names

    # Topological ordering of the subset
    plan = topo_sort(target_set, rg)

    # If the user passed --only with --no-upstream, preserve the user-provided order,
    # while still validating it with topo_sort (to detect conflicts).
    if args.only and args.no_upstream:
        # order the plan according to args.only (keeping only the selected steps)
        order_map = {name: i for i, name in enumerate(args.only)}
        plan = sorted([n for n in plan if n in order_map], key=lambda n: order_map[n])

    # --from + --to together: limit to the window
    if args.from_step and args.to_step:
        i_from = plan.index(args.from_step) if args.from_step in plan else 0
        i_to   = plan.index(args.to_step)   if args.to_step in plan else len(plan)-1
        if i_from > i_to:
            raise ValueError(f"--from '{args.from_step}' appears after --to '{args.to_step}' in the plan.")
        plan = plan[i_from:i_to+1]

    # Additional --after / --before trimming when combined with other filters
    if args.after and args.after in plan:
        idx = plan.index(args.after)
        plan = plan[idx+1:]
    if args.before and args.before in plan:
        idx = plan.index(args.before)
        plan = plan[:idx]

    return plan


# ============================
# Execution
# ============================

PROJECT_ROOT = Path(__file__).resolve().parent

def fmt_ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def run_step(step: Step, project_root: Path, keep_going: bool, dry_run: bool) -> int:
    # Prepare env (inherit from process + step overrides)
    env = os.environ.copy()
    env.update({k: str(v) for k, v in step.env.items()})
    # Always expose the project root (useful for scripts)
    env.setdefault("RUNNER_PROJECT_ROOT", str(project_root))

    # Working directory
    if step.cwd in (None, "", "null"):
        cwd = project_root
    else:
        # path relative to the project root (if not absolute)
        cwd = Path(step.cwd)
        if not cwd.is_absolute():
            cwd = project_root / cwd
        cwd = cwd.resolve()

    # Script path
    script_path = Path(step.script)
    if not script_path.is_absolute():
        script_path = (project_root / script_path).resolve()

    if not script_path.exists():
        print(f"[{fmt_ts()}] ERROR: script not found: {script_path}")
        return 1

    cmd = step.cmd(sys.executable)
    print(f"[{fmt_ts()}] Running '{step.name}': {cmd} (cwd={cwd})")
    if dry_run:
        return 0

    try:
        proc = subprocess.run(cmd, cwd=str(cwd), env=env)
        rc = proc.returncode
    except KeyboardInterrupt:
        print(f"[{fmt_ts()}] Interrupted by user at '{step.name}'.")
        return 130

    status = "OK" if rc == 0 else f"ERROR({rc})"
    print(f"[{fmt_ts()}] Finished '{step.name}' → {status}")
    if rc != 0 and not (keep_going or step.continue_on_error):
        print("[runner] Stopping pipeline due to failure (use --keep-going to continue).")
    return rc


def print_plan(pipeline: Pipeline, plan: List[str]):
    print("Execution plan:")
    for i, name in enumerate(plan, 1):
        s = pipeline.steps[name]
        deps = s.depends_on or []
        print(f"  {i:02d}. {name} -> {s.script} deps={deps}")


def main():
    ap = argparse.ArgumentParser(description="Pipeline runner (DAG)", epilog=HELP_EPILOG, formatter_class=RawTextHelpFormatter)
    ap.add_argument("--pipeline", "-p", default="pipeline.yaml", help="Pipeline YAML file")
    sel = ap.add_argument_group("Selection")
    sel.add_argument("--only", nargs="+", help="Run only the specified step(s). By default, dependencies are included.")
    sel.add_argument("--no-upstream", action="store_true", help="Use with --only: DO NOT include dependencies (run exactly the specified steps).")
    sel.add_argument("--from", dest="from_step", help="Start step (inclusive) and everything downstream")
    sel.add_argument("--to", dest="to_step", help="Up to the step (inclusive)")
    sel.add_argument("--after", help="Only steps that come after X (excluding X)")
    sel.add_argument("--before", help="Only steps that come before Y (excluding Y)")
    ap.add_argument("--list", action="store_true", help="Only list the plan, do not execute")
    ap.add_argument("--dry-run", action="store_true", help="Do not execute commands (print only)")
    ap.add_argument("--keep-going", action="store_true", help="Continue even if a step fails")
    args = ap.parse_args()

    yml_path = Path(args.pipeline)
    if not yml_path.is_file():
        print(f"[runner] ERROR: pipeline not found: {yml_path}")
        return 2

    pipeline = Pipeline.load(yml_path)
    plan = select_plan(pipeline, args)

    print_plan(pipeline, plan)
    if args.list or args.dry_run:
        return 0

    # Execute
    overall_rc = 0
    for name in plan:
        step = pipeline.steps[name]
        rc = run_step(step, project_root=PROJECT_ROOT, keep_going=args.keep_going, dry_run=False)
        if rc != 0:
            overall_rc = rc
            if not (args.keep_going or step.continue_on_error):
                break
    return overall_rc


if __name__ == "__main__":
    sys.exit(main())
