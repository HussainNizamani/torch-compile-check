#!/usr/bin/env bash
# Job-summary rendering for the composite action in action.yml.
#
# Split out of the action's "Run torch-compile-check" step rather than inlined in the
# YAML so that it can be executed outside a workflow: tests/test_action_summary.py
# runs this exact file against JSON reports the CLI has just written. A `run:`
# block could only be tested by copying it into the test, and the copy would
# drift from the block it claims to cover.
#
# Two subcommands, one per place the output lands:
#
#   row <target> <exit-code> <status> <stage> <report.json>
#       One Markdown table row. The graph cell is the graph oracle's break
#       count, read from the report: PLAN.md "graph" keeps breaks informational,
#       and the count is the number that explains why a user is not getting the
#       speedup they expect, which is worth a column even when nothing failed.
#
#   minimized <target> <report.json>
#       The block for the report's top-level `minimized` object (JSON
#       schema_version 2, PLAN.md "Minimizer, v1"), or nothing at all when the
#       run did not pass --minimize (`minimized` is null), when the document
#       predates schema 2 (no such key), or when there is no readable report.
#
# Both degrade to a placeholder instead of failing. A summary is a convenience
# beside the verdict, and a run that has already decided its exit code must not
# be turned red by the renderer that describes it.
set -uo pipefail

usage() {
	cat >&2 <<'USAGE'
usage: summary.sh row <target> <exit-code> <status> <stage> <report.json>
       summary.sh minimized <target> <report.json>
USAGE
}

# setup-python puts both names on PATH; a bare `python` is the fallback for a
# runner image where only that one exists. Only the standard library is used,
# so whichever interpreter answers is good enough to read a JSON document.
py() {
	if command -v python3 >/dev/null 2>&1; then
		python3 "$@"
	else
		python "$@"
	fi
}

# The graph cell: one number when every compiled lane agrees, "<lane> <n>" pairs
# when they do not, "-" when no lane measured graph health (the eager reference
# never does, and neither does a lane that raised).
graph_breaks() {
	local report="$1"
	if [ ! -f "$report" ]; then
		printf '%s\n' '-'
		return 0
	fi
	py - "$report" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], encoding="utf-8") as handle:
        document = json.load(handle)
    cells = []
    for record in document.get("backends") or []:
        if not isinstance(record, dict) or record.get("reference"):
            continue
        graph = record.get("graph")
        if not isinstance(graph, dict):
            continue
        name = str(record.get("backend", "?"))
        cells.append((name, int(graph.get("break_count", 0)) if graph.get("measured") else None))
    if not cells:
        print("-")
    elif len({count for _, count in cells}) == 1 and cells[0][1] is not None:
        print(cells[0][1])
    else:
        print(", ".join(f"{n} {'n/a' if c is None else c}" for n, c in cells))
except Exception:
    # Deliberately broad: a summary must not fail a run that already
    # produced a verdict, whatever shape the document turned out to be.
    print("-")
PY
}

minimized_block() {
	local target="$1" report="$2"
	[ -f "$report" ] || return 0
	py - "$target" "$report" <<'PY'
import json
import sys

target = sys.argv[1]
try:
    with open(sys.argv[2], encoding="utf-8") as handle:
        document = json.load(handle)
except Exception:
    # Same rule as above: no readable report, no section, no failure.
    raise SystemExit(0)

minimized = document.get("minimized")
if not isinstance(minimized, dict):
    # null (the run did not ask for --minimize) or absent (schema 1).
    raise SystemExit(0)


def shape(dims):
    return "(" + ", ".join(str(d) for d in dims or ()) + ")"


lines = [
    f"<details><summary><code>{target}</code> &mdash; "
    f"minimized: {minimized.get('summary') or 'ran'}</summary>",
    "",
]
finding = minimized.get("finding")
if isinstance(finding, dict):
    where = f"`{finding.get('oracle')}` on `{finding.get('backend')}`"
    if finding.get("output_index") is not None:
        where += f", output {finding['output_index']}"
    if finding.get("field"):
        where += f", field `{finding['field']}`"
    lines.append(f"- finding: {where} ({finding.get('severity')})")
for shrink in minimized.get("shrinks") or []:
    lines.append(
        f"- input: leaf {shrink.get('index')} {shape(shrink.get('before'))}"
        f" -> {shape(shrink.get('after'))}"
    )
for stub in minimized.get("stubs") or []:
    lines.append(
        f"- stubbed: `{stub.get('path')}` ({stub.get('module')}) -> `torch.nn.Identity()`"
    )
for kept in minimized.get("kept") or []:
    lines.append(f"- kept: `{kept.get('path')}` ({kept.get('module')}) &mdash; {kept.get('reason')}")
for note in minimized.get("notes") or []:
    lines.append(f"- note: {note}")
if minimized.get("partial"):
    lines.append(
        "- **partial**: "
        + str(minimized.get("partial_reason") or "the budget or the candidate ceiling ran out")
    )
lines.append(
    f"- cost: {minimized.get('steps', 0)} candidate re-runs"
    f" in {minimized.get('seconds', 0)}s"
)
# `handoff` is deliberately left out: it is the same constant paragraph of
# advice for every target, it is already in the step log the summary sits
# beside, and repeating it once per target would bury the rows above it.
lines += ["", "</details>", ""]
print("\n".join(lines))
PY
}

command="${1:-}"
shift || true

case "$command" in
row)
	[ "$#" -eq 5 ] || {
		usage
		exit 2
	}
	printf '| `%s` | %s | %s | %s | %s |\n' \
		"$1" "$2" "$3" "$(graph_breaks "$5")" "${4:--}"
	;;
minimized)
	[ "$#" -eq 2 ] || {
		usage
		exit 2
	}
	minimized_block "$1" "$2"
	;;
*)
	usage
	exit 2
	;;
esac
