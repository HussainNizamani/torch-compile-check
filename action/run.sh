#!/usr/bin/env bash
# The composite action's "Run torch-compile-check" step, as a file.
#
# Split out of action.yml for the reason summary.sh was (see its header): a
# `run:` block can only be tested by copying it into a test, and the copy drifts
# from the block it claims to cover. tests/test_action_run.py drives *this* file
# with the same environment action.yml hands it, so the loop below is the loop
# CI runs.
#
# Everything it reads comes from the environment, one variable per action input,
# and nothing is interpolated into it: `${{ inputs.x }}` stays in the YAML's
# `env:` block, so a target path containing a quote cannot rewrite this script.
#
# What it does, per PLAN.md "GitHub Action": disable torch's compile caches by
# default so a run measures the current compiler (cache: true opts out), loop
# over the targets, collect the worst exit code, write a job-summary row per
# target through summary.sh -- exit code, graph-break count, stage, and a
# minimized block under the table when --minimize ran -- and degrade honestly on
# a pre-M1-3 ref, where the CLI main path is not implemented yet (see the
# allow-unimplemented input; main is past M1-3 already).
#
# There is deliberately no `set -e`.
#
# That is not laziness, it is the M4-1 verifier's finding. Under errexit the
# `stage=$(... | grep -m1 ... )` assignment below aborts the whole step whenever
# a target's output carries neither marker -- which is every tool error: a bad
# flag, a missing target, an unknown --fail-on category, a discovery failure.
# The step then exited 1 with no exit-code output, a truncated summary, and the
# remaining targets never ran, so one typo in one target silently cancelled the
# check on all the others. A CI step whose job is to report what happened must
# not be killed by the reporting. Every command whose status matters is checked
# explicitly instead, and `finish` is the only way out, so the outputs are
# written on every path.
set -uo pipefail

: "${GITHUB_STEP_SUMMARY:=/dev/null}"
: "${GITHUB_OUTPUT:=/dev/null}"
: "${GITHUB_ACTION_PATH:=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"

# One variable per input of the "Run torch-compile-check" step in action.yml. Listed
# rather than defaulted: a default here would be a second copy of action.yml's
# defaults, and the copy that drifts is the one nobody reads. A test asserts
# this list and that `env:` block name the same variables.
REQUIRED_ENV=(
	TARGETS
	BACKENDS
	FAIL_ON
	BASELINE
	WRITE_BASELINE
	MINIMIZE
	BUDGET
	CACHE
	JSON_OUT
	EXTRA_ARGS
	ALLOW_UNIMPLEMENTED
)

worst=0
minimized_sections=""

# The only exit. Writes the two step outputs first, because a caller with
# `continue-on-error: true` reads exit-code to decide what to do next, and an
# output that exists only on the happy path is an output that is missing exactly
# when it is needed.
finish() {
	local code="$1"
	[ "$code" -gt "$worst" ] && worst="$code"
	{
		echo "exit-code=$worst"
		echo "json-path=${JSON_OUT-}"
	} >>"$GITHUB_OUTPUT"
	[ -n "$minimized_sections" ] && rm -f "$minimized_sections"
	exit "$worst"
}

missing=()
for name in "${REQUIRED_ENV[@]}"; do
	[ -n "${!name+set}" ] || missing+=("$name")
done
if [ "${#missing[@]}" -gt 0 ]; then
	echo "::error::action/run.sh: unset input environment variable(s): ${missing[*]}" >&2
	JSON_OUT="${JSON_OUT-}"
	finish 2
fi

# The three boolean inputs are compared against the string "true", so anything
# else is off. Reject a third value rather than silently reading it as false:
# `cache: yes` quietly measuring the wrong thing for a month is exactly the
# failure this tool exists to complain about.
for pair in "minimize=$MINIMIZE" "cache=$CACHE" "allow-unimplemented=$ALLOW_UNIMPLEMENTED"; do
	case "${pair#*=}" in
	true | false) ;;
	*)
		echo "::error::input ${pair%%=*} must be \"true\" or \"false\", got \"${pair#*=}\"" >&2
		finish 2
		;;
	esac
done

# One baseline file cannot hold two targets' graph health: it is keyed by
# backend, not by target (docs/action.md "Baseline semantics"). So a
# write-baseline run takes exactly one target, instead of letting the last
# target silently overwrite what the others wrote.
if [ -n "$WRITE_BASELINE" ] &&
	[ "$(printf '%s\n' "$TARGETS" | grep -c '[^[:space:]]')" -gt 1 ]; then
	echo "::error::write-baseline takes a single target, since one baseline file is keyed by backend and not by target" >&2
	finish 2
fi

# The one-line reason a tool error gives, for the status cell. The CLI prints
# every tool error as `torch-compile-check: <sentence>` on stderr (cli.py
# `_tool_error`), and argparse's own failures take the same shape
# (`torch-compile-check: error: ...`), so the last such line is the sentence a reader
# wants. Pipes are escaped and the line is capped: the cell lives in a Markdown
# table row, and a raw `|` there would split it into columns.
tool_error_reason() {
	local line
	line="$(printf '%s\n' "$1" | grep -E '^torch-compile-check: ' | tail -n 1)"
	line="${line#torch-compile-check: }"
	# argparse writes "torch-compile-check: error: <sentence>"; the cell already says
	# "tool error", so the second "error:" is a word of noise in a narrow column.
	line="${line#error: }"
	[ -n "$line" ] || return 0
	line="${line//|/\\|}"
	if [ "${#line}" -gt 160 ]; then
		line="${line:0:159}…"
	fi
	printf '%s' "$line"
}

index=0
version="$(torch-compile-check --version 2>/dev/null)"
[ -n "$version" ] || version="version unknown"
# Collected during the loop and appended after the table: a <details> block per
# target that ran the minimizer, which has to come after the rows rather than
# between two of them.
minimized_sections="$(mktemp)"

{
	echo "## torch-compile-check results ($version)"
	echo
	echo "| target | exit code | status | graph breaks | stage |"
	echo "|---|---|---|---|---|"
} >>"$GITHUB_STEP_SUMMARY"

# `read -r target` and not `IFS= read -r target`: with one variable, read strips
# the leading and trailing whitespace of the line, which is what a YAML block
# scalar leaves behind on an indented `targets:` list. Without it a line of two
# spaces is a target named "  " and gets an import error of its own.
while read -r target; do
	[ -z "$target" ] && continue
	index=$((index + 1))

	target_json="${JSON_OUT%.json}.${index}.json"
	# Deleted before the run, not after it: summary.sh reads the graph-break
	# cell out of this file, and a file an earlier invocation left at the same
	# path would give a target that never got as far as running a break count
	# it never measured.
	rm -f "$target_json"

	args=(torch-compile-check "$target" --backends "$BACKENDS" --fail-on "$FAIL_ON" --json "$target_json")
	[ -n "$BASELINE" ] && args+=(--baseline "$BASELINE")
	[ -n "$WRITE_BASELINE" ] && args+=(--write-baseline "$WRITE_BASELINE")
	[ "$MINIMIZE" = "true" ] && args+=(--minimize)
	# --budget bounds --minimize and nothing else (PLAN.md "GitHub Action"), so
	# passing it without minimize: true gets one line on stderr from the CLI
	# saying the ceiling had nothing to bound.
	[ -n "$BUDGET" ] && args+=(--budget "$BUDGET")
	# cache: true means "keep torch's compile caches", which is the CLI's
	# --allow-caches; the default leaves the CLI to set
	# TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 so the run measures the current
	# compiler. The report records which mode was in force.
	[ "$CACHE" = "true" ] && args+=(--allow-caches)
	# shellcheck disable=SC2206
	[ -n "$EXTRA_ARGS" ] && args+=($EXTRA_ARGS)

	output="$("${args[@]}" 2>&1)"
	code=$?

	echo "$output"

	# The stage verdict is the line right after the terminal report's "stage"
	# heading (report/terminal.py): "first diverges at <backend>, ..." or
	# "clean: ...". Absent entirely on a path that never got as far as a
	# comparison (--probe, an unimplemented run, a tool error before
	# discovery), so an empty cell there is accurate, not a parsing miss --
	# and `|| true` because grep finding nothing is that accurate answer and
	# not a failure of this step.
	stage="$(printf '%s\n' "$output" |
		grep -m1 -E '^[[:space:]]*(first diverges at|clean:)' |
		sed 's/^[[:space:]]*//' || true)"

	if [ "$code" -eq 2 ] && printf '%s' "$output" | grep -q "not implemented"; then
		status="not implemented on this ref (pre-M1-3; PR #6 implements the main path on main)"
		if [ "$ALLOW_UNIMPLEMENTED" = "true" ]; then
			row_code=0
		else
			row_code=2
			worst=2
		fi
	elif [ "$code" -eq 2 ]; then
		# A tool error: a bad flag, a missing target, an unknown --fail-on
		# category, a target that would not import. It gets a row like every
		# other target and the loop carries on to the next one -- a typo in
		# one line of `targets` must not cancel the check on the others.
		reason="$(tool_error_reason "$output")"
		status="tool error${reason:+: $reason}"
		row_code=2
		worst=2
	else
		status="exit $code"
		row_code=$code
		[ "$code" -gt "$worst" ] && worst=$code
	fi

	# The row's graph-break cell and the minimized block are read from the JSON
	# the run just wrote, not from the terminal output: they are structured
	# fields (backends[].graph.break_count and the top-level `minimized` object
	# of schema_version 2) and re-parsing prose for them would be a second,
	# worse copy of the report.
	bash "$GITHUB_ACTION_PATH/summary.sh" \
		row "$target" "$row_code" "$status" "${stage:--}" "$target_json" \
		>>"$GITHUB_STEP_SUMMARY"
	bash "$GITHUB_ACTION_PATH/summary.sh" \
		minimized "$target" "$target_json" >>"$minimized_sections"
done <<<"$TARGETS"

if [ "$index" -eq 0 ]; then
	# `targets` is a required input, so this is whitespace rather than absence.
	# An empty table under a green check would say "nothing diverged" about a
	# run that checked nothing.
	echo "::error::targets is empty: give at least one path[:entry][:inputs] line" >&2
	echo "| _(no targets)_ | 2 | tool error: the targets input has no non-blank line | - | - |" \
		>>"$GITHUB_STEP_SUMMARY"
	finish 2
fi

if [ -s "$minimized_sections" ]; then
	{
		echo
		echo "### Minimized"
		echo
		cat "$minimized_sections"
	} >>"$GITHUB_STEP_SUMMARY"
fi

finish 0
