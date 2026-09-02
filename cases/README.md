# cases

The regression corpus: one tiny model per known bug class, each carrying a
known-bad torch version marker. See PLAN.md "Regression corpus".

Two shapes of file live here, and the difference is deliberate.

- **Standalone RED/GREEN scripts**, one per case in the C-1 corpus
  (`alias_slice_scatter_copyback.py`, `alias_noop_view_identity.py`,
  `dtype_int8_matmul_promotion.py`, `distributions_validation_branch.py`,
  `numerics_cpu_inductor_miscompile.py`). Each runs itself, prints one
  `RED`/`GREEN` line with the torch version and build hash it measured, and
  exits non-zero when the bug reproduces. `FINDINGS.md` is the ground truth
  table they fill in.
- **compile-check targets**, following the discovery convention of PLAN.md: a
  module-level `model` or `fn` plus `inputs` or `get_inputs()`, and nothing
  else, so `compile-check cases/<file>.py` runs the case through the tool
  itself. `dtype_promotion.py` is the discovery-convention twin of
  `dtype_int8_matmul_promotion.py`, `alias_copyback.py` the twin of
  `alias_slice_scatter_copyback.py`, `alias_noop_view.py` the twin of
  `alias_noop_view_identity.py`, `distributions_binomial_kl.py` the twin of
  `distributions_validation_branch.py`, and `numerics_polyjuice_minmax.py`
  the twin of `numerics_cpu_inductor_miscompile.py`. `tests/test_corpus_twins.py`
  runs the standalone script and its twin together on every test run and
  asserts they agree, so the two files cannot drift apart silently.

## Adding a case

Write the standalone script first: reproduce the bug from the issue as
literally as you can (`build()` returning `(fn, example_inputs)`, `check()`
comparing eager against compiled, a `main()` that prints one `RED`/`GREEN`
line and exits accordingly), add its row to `FINDINGS.md`, and record a
version marker -- the torch build, git hash, and architecture the RED or
GREEN verdict was measured on, since a case can flip between them purely
because of which torch it runs against. Then write the discovery-convention
twin: the same reproducer, trimmed to a module-level `fn` (or `model`) plus
`inputs`, with nothing else at module scope, so `compile-check
cases/<twin>.py` exercises it through the real tool rather than the
standalone script's own comparison. Add the pair to
`tests/test_corpus_twins.py`'s parametrize list, naming the backend the
stage verdict lands on when the standalone script is RED (deterministic
across torch builds, since it follows the tool's fixed lane order rather
than anything environment-specific -- explain why in the twin's docstring
if the first *diverging* backend and the bug's actual fix location differ,
the way `alias_noop_view.py` does). A twin whose bug gets fixed upstream is
not deleted: its docstring records the version marker for RED and the test
already handles the GREEN case, which is the point of anchoring the
assertion to the standalone script's live verdict rather than a hardcoded
exit code.
