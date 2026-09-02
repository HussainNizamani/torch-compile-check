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
  itself. `dtype_promotion.py` is the first, and it is the discovery-convention
  twin of `dtype_int8_matmul_promotion.py`.
