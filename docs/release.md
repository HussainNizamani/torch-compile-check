# Releasing v0.1.0

The runbook for the four things M4 prepares and deliberately does not do: the
tag, the PyPI upload, the public flip, and the Marketplace listing. Each of them
is visible outside this machine and three of them cannot be undone, so each one
is the maintainer's own command, run knowingly, rather than something a merged
pull request sets off.

Read the whole file before running any of it. The order matters: everything up
to and including step 3 is reversible, and nothing after step 4 is.

Commands are written for a checkout of `main` at the commit you intend to
release, with the `dev` extra installed (`python -m pip install -e ".[dev]"`,
which is what brings in `build` and `twine`).

---

## 0. Preconditions

```console
$ git switch main && git pull --ff-only
$ git status --porcelain                 # must print nothing
$ grep '^version' pyproject.toml         # version = "0.1.0"
$ python -c "import torch_compile_check; print(torch_compile_check.__version__)"   # 0.1.0
$ ruff check . && ruff format --check . && mypy src/ && pytest -q
```

`CHANGELOG.md` still says `## [Unreleased] → 0.1.0` at this point, which is
accurate: the tag has not happened. Step 8 is where that heading changes.

Check that CI is green on the commit you are about to tag, not merely on some
recent commit. Asking for that commit's own check runs is the version-agnostic
way — `gh run list --branch` needs a `gh` newer than the 2.4 on this box:

```console
$ gh api "repos/HussainNizamani/torch-compile-check/commits/$(git rev-parse HEAD)/check-runs" \
    --jq '.check_runs[] | "\(.name): \(.conclusion)"'
py3.10 / torch stable: success
py3.10 / torch nightly: success
...
```

Every row must say `success`; eight of them, one per cell of the CI matrix.

**Visible to others:** nothing. Everything above is local or read-only.

---

## 1. Build the distributions

```console
$ rm -rf dist build
$ python -m build
$ twine check dist/*
```

Expect exactly two files, `dist/torch_compile_check-0.1.0.tar.gz` and
`dist/torch_compile_check-0.1.0-py3-none-any.whl`, and `PASSED` on both.

`twine check` validates the metadata and the long description that PyPI will
render. If it ever complains about the license fields, the fallback is in
`pyproject.toml`'s own comment: swap `license = "MIT"` for
`license = {text = "MIT"}` and add `License :: OSI Approved :: MIT License`
back to the classifiers. As of hatchling 1.32 and twine 7.0 the PEP 639 form
passes and produces `License-Expression: MIT`.

**Visible to others:** nothing. `dist/` is gitignored.

---

## 2. Install the built artifacts somewhere clean

Not the same as running the tests: this is the step that catches a packaging
mistake the test suite cannot see, because the suite runs against the source
tree.

```console
$ python -m venv /tmp/cc-release && /tmp/cc-release/bin/python -m pip install -q --upgrade pip
$ /tmp/cc-release/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
$ /tmp/cc-release/bin/pip install dist/torch_compile_check-0.1.0-py3-none-any.whl
$ /tmp/cc-release/bin/torch-compile-check --version          # torch-compile-check 0.1.0
$ /tmp/cc-release/bin/torch-compile-check --probe
$ cp cases/dtype_promotion.py /tmp/case.py
$ /tmp/cc-release/bin/torch-compile-check /tmp/case.py ; echo "exit $?"    # exit 1
```

`--probe` is read for the *same* rows it prints in a source checkout, not for
"all present": three are `absent` on a current torch and always have been
(`torch._inductor.config.repro_after` and the two `CompileProfiler` spellings).
A row that moved from `present` to `absent` between the checkout and the wheel
would mean the packaging dropped something, which is what this is here for.

Exit 1 is the expected answer for that case on a torch where the bug
reproduces, and exit 0 on one where it is fixed; either way the point is that
the installed console script ran the whole pipeline. Repeat with the sdist, so
the source distribution is proved to build rather than assumed to:

```console
$ /tmp/cc-release/bin/pip uninstall -y torch-compile-check
$ /tmp/cc-release/bin/pip install --no-binary torch-compile-check dist/torch_compile_check-0.1.0.tar.gz
$ /tmp/cc-release/bin/torch-compile-check --version
$ /tmp/cc-release/bin/torch-compile-check /tmp/case.py ; echo "exit $?"    # exit 1
$ rm -rf /tmp/cc-release
```

`--no-binary torch-compile-check` because pip will otherwise reuse the wheel it
already built from this version and the sdist path goes untested.

**Visible to others:** nothing.

---

## 3. Tag

```console
$ git tag -a v0.1.0 -m "torch-compile-check 0.1.0"
$ git push origin v0.1.0
```

**Visible to others:** a tag under `Tags` and `Releases` on the repository page,
to anyone who can see the repository — nobody outside the collaborators while it
is private. **Reversible**, awkwardly: `git tag -d v0.1.0` and
`git push origin :refs/tags/v0.1.0` remove it, but anyone who already fetched
keeps their copy. Do not reuse a tag name for different content.

Optionally, a GitHub release. Its notes are the `[Unreleased] → 0.1.0` section
of `CHANGELOG.md` — that is still its heading at this point, because step 8 is
where it is renamed:

```console
$ sed -n '/^## \[Unreleased\]/,/^\[Unreleased\]:/p' CHANGELOG.md > /tmp/notes.md
$ gh release create v0.1.0 --draft --title "torch-compile-check 0.1.0" --notes-file /tmp/notes.md
```

`--draft` so it can be read before anyone else sees it; drop it, or publish the
draft from the web UI, when it looks right. A release is also where the
Marketplace checkbox of step 7 lives.

---

## 4. TestPyPI first

```console
$ twine upload --repository testpypi dist/*
$ python -m venv /tmp/cc-testpypi
$ /tmp/cc-testpypi/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
$ /tmp/cc-testpypi/bin/pip install --index-url https://test.pypi.org/simple/ --no-deps torch-compile-check
$ /tmp/cc-testpypi/bin/torch-compile-check --version
$ rm -rf /tmp/cc-testpypi
```

`--no-deps` because TestPyPI does not carry torch.

**Visible to others:** a public page at
`https://test.pypi.org/project/torch-compile-check/`, the README rendered on it, and
the author name in the metadata. **Not reversible in the way that matters:** a
filename that has been uploaded can never be uploaded again, even after
deleting the release, so a mistake here costs a version number on TestPyPI.
That is the whole reason this step exists — it costs a TestPyPI version number
instead of a real one.

---

## 5. PyPI

```console
$ twine upload dist/*
```

Use an API token as the password (`__token__` as the username), not an
account password. The *first* upload of a new project cannot use a
project-scoped token — PyPI cannot scope a token to a project that does not
exist there yet — so it has to run under an account-scoped token (one that
can publish any of the account's projects). Once this upload creates the
`torch-compile-check` project on PyPI, go to its own Publishing settings and mint a
token scoped to just that project, and use that narrower one for every upload
after this first one; then revoke or forget the account-scoped token so
nothing keeps standing access to every other project on the account. (5b
below replaces this token dance entirely, for the tagged-release path.)

**Visible to others: permanently and to everyone.**
`https://pypi.org/project/torch-compile-check/` becomes a public page carrying the
README, the metadata, the author name, and both files. `pip install
torch-compile-check` starts working worldwide. **This cannot be undone.** A release
can be *yanked* (`pip` stops resolving it for new installs) but not removed,
the files stay downloadable by exact version, and the version number can never
be reused. Nothing about the upload is reversible enough to treat it as a
rehearsal — that is what step 4 is.

Verify from a clean environment:

```console
$ python -m venv /tmp/cc-pypi
$ /tmp/cc-pypi/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
$ /tmp/cc-pypi/bin/pip install torch-compile-check
$ /tmp/cc-pypi/bin/torch-compile-check --version
$ rm -rf /tmp/cc-pypi
```

---

## 5b. Trusted Publishing (no token)

An alternative to steps 4 and 5 above: `.github/workflows/publish.yml`, triggered
by pushing a tag, instead of a maintainer running `twine upload` by hand. It
uses PyPI's Trusted Publishing, where GitHub's own OIDC token stands in for the
account- or project-scoped token of step 5 — no token exists anywhere, not in a
secret, not on this machine, not in shell history.

### One-time setup, before the first tagged push

**On PyPI** — and, separately, on TestPyPI; the two are different accounts with
different project registries, so this whole flow runs twice:

1. Sign in at `https://pypi.org` (and, separately, `https://test.pypi.org`).
2. Account → Publishing → "Add a new pending publisher".
3. Fill in:
   - **PyPI project name**: `torch-compile-check` — the name `pyproject.toml`'s
     `name =` field carries once the m4-5-rename slice merges. If a release
     ever has to happen before that lands, register whatever `name =` says at
     the time instead; the registered project name and the built package's
     name have to match exactly, or the upload job fails.
   - **Owner**: `HussainNizamani`
   - **Repository name**: `torch-compile-check`
   - **Workflow name**: `publish.yml`
   - **Environment name**: `pypi` on pypi.org, `testpypi` on test.pypi.org

This does **not** require the repository to be public. GitHub issues the OIDC
token from a private repository the same as it does from a public one, and
PyPI checks the token's claims — owner, repository, workflow file, environment
— against the pending-publisher record above, not against the repository's
visibility. That record has to exist before the first run, on both PyPI and
TestPyPI, or the corresponding upload job fails with "no matching pending
publisher" or equivalent.

**On GitHub**: Settings → Environments → `pypi` (create it if it does not
exist yet) → Required reviewers, and again for `testpypi`. Name whoever should
see and approve the upload before it runs. Without a required reviewer an
environment auto-approves the moment the job reaches it, and a tag push
publishes with nobody in the loop — decide the reviewer list before the first
tag, not after.

### How the flow runs

```console
$ git tag v0.1.0 && git push origin v0.1.0
```

Pushing a `v*` tag is what starts it. (`workflow_dispatch` also runs the
`build` job, for rebuilding the artifact on its own, but `publish-testpypi`
only runs `if: github.event_name == 'push'`, and `publish-pypi` needs
`publish-testpypi` — so a manual dispatch never uploads anywhere.) From there:

1. `build`: checks out the tag, runs `python -m build` and `twine check dist/*`
   — the same commands as step 1 above — and uploads `dist/` as a workflow
   artifact.
2. `publish-testpypi`: downloads that artifact and uploads it to TestPyPI,
   under the `testpypi` environment — gated on that environment's required
   reviewers, if any were configured. `skip-existing: true` so re-running the
   same tag does not fail on a filename TestPyPI already has.
3. `publish-pypi`: waits on the `pypi` environment's required reviewers — this
   is the manual approval step; the job sits queued until someone with access
   clicks Approve on the workflow run's page — then uploads the same artifact
   to PyPI.

**Visible to others:** the same as steps 4 and 5 above, once each job actually
uploads — a public TestPyPI project page, then a public, permanent PyPI page.
The workflow run itself, and who approved `publish-pypi`, are visible to
anyone who can see this repository's Actions tab.

---

## 6. The public flip

Making the repository public is a separate decision from shipping the package,
and it is the one that exposes history rather than a snapshot. Check, in this
order:

- [ ] **Git history**, not just the working tree. A file deleted in a later
      commit is still in the history of a public repository:

      ```console
      $ git log -p --all | grep -nEi "ghp_|github_pat_|BEGIN [A-Z ]*PRIVATE KEY|AKIA[0-9A-Z]{16}|xox[baprs]-"
      ```

      Zero hits as of the M4-3 branch. Widen it (`token`, `secret`,
      `password`) and read what that finds too — it is noisy, and reading the
      noise is the job.
- [ ] **Absolute paths and machine names** in committed output:
      `grep -rn "/home/ubuntu" --include="*.md" --include="*.py" .` — pasted
      terminal output is the usual source. One hit as of the M4-3 branch, and
      it is this line.
- [ ] **`validation/results/*.json`** and `docs/validation.md`: they carry the
      platform string of the machine that produced them. Intended, but look at
      what is actually in them.
- [ ] **Issue and PR references** in `FINDINGS.md`, `CHANGELOG.md` and
      `PLAN.md`: they point at public PyTorch issues, which is fine, and at
      internal numbers, which read as this repository's own once it is public.
- [ ] **`PLAN.md`** in full. It is a design document written for an internal
      audience; decide deliberately whether it ships as it is.
- [ ] **CI**: workflows start consuming public-repository Actions minutes and
      the self-test's `source: git` path starts working for the first time
      (`action/README.md` "Installing from source").

Then, on the repository's Settings page: *General* → *Danger Zone* → *Change
visibility* → *Make public*, and type the repository name to confirm.

**Visible to others: everything, including every commit ever pushed.** GitHub
lets you make a repository private again, but anything that was fetched, forked,
cached by a search engine, or archived while it was public stays out.

---

## 7. The Marketplace listing

`uses: HussainNizamani/torch-compile-check/action@v0.1.0` works for any consumer as
soon as the repository is public, with or without a listing. The Marketplace is
discoverability, not a requirement.

**Blocker to settle first:** GitHub requires an action's metadata file to be in
the **root** of the repository for it to be published to the Marketplace, and
ours is `action/action.yml` so that the composite action sits beside the package
it installs. Confirm the current rule in GitHub's "Publishing actions in GitHub
Marketplace" documentation before spending time on this, then pick one:

1. Move `action.yml` to the repository root at release time and adjust
   `$GITHUB_ACTION_PATH` usage (`run.sh` and `summary.sh` are found relative to
   it, so they move too). Costs the tidy layout.
2. Publish the action from a small repository of its own that pins a version of
   this one. Costs a second repository to keep in step.
3. Do not list it. Costs discoverability and nothing else; the `uses:` line
   above still works.

If you go ahead, the rest of the checklist is:

- [ ] Repository public (step 6).
- [ ] Two-factor authentication enabled on the account.
- [ ] `action.yml` has `name`, `description`, `author`, and a `branding` block
      with an icon and a colour — it does; `name: "torch-compile-check"` must be
      unique across the Marketplace.
- [ ] A `README.md` next to the metadata file describing the action —
      `action/README.md` is written for exactly this.
- [ ] Draft a release (step 3), tick **Publish this Action to the GitHub
      Marketplace**, accept the agreement, choose the categories, publish.

**Visible to others:** a public listing page, the action's README, and the
author account. A listing can be unpublished later; the release and the tag it
points at stay.

---

## 8. After the upload

Once `pip install torch-compile-check` genuinely works, four things in the repository
stop being true and should be fixed in one small pull request:

- `CHANGELOG.md`: change `## [Unreleased] → 0.1.0` to `## [0.1.0] - YYYY-MM-DD`
  and start a fresh empty `## [Unreleased]` above it.
- `README.md` "Quick start": the `pip install git+https://...` paragraph and
  the sentence about the repository being private are both obsolete; the
  `pip install torch-compile-check` block becomes the first one.
- `README.md` badges. They are held back deliberately, because a badge for a
  private repository or an unpublished package renders as an error image. The
  lines to uncomment, in the placeholder comment under the title:

  ```markdown
  [![CI](https://github.com/HussainNizamani/torch-compile-check/actions/workflows/ci.yml/badge.svg)](https://github.com/HussainNizamani/torch-compile-check/actions/workflows/ci.yml)
  [![PyPI](https://img.shields.io/pypi/v/torch-compile-check.svg)](https://pypi.org/project/torch-compile-check/)
  [![Python](https://img.shields.io/pypi/pyversions/torch-compile-check.svg)](https://pypi.org/project/torch-compile-check/)
  ```

- `pyproject.toml` and `torch_compile_check.__version__`: bump both to the next
  development version, so `main` cannot be mistaken for the released artifact.
  `tests/test_cli.py` fails if only one of them moves.
