# Upstream DAOC Source

`upstream/daoc` is a Git submodule pointing to the authors' public DAOC repository:

- Repository: https://github.com/MR-Golzari/distributed-dag-offloading-caching
- Pinned commit: `15596b3137e2e0a61d8b36c073c8a250deb5f2f5`
- Commit date: 2026-06-09

The pinned submodule is the pristine upstream baseline. It must not be confused with `reproducible_code/`, which contains the audited DAOC reproduction path together with this project's extensions, protocols, tests, and OUR implementation.

## Clone And Initialize

Clone everything in one command:

```bash
git clone --recurse-submodules https://github.com/595986772/workflow.git
```

For an existing clone:

```bash
git submodule update --init --recursive
```

Verify the pinned revision:

```bash
git -C upstream/daoc rev-parse HEAD
```

The output must be `15596b3137e2e0a61d8b36c073c8a250deb5f2f5`.

## Run The Upstream Code

```bash
cd upstream/daoc
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py -folder demo_run -nuser 20 -nserver 10
```

The upstream README and `input.py` describe its available parameters. The paper-scale comparison protocol used in this project remains documented under `reproducible_code/`; running the upstream demo command alone does not reproduce this project's formal figures.

At the pinned commit, the unused legacy fragment `plotcoverge.py` begins with an unexpected indent and does not compile. No tracked file references it, and every other tracked Python file passes `py_compile` under the verified local Python 3.12 environment. The submodule is intentionally left unchanged so that it remains an exact upstream snapshot.

## Rights And Attribution

The upstream repository does not contain an explicit license file at the pinned commit. The submodule preserves authorship and points directly to the authors' repository instead of redistributing or relicensing their source. Use and modification remain subject to the upstream authors' terms.
