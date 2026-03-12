# Plan: CLI for `condensed_hks_pipeline`

**TL;DR:** Typer-based `meshmash` CLI, structured as a package for extensibility. Wraps `condensed_hks_pipeline`, reads mesh via meshio, params from CLI flags or TOML. Outputs compressed `.npz`.

## Tool Recommendation: Typer

| Tool                  | Verdict                                                                                                                                  |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **argparse** (stdlib) | Verbose, no type coercion, no TOML. Skip.                                                                                                |
| **click**             | Mature, decorator-based, battle-tested. High boilerplate for 18 params.                                                                  |
| **typer**             | Built on Click, uses Python type annotations, bool `--flag/--no-flag` auto-generated, clean TOML via `ctx.default_map`. **Recommended.** |
| **cyclopts**          | Newer/smaller community. Not needed.                                                                                                     |
| **fire**              | No TOML support, poor help text. Skip.                                                                                                   |

## Phase 1 — Dependencies

1. Add `typer>=0.12` and `meshio` to `pyproject.toml` `[project.dependencies]`
2. Add `[project.scripts]` entry point: `meshmash = "meshmash.cli:app"` — the top-level `app` is a Typer group; future scripts attach as additional sub-apps

## Phase 2 — `src/meshmash/cli/` package

3. **`src/meshmash/cli/__init__.py`** — creates the root `app = typer.Typer()` and registers the `hks` sub-app
4. **`src/meshmash/cli/hks.py`** — `hks` Typer app with one command:
   - `mesh_path: Path` — positional
   - `--output / -o: Path` — required
   - `--config: Optional[Path]` — TOML; `is_eager=True` callback sets `ctx.default_map` (TOML = overrideable defaults, CLI always wins)
   - All 18 `condensed_hks_pipeline` params as typed `--options` with real defaults:
     - `simplify_agg: int = 7`
     - `simplify_target_reduction: Optional[float] = 0.7`
     - `overlap_distance: float = 20_000`
     - `max_vertex_threshold: int = 20_000`
     - `min_vertex_threshold: int = 200`
     - `max_overlap_neighbors: int = 40_000`
     - `n_components: int = 32`
     - `t_min: float = 5e4`
     - `t_max: float = 2e7`
     - `max_eigenvalue: float = 5e-6`
     - `robust: bool = True`
     - `mollify_factor: float = 1e-5`
     - `truncate_extra: bool = True`
     - `drop_first: bool = True`
     - `decomposition_dtype: str = "float32"`
     - `distance_threshold: float = 3.0`
     - `auxiliary_features: bool = True`
     - `n_jobs: int = -1`
     - `verbose: bool = False`
   - `decomposition_dtype: str = "float32"` — passed as a plain string directly to the pipeline, no conversion needed (`decompose.py` line 316 already handles `"float32"`/`"float64"` strings — no code changes required)
   - Skip: `compute_hks_kwargs` (too advanced), `nuc_point` (rarely used)
   - Mesh: `meshio.read(mesh_path)` → `.points` + `.cells_dict["triangle"]`; raise `typer.BadParameter` if no triangles found
   - Save: `save_condensed_features(output, result.condensed_features, result.labels)`

   Usage: `meshmash hks input.obj --output out.npz`

## Phase 3 — TOML config format

5. Keys are snake_case param names matching Typer's internal names; priority: CLI arg > TOML > code default

   ```toml
   n_components = 16
   distance_threshold = 2.5
   decomposition_dtype = "float64"
   ```

## Relevant files

- `pyproject.toml` — add deps + `[project.scripts]`
- `src/meshmash/cli/__init__.py` — new
- `src/meshmash/cli/hks.py` — new
- `src/meshmash/io.py` — `save_condensed_features` (line 63)
- `src/meshmash/pipeline.py` — `condensed_hks_pipeline` (line 386)
- `src/meshmash/decompose.py` — **no changes needed**; string dtype already handled (line 316)

## Verification

1. `pip install -e .` → `meshmash --help` shows `hks` subcommand
2. `meshmash hks --help` shows all options
3. `meshmash hks input.obj --output out.npz` runs end-to-end
4. `meshmash hks input.obj -o out.npz --config params.toml` with `n_components = 16` in TOML — confirm used
5. `meshmash hks input.obj -o out.npz --n-components 8` — CLI overrides TOML
