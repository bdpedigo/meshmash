# meshmash CLI

## Commands

### `meshmash datasets`

Download and cache sample mesh datasets for use with meshmash.

> **Note:** This command requires the optional `datasets` extra.
> Install it with `pip install meshmash[datasets]` (adds `pooch`).

```
meshmash datasets [NAME]
```

**Arguments**

| Name | Description |
|------|-------------|
| `NAME` | Dataset name to fetch (without file extension), or `all` to fetch every available dataset. Defaults to `all`. Available datasets: `microns_dendrite_sample`, `microns_neuron_sample`. |

Files are downloaded once and stored in the OS-appropriate user cache directory
(e.g. `~/.cache/meshmash` on Linux/macOS). Subsequent calls reuse the cached copy.

**Examples**

Fetch all available sample datasets:

```bash
meshmash datasets
# or equivalently:
meshmash datasets all
```

Fetch a single dataset:

```bash
meshmash datasets microns_dendrite_sample
```

---

### `meshmash hks`

Compute condensed Heat Kernel Signature (HKS) features on a triangular mesh and
write the result to a compressed `.npz` file.

```
meshmash hks MESH_PATH --output OUTPUT [OPTIONS]
```

**Arguments**

| Name | Description |
|------|-------------|
| `MESH_PATH` | Path to the input mesh file. Any format supported by [meshio](https://github.com/nschloe/meshio) is accepted (`.ply`, `.obj`, `.stl`, `.vtu`, …). The mesh must contain triangle cells. |

**Required options**

| Flag | Description |
|------|-------------|
| `--output` / `-o` | Path to write the output `.npz` file. |

**Config file**

| Flag | Description |
|------|-------------|
| `--config` / `-c` | Path to a TOML file whose keys correspond to the option names below. CLI flags override config-file values. |

Example config file (`params.toml`):

```toml
n_components = 16
max_eigenvalue = 1e-7
distance_threshold = 2.5
n_jobs = 4
```

**Simplification options**

| Flag | Default | Description |
|------|---------|-------------|
| `--simplify-agg` | `7` | Aggressiveness of mesh decimation (0 = best quality, 10 = fastest). |
| `--simplify-target-reduction` | `0.7` | Fraction of triangles to remove. Set to `None` (pass `--simplify-target-reduction none`) to skip simplification. |

**Chunking options**

| Flag | Default | Description |
|------|---------|-------------|
| `--overlap-distance` | `20000` | Geodesic distance (in mesh units) by which chunks overlap. |
| `--max-vertex-threshold` | `20000` | Maximum vertices per chunk before overlapping. |
| `--min-vertex-threshold` | `200` | Minimum vertices for a chunk to be included. |
| `--max-overlap-neighbors` | `40000` | Maximum neighbours when building overlaps; overrules `--overlap-distance`. |

**HKS options**

| Flag | Default | Description |
|------|---------|-------------|
| `--n-components` | `32` | Number of HKS timescales (= number of output feature columns). |
| `--t-min` | `5e4` | Minimum HKS timescale. |
| `--t-max` | `2e7` | Maximum HKS timescale. |
| `--max-eigenvalue` | `5e-6` | Maximum Laplacian eigenvalue to compute. Larger values give more spectral detail at the cost of time. |
| `--robust` / `--no-robust` | `--robust` | Use the robust Laplacian (recommended for noisy meshes). |
| `--mollify-factor` | `1e-5` | Mollification factor for the robust Laplacian. |
| `--truncate-extra` / `--no-truncate-extra` | `--truncate-extra` | Discard eigenpairs computed beyond `--max-eigenvalue`. |
| `--drop-first` / `--no-drop-first` | `--drop-first` | Drop the first eigenpair (always proportional to vertex areas). |
| `--decomposition-dtype` | `float32` | Floating-point precision for the eigendecomposition (`float32` or `float64`). |

**Agglomeration options**

| Flag | Default | Description |
|------|---------|-------------|
| `--distance-threshold` | `3.0` | Ward linkage distance threshold for agglomerating vertices into domains. |

**Output options**

| Flag | Default | Description |
|------|---------|-------------|
| `--auxiliary-features` / `--no-auxiliary-features` | `--auxiliary-features` | Include auxiliary per-domain features (e.g. component size, centroid coordinates). |

**Misc**

| Flag | Default | Description |
|------|---------|-------------|
| `--n-jobs` | `-1` | Parallel workers (`-1` = all available cores). |
| `--verbose` / `--no-verbose` | `--no-verbose` | Print progress and timing information. |

## Output format

The output `.npz` file contains two arrays:

| Key | Shape | Description |
|-----|-------|-------------|
| `X` | `(n_domains + 1, n_features)` | Per-domain feature matrix. Row index 0 corresponds to the null domain (unlabelled vertices). |
| `labels` | `(n_vertices,)` | Per-vertex domain label. `-1` marks vertices excluded from the computation. |

A `header.txt` file is written alongside the `.npz` containing the tab-separated column names, so that multiple runs to the same output directory stay consistent.

Read the result back in Python:

```python
from meshmash import read_condensed_features

features, labels = read_condensed_features("output/features.npz")
# features: pd.DataFrame, shape (n_domains + 1, n_features)
# labels:   np.ndarray,  shape (n_vertices,)
```

## Examples

**Minimal invocation**

```bash
meshmash hks neuron.ply --output results/features.npz
```

**Fast run with a config file**

```bash
meshmash hks neuron.ply --output results/features.npz --config params.toml
```

**Override individual parameters**

```bash
meshmash hks neuron.ply \
  --output results/features.npz \
  --n-components 16 \
  --max-eigenvalue 1e-7 \
  --distance-threshold 2.5 \
  --n-jobs 4 \
  --verbose
```

**Skip mesh simplification**

```bash
meshmash hks neuron.ply --output results/features.npz --simplify-target-reduction none
```

**High-quality, single-threaded run**

```bash
meshmash hks neuron.ply \
  --output results/features.npz \
  --simplify-agg 0 \
  --n-components 64 \
  --max-eigenvalue 1e-5 \
  --decomposition-dtype float64 \
  --n-jobs 1
```
