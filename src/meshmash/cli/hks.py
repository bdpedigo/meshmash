from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Annotated, Optional

import typer

from meshmash.io import save_condensed_features
from meshmash.pipeline import condensed_hks_pipeline


def _load_config(ctx: typer.Context, param: typer.CallbackParam, value: Optional[Path]):
    if value is not None:
        with open(value, "rb") as f:
            toml_data = tomllib.load(f)
        ctx.default_map = ctx.default_map or {}
        ctx.default_map.update(toml_data)
    return value


def hks(
    mesh_path: Annotated[
        Path,
        typer.Argument(
            help="Path to the input mesh file (any format supported by meshio)."
        ),
    ],
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Path to write the output .npz file.")
    ],
    config: Annotated[
        Optional[Path],
        typer.Option(
            "--config",
            "-c",
            help="Path to a TOML config file. Parameter values from the file act as defaults and can be overridden by CLI flags.",
            is_eager=True,
            callback=_load_config,
        ),
    ] = None,
    simplify_agg: Annotated[
        int,
        typer.Option(
            help="Aggressiveness of mesh decimation (0=best quality, 10=fastest)."
        ),
    ] = 7,
    simplify_target_reduction: Annotated[
        Optional[float],
        typer.Option(
            help="Fraction of triangles to remove during simplification. Set to None to skip simplification."
        ),
    ] = 0.7,
    overlap_distance: Annotated[
        float, typer.Option(help="Geodesic distance to overlap mesh chunks.")
    ] = 20_000,
    max_vertex_threshold: Annotated[
        int,
        typer.Option(
            help="Maximum number of vertices per mesh chunk before overlapping."
        ),
    ] = 20_000,
    min_vertex_threshold: Annotated[
        int, typer.Option(help="Minimum number of vertices for a chunk to be included.")
    ] = 200,
    max_overlap_neighbors: Annotated[
        int,
        typer.Option(
            help="Maximum neighbors to consider when overlapping chunks; overrules overlap_distance."
        ),
    ] = 40_000,
    n_components: Annotated[
        int, typer.Option(help="Number of HKS timescales (= number of HKS features).")
    ] = 32,
    t_min: Annotated[
        float, typer.Option(help="Minimum timescale for the HKS computation.")
    ] = 5e4,
    t_max: Annotated[
        float, typer.Option(help="Maximum timescale for the HKS computation.")
    ] = 2e7,
    max_eigenvalue: Annotated[
        float, typer.Option(help="Maximum eigenvalue for the HKS computation.")
    ] = 5e-6,
    robust: Annotated[
        bool, typer.Option("--robust/--no-robust", help="Use the robust Laplacian.")
    ] = True,
    mollify_factor: Annotated[
        float, typer.Option(help="Mollification factor for the robust Laplacian.")
    ] = 1e-5,
    truncate_extra: Annotated[
        bool,
        typer.Option(
            "--truncate-extra/--no-truncate-extra",
            help="Truncate extra eigenpairs computed past max_eigenvalue.",
        ),
    ] = True,
    drop_first: Annotated[
        bool,
        typer.Option(
            "--drop-first/--no-drop-first",
            help="Drop the first eigenpair (proportional to vertex areas).",
        ),
    ] = True,
    decomposition_dtype: Annotated[
        str, typer.Option(help="Dtype for the decomposition: 'float32' or 'float64'.")
    ] = "float32",
    distance_threshold: Annotated[
        float,
        typer.Option(
            help="Distance threshold for agglomerating the mesh into domains."
        ),
    ] = 3.0,
    auxiliary_features: Annotated[
        bool,
        typer.Option(
            "--auxiliary-features/--no-auxiliary-features",
            help="Compute and include auxiliary features (e.g. component size).",
        ),
    ] = True,
    n_jobs: Annotated[
        int, typer.Option(help="Number of parallel jobs. -1 uses all available cores.")
    ] = -1,
    verbose: Annotated[
        bool, typer.Option("--verbose/--no-verbose", help="Print progress messages.")
    ] = False,
):
    import meshio

    try:
        mesh_obj = meshio.read(mesh_path)
    except Exception as e:
        raise typer.BadParameter(f"Could not read mesh file '{mesh_path}': {e}") from e

    if "triangle" not in mesh_obj.cells_dict:
        raise typer.BadParameter(
            f"No triangle cells found in '{mesh_path}'. "
            "meshmash requires a triangular mesh. "
            f"Available cell types: {list(mesh_obj.cells_dict.keys())}"
        )

    vertices = mesh_obj.points
    faces = mesh_obj.cells_dict["triangle"]

    result = condensed_hks_pipeline(
        mesh=(vertices, faces),
        simplify_agg=simplify_agg,
        simplify_target_reduction=simplify_target_reduction,
        overlap_distance=overlap_distance,
        max_vertex_threshold=max_vertex_threshold,
        min_vertex_threshold=min_vertex_threshold,
        max_overlap_neighbors=max_overlap_neighbors,
        n_components=n_components,
        t_min=t_min,
        t_max=t_max,
        max_eigenvalue=max_eigenvalue,
        robust=robust,
        mollify_factor=mollify_factor,
        truncate_extra=truncate_extra,
        drop_first=drop_first,
        decomposition_dtype=decomposition_dtype,
        distance_threshold=distance_threshold,
        auxiliary_features=auxiliary_features,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    save_condensed_features(output, result.condensed_features, result.labels)

    if verbose:
        typer.echo(f"Saved features to {output}")
        for key, val in result.timing_info.items():
            typer.echo(f"  {key}: {val:.2f}s")
