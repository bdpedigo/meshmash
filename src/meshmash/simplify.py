from typing import Optional

import numpy as np
from fast_simplification import _replay, simplify
from fast_simplification.replay import _map_isolated_points

from .types import Mesh, interpret_mesh
from .utils import surface_area, vertex_density


def _decimate(
    vertices: np.ndarray, faces: np.ndarray, agg: float, target_reduction: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decimate a mesh, returning its vertices, faces, and each input vertex's new index.

    The same three arrays as ``fast_simplification.replay_simplification`` on
    the collapses of one ``simplify`` call, including ``-1`` for an input
    vertex that no face references.
    """
    points, _, collapses = simplify(
        vertices,
        faces,
        agg=agg,
        target_reduction=target_reduction,
        return_collapses=True,
    )
    # NOTE: this is replay_simplification with the replay itself left out.
    # The replay rebuilds the decimated mesh in the Replay module's global C++
    # vectors, which fast-simplification clears but never frees, and all it
    # adds is the vertex positions, which `simplify` already returned in the
    # same order. The rest is its own array bookkeeping, using its private
    # helpers, so the fast-simplification pin and tests/test_simplify.py hold
    # it to replay_simplification's output.
    n_vertices = len(vertices)
    faces = np.ascontiguousarray(faces)
    referenced = np.zeros(n_vertices, dtype=bool)
    if faces.size:
        referenced[np.unique(faces)] = True
    kept_vertices = None
    if not referenced.all():
        kept_vertices = np.flatnonzero(referenced)
        old_to_new = np.full(n_vertices, -1, dtype=np.int64)
        old_to_new[kept_vertices] = np.arange(len(kept_vertices))
        faces = np.ascontiguousarray(old_to_new[faces].astype(faces.dtype, copy=False))
        collapses = np.ascontiguousarray(
            old_to_new[collapses].astype(np.int32, copy=False)
        )
        n_vertices = len(kept_vertices)

    index_mapping = _replay.compute_indice_mapping(collapses, n_vertices)
    edges, new_faces = _replay.clean_triangles_and_edges(index_mapping[faces])
    n_decimated = int(index_mapping.max()) + 1 if len(index_mapping) else 0
    # Only the point count is read, never the positions.
    merge, merged, outliers = _map_isolated_points(
        np.empty((n_decimated, 0)), edges, new_faces, return_outliers=True
    )
    new_faces = merge[new_faces]
    index_mapping = merge[index_mapping]
    # Merged and outlying points leave the vertex array, and every index above
    # one moves down by one.
    dropped = np.union1d(merged, outliers)
    positions = np.arange(n_decimated)
    shift = positions - np.searchsorted(dropped, positions, side="right")
    new_faces = shift[new_faces]
    index_mapping = shift[index_mapping]

    if kept_vertices is not None:
        full_mapping = np.full(len(vertices), -1, dtype=index_mapping.dtype)
        full_mapping[kept_vertices] = index_mapping
        index_mapping = full_mapping
    return points.astype(np.float32), new_faces, index_mapping


def simplify_mesh(
    mesh: Mesh,
    agg: int = 7,
    target_reduction: Optional[float] = 0.7,
) -> tuple[Mesh, np.ndarray]:
    """Decimate a mesh, and recover where each original vertex went.

    The mesh and the mapping come back together and always agree, so a caller
    never has to pair a mesh from one call with indices from another.

    Parameters
    ----------
    mesh :
        Input mesh accepted by [interpret_mesh][meshmash.types.interpret_mesh].
    agg :
        Decimation aggressiveness (0-10).  Higher values are faster but
        reduce mesh quality.  Low values may prevent reaching
        ``target_reduction``.
    target_reduction :
        Fraction of triangles to remove.  ``None`` skips simplification: the
        mesh comes back untouched with an identity mapping.

    Returns
    -------
    mesh :
        The decimated ``(vertices, faces)`` tuple.
    mapping :
        Array of length ``V``, where ``mapping[i]`` is the index in the
        decimated mesh of vertex ``i`` of the input.  Every vertex that a
        face references has one, because a collapse merges vertices rather
        than discarding them.  A vertex no face references maps to ``-1``.
    """
    vertices, faces = interpret_mesh(mesh)

    if target_reduction is None:
        return (vertices, faces), np.arange(len(vertices))

    new_vertices, new_faces, mapping = _decimate(
        vertices, faces, agg=agg, target_reduction=target_reduction
    )
    return (new_vertices, new_faces), mapping


def simplify_to_density(
    mesh: Mesh,
    target_density: float,
    simplify_agg: int = 7,
    tolerance: float = 0.05,
    max_iter: int = 5,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Iteratively simplify a mesh toward a target vertex density.

    Repeatedly decimates the mesh with ``fast-simplification``, estimating
    the reduction needed each pass from the assumption that vertex density
    scales roughly linearly with face count.  Stops once the density is
    within ``tolerance`` of ``target_density`` or ``max_iter`` is reached.

    Parameters
    ----------
    mesh :
        Input mesh accepted by [interpret_mesh][meshmash.types.interpret_mesh].
    target_density :
        Desired [vertex_density][meshmash.utils.vertex_density] (vertices per
        unit surface area).  If the mesh is already at or below this density,
        it is returned unchanged.
    simplify_agg :
        Decimation aggressiveness (0-10) passed to ``fast-simplification``.
    tolerance :
        Acceptable relative overshoot of the density above the target before
        stopping (default 5%).
    max_iter :
        Maximum number of simplification passes.
    verbose :
        If ``True``, print per-iteration density and reduction estimates.

    Returns
    -------
    vertices :
        Simplified vertex positions.
    faces :
        Simplified triangle face indices.
    mapping :
        Array of length ``V_input`` mapping each input vertex to its index in
        the simplified mesh, composed across all iterations, as
        [simplify_mesh][meshmash.simplify.simplify_mesh] gives it for one pass.
    """
    vertices, faces = interpret_mesh(mesh)
    mapping = np.arange(len(vertices))
    if len(faces) == 0 or surface_area((vertices, faces)) == 0:
        # Degenerate mesh (empty or zero total area): nothing to simplify.
        if verbose:
            print("[simplify_to_density] degenerate mesh, returning unchanged")
        return vertices, faces, mapping
    for i in range(max_iter):
        current_density = vertex_density((vertices, faces))
        if current_density <= target_density * (1 + tolerance):
            break
        target_reduction = float(
            np.clip(1 - target_density / current_density, 0.05, 0.99)
        )
        if verbose:
            print(
                f"[simplify_to_density] iter {i}: density {current_density:.3e} "
                f"-> target {target_density:.3e}, reduction {target_reduction:.2%}"
            )
        vertices, faces, step_mapping = _decimate(
            vertices, faces, agg=simplify_agg, target_reduction=target_reduction
        )
        mapping = step_mapping[mapping]
    if verbose:
        print(
            f"[simplify_to_density] final density "
            f"{vertex_density((vertices, faces)):.3e} (target {target_density:.3e})"
        )
    return vertices, faces, mapping
