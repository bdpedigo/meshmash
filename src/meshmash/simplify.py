from typing import Optional

import numpy as np
from fast_simplification import replay_simplification, simplify

from .types import Mesh, interpret_mesh
from .utils import surface_area, vertex_density


def simplify_mesh(
    mesh: Mesh,
    agg: int = 7,
    target_reduction: Optional[float] = 0.7,
) -> tuple[Mesh, np.ndarray]:
    """Decimate a mesh, and recover where each original vertex went.

    ``fast_simplification.simplify`` does not return a vertex mapping, and the
    mesh it returns is not ordered the same way as the one
    ``replay_simplification`` rebuilds from the same collapses.  Only the
    replayed ordering agrees with the mapping, so the decimation is run once
    for its collapses and then replayed for the mesh and the mapping
    together.  Returning both from one function is what keeps a caller from
    taking the mesh from one and the indices from the other.

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
        decimated mesh of vertex ``i`` of the input.  Every input vertex has
        one: a collapse merges vertices, it does not discard them, so there
        is no null entry.
    """
    vertices, faces = interpret_mesh(mesh)

    if target_reduction is None:
        return (vertices, faces), np.arange(len(vertices))

    _, _, collapses = simplify(
        vertices,
        faces,
        agg=agg,
        target_reduction=target_reduction,
        return_collapses=True,
    )
    new_vertices, new_faces, mapping = replay_simplification(
        points=vertices,
        triangles=faces,
        collapses=collapses,
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
        the simplified mesh, composed across all iterations.  Mirrors the
        ``replay_simplification`` mapping of the single-pass reduction path.
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
        _, _, collapses = simplify(
            vertices,
            faces,
            agg=simplify_agg,
            target_reduction=target_reduction,
            return_collapses=True,
        )
        vertices, faces, step_mapping = replay_simplification(
            points=vertices,
            triangles=faces,
            collapses=collapses,
        )
        mapping = step_mapping[mapping]
    if verbose:
        print(
            f"[simplify_to_density] final density "
            f"{vertex_density((vertices, faces)):.3e} (target {target_density:.3e})"
        )
    return vertices, faces, mapping
