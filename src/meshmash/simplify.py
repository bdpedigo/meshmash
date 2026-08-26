from typing import Optional

import numpy as np
from fast_simplification import replay_simplification, simplify

from .types import Mesh, interpret_mesh


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
