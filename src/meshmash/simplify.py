from typing import Optional

import numpy as np
from fast_simplification import simplify

from .types import Mesh, interpret_mesh
from .utils import remove_repeated_vertex_faces, surface_area, vertex_density


def _collapse_roots(n_vertices: int, collapses: np.ndarray) -> np.ndarray:
    """The vertex each input vertex ends up in, following its chain of collapses.

    ``collapses[i] = [i0, i1]`` merges ``i1`` into ``i0``, and a vertex is
    merged away at most once, so the collapses form a forest.
    """
    dtype = np.int32 if n_vertices < np.iinfo(np.int32).max else np.int64
    parent = np.arange(n_vertices, dtype=dtype)
    if len(collapses):
        parent[collapses[:, 1]] = collapses[:, 0]
    # Pointer jumping: each pass halves every remaining chain.
    while True:
        grandparent = parent[parent]
        if np.array_equal(grandparent, parent):
            return parent
        parent = grandparent


# NOTE: copied from fast-simplification 0.2.0, fast_simplification/replay.py,
# with the points argument replaced by the point count, its only use there.
# The tests compare _decimate with replay_simplification, so a change upstream
# shows up as a failure there.
#
# Copyright (c) 2017-2021 The PyVista Developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
def _map_isolated_points(
    n_points: int, edges: np.ndarray, triangles: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge points that only edges use into the triangle point an edge reaches.

    Returns the mapping over all ``n_points`` points, the merged points, and
    the points no chain of edges connects to a triangle.
    """
    # The points to connect are the points that are not in the triangles
    # but are in the edges
    points_to_connect = np.intersect1d(
        np.setdiff1d(np.arange(n_points), np.unique(triangles)), np.unique(edges)
    )
    # Start with the identity mapping
    mapping = np.arange(n_points, dtype=np.int64)
    # Remove edges that do not contains points to connect
    edges = edges[np.isin(edges, points_to_connect).any(axis=1)]
    n_edges = edges.shape[0]
    n_edges_old = 0
    # Iterate until there is no more edges to collapse
    # or until a statiionary state is reached
    while n_edges > 0 and n_edges != n_edges_old:
        n_edges_old = n_edges
        # Edges that connect two points to connect
        # are kept for the next iteration
        keep = np.isin(edges, points_to_connect).all(axis=1)
        # Edges that connect a point to connect to a point
        # that is not to connect are merged
        connexions = edges[~keep]
        a = np.isin(connexions, points_to_connect)
        merged = connexions[np.where(a)]
        target = connexions[np.where(~a)]
        # Update the mapping array and the points to connect
        mapping[merged] = mapping[target]
        points_to_connect = np.setdiff1d(points_to_connect, merged)
        # Remove the edges that are merged
        edges = edges[keep]
        # Remove edges that do not contains points to connect
        edges = edges[np.isin(edges, points_to_connect).any(axis=1)]
        n_edges = edges.shape[0]
    # The points that have been merged are the ones
    # such that mapping[i] != i
    merged_points = np.where(mapping != np.arange(len(mapping)))[0]
    isolated_points = points_to_connect
    return mapping, merged_points, isolated_points


def _decimate(
    vertices: np.ndarray, faces: np.ndarray, agg: float, target_reduction: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decimate a mesh, returning its vertices, faces, and each input vertex's new index.

    The vertices and faces are ``fast_simplification.simplify``'s own output,
    in the dtypes of the input vertices and faces.  The mapping follows ``fast_simplification.replay_simplification``: a
    vertex whose faces all collapsed away maps to the vertex its remaining
    edges reach, and ``-1`` means that no face, and no edge to a face, uses
    the vertex. Faces that name a vertex twice are dropped before decimating.

    Raises
    ------
    RuntimeError
        If the mapping does not reproduce ``simplify``'s faces.
    """
    vertices, faces = remove_repeated_vertex_faces((vertices, faces))
    faces = np.ascontiguousarray(faces)
    points, new_faces, collapses = simplify(
        vertices,
        faces,
        agg=agg,
        target_reduction=target_reduction,
        return_collapses=True,
    )
    # NOTE: `simplify` returns no mapping, and its point order is not the one
    # replay_simplification builds. Simplify.h's compact_mesh fixes the order:
    # the output points are the surviving vertices that a face still uses, in
    # input order, and the output faces are the input faces that no collapse
    # deleted, in input order. A collapse deletes exactly the faces that hold
    # both of its vertices, so with no face naming a vertex twice on the way
    # in, a face survives when its corners' roots are still distinct. That
    # rebuilds both arrays from the collapses alone, without the replay, whose
    # C++ copy of the input mesh costs gigabytes on a neuron. The check below
    # holds the rebuild to `simplify`'s faces on every call, so a change in
    # fast-simplification's internals raises here instead of shuffling faces.
    n_vertices = len(vertices)
    roots = _collapse_roots(n_vertices, collapses)
    rooted = roots[faces]
    ab = rooted[:, 0] != rooted[:, 1]
    ac = rooted[:, 0] != rooted[:, 2]
    bc = rooted[:, 1] != rooted[:, 2]
    kept = ab & ac & bc
    # Faces that collapsed to two distinct roots, kept apart for the edges below.
    line = ~kept & (ab | ac | bc)
    line_faces, line_ab, line_ac = rooted[line], ab[line], ac[line]
    # The full per-face arrays are the size of the input mesh; drop them early.
    rooted = rooted[kept]
    del ab, ac, bc, kept, line

    on_face = np.zeros(n_vertices, dtype=bool)
    on_face[rooted] = True
    face_roots = np.flatnonzero(on_face)
    index = np.full(n_vertices, -1, dtype=np.int64)
    index[face_roots] = np.arange(len(face_roots))
    rebuilt_faces = index[rooted]
    del rooted
    if len(face_roots) != len(points) or not np.array_equal(rebuilt_faces, new_faces):
        raise RuntimeError(
            "the vertex mapping rebuilt from fast_simplification.simplify's "
            f"collapses does not reproduce its mesh ({len(face_roots)} vertices "
            f"and {len(rebuilt_faces)} faces rebuilt, {len(points)} and "
            f"{len(new_faces)} returned); its output order has changed"
        )

    # Each line face leaves one edge, picked and ordered as
    # replay_simplification's clean_triangles_and_edges does.
    a, b, c = line_faces[:, 0], line_faces[:, 1], line_faces[:, 2]
    edges = np.stack(
        [
            np.where(line_ab, a, np.where(line_ac, a, c)),
            np.where(line_ab, b, np.where(line_ac, c, b)),
        ],
        axis=1,
    )
    # Roots that only edges use go after the face roots, so the merge never
    # renumbers a face root.
    on_edge = np.zeros(n_vertices, dtype=bool)
    on_edge[edges] = True
    edge_roots = np.flatnonzero(on_edge & ~on_face)
    n_kept = len(face_roots)
    index[edge_roots] = n_kept + np.arange(len(edge_roots))
    merge, _, isolated = _map_isolated_points(
        n_kept + len(edge_roots), index[edges], rebuilt_faces
    )
    # NOTE: replay_simplification deletes these points but leaves their input
    # vertices on the point numbered just before them, an accident of its
    # index shift. Nothing of their piece of the mesh survives, so -1.
    merge[isolated] = -1

    mapping = index[roots]
    reached = mapping >= 0
    mapping[reached] = merge[mapping[reached]]
    # `simplify` works in float64 points and int32 faces whatever it is given.
    return (
        points.astype(vertices.dtype, copy=False),
        new_faces.astype(faces.dtype, copy=False),
        mapping,
    )


def simplify_mesh(
    mesh: Mesh,
    agg: int = 7,
    target_reduction: Optional[float] = 0.7,
) -> tuple[Mesh, np.ndarray]:
    """Decimate a mesh, and recover where each original vertex went.

    The mesh and the mapping come back together and always agree, so a caller
    never has to pair a mesh from one call with indices from another.  Faces
    that name a vertex twice, such as ``(a, a, b)``, are dropped before
    decimating.

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
        decimated mesh of vertex ``i`` of the input.  A collapse merges
        vertices rather than discarding them, so a vertex maps to ``-1`` only
        when nothing of its piece of the mesh survives: no face references
        it, only faces that name a vertex twice reference it, or its whole
        piece collapsed to lines that reach no face.
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
        # A vertex an earlier pass dropped stays dropped: indexing with its -1
        # would hand it the last vertex of this pass.
        mapping = np.where(mapping >= 0, step_mapping[mapping], -1)
    if verbose:
        print(
            f"[simplify_to_density] final density "
            f"{vertex_density((vertices, faces)):.3e} (target {target_density:.3e})"
        )
    return vertices, faces, mapping
