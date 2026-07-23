from typing import Literal, Optional, Tuple, Union

import numpy as np
import point_cloud_utils as pcu
from gpytoolbox import fast_winding_number
from tqdm.auto import tqdm

from .types import Mesh, interpret_mesh
from .utils import mask_mesh_by_faces, mesh_to_poly, poly_to_mesh


def orient_faces_by_adjacency(mesh: Mesh) -> Mesh:
    """Make face windings consistent within each connected component.

    Uses [point_cloud_utils.orient_mesh_faces][], which propagates a consistent
    winding order across faces sharing an edge.  Note that this only makes
    orientation consistent *within* each connected component; the overall sign
    of each component's normals is left arbitrary.  Use
    [orient_faces_by_winding][meshmash.clean.orient_faces_by_winding] afterwards
    to fix each component's sign, or [orient_mesh][meshmash.clean.orient_mesh]
    to do both.

    Parameters
    ----------
    mesh :
        Input mesh accepted by [interpret_mesh][meshmash.types.interpret_mesh].

    Returns
    -------
    :
        The mesh as a ``(vertices, faces)`` tuple with consistently-oriented
        faces.  Vertices are unchanged.
    """
    vertices, faces = interpret_mesh(mesh)
    faces, _ = pcu.orient_mesh_faces(faces)
    return vertices, faces


def orient_faces_by_winding(
    mesh: Mesh,
    flip_eps: float = 25.0,
    max_component_faces: Optional[int] = None,
    verbose: Union[bool, int] = False,
) -> Mesh:
    """Flip each connected component's overall face sign to point outward.

    [orient_faces_by_adjacency][meshmash.clean.orient_faces_by_adjacency] makes
    windings consistent *within* each component but leaves each component's
    overall sign arbitrary.  This function decides each component's sign from its
    OWN self-winding-number and reverses the whole component when its normals
    point inward.  It assumes each component is already internally consistent, so
    it is normally run *after*
    [orient_faces_by_adjacency][meshmash.clean.orient_faces_by_adjacency].

    For each face, two probe points are placed ``flip_eps`` along the ``+`` and
    ``-`` normal directions, and the component's self-winding-number is evaluated
    at both.  The inside side has ``|WN| ~ 1`` and the outside side ``|WN| ~ 0``.
    Comparing *magnitudes* (rather than signed values) makes the test
    orientation-independent, so the operation is idempotent.  A component is
    flipped when its ``+`` normal side reads as inside.

    Parameters
    ----------
    mesh :
        Input mesh accepted by [interpret_mesh][meshmash.types.interpret_mesh].
    flip_eps :
        Distance, in mesh units, to offset probe points along each face normal
        when sampling the winding number.
    max_component_faces :
        If given, components with more than this many faces are skipped (left
        untouched).  Useful to avoid re-testing a large component that is already
        known to be correctly oriented.  ``None`` (default) tests every
        component.
    verbose :
        If truthy, show a progress bar and print a summary of flipped
        components (sorted by decision margin, smallest/most ambiguous first).

    Returns
    -------
    :
        The mesh as a ``(vertices, faces)`` tuple with each component's faces
        reversed as needed to point outward.  Vertices are unchanged.
    """
    vertices, faces = interpret_mesh(mesh)
    _, _, face_components, _ = pcu.connected_components(vertices, faces)

    oriented_faces = faces.copy()
    component_ids = np.unique(face_components)
    flipped_info = []  # (component_id, n_faces, margin) for each flipped component
    for component_id in tqdm(
        component_ids, desc="Orienting components", disable=not verbose
    ):
        component_face_mask = face_components == component_id
        if (
            max_component_faces is not None
            and component_face_mask.sum() > max_component_faces
        ):
            continue
        component_faces = oriented_faces[component_face_mask]

        component_normals = pcu.estimate_mesh_face_normals(vertices, component_faces)
        component_centers = vertices[component_faces].mean(axis=1)
        probe_out = component_centers + component_normals * flip_eps
        probe_in = component_centers - component_normals * flip_eps

        # self-winding-number: only this component's faces contribute, so the
        # result is independent of any (mis)orientation elsewhere in the mesh
        w_out = fast_winding_number(probe_out, vertices, component_faces)
        w_in = fast_winding_number(probe_in, vertices, component_faces)

        # outward normal should point to the smaller-|WN| (outside) side; compare
        # magnitudes so the test does not co-rotate with the normal (idempotent)
        margin = np.mean(np.abs(w_out)) - np.mean(np.abs(w_in))
        if margin > 0:
            oriented_faces[component_face_mask] = component_faces[:, ::-1]
            flipped_info.append(
                (int(component_id), int(component_face_mask.sum()), float(margin))
            )

    if verbose:
        print(
            f"Flipped {len(flipped_info)} / {len(component_ids)} components to "
            "outward orientation"
        )
        # smallest |margin| = near-tie = most ambiguous (e.g. open sheets)
        for component_id, n_faces, margin in sorted(
            flipped_info, key=lambda x: abs(x[2])
        ):
            print(f"  component {component_id}: {n_faces} faces, margin={margin:+.4f}")

    return vertices, oriented_faces


def orient_mesh(
    mesh: Mesh,
    flip_eps: float = 25.0,
    max_component_faces: Optional[int] = None,
    verbose: Union[bool, int] = False,
) -> Mesh:
    """Orient all faces to a consistent, outward-pointing winding.

    Convenience wrapper that runs
    [orient_faces_by_adjacency][meshmash.clean.orient_faces_by_adjacency] to make
    windings consistent within each connected component, then
    [orient_faces_by_winding][meshmash.clean.orient_faces_by_winding] to flip
    each component's overall sign so its normals point outward.

    Parameters
    ----------
    mesh :
        Input mesh accepted by [interpret_mesh][meshmash.types.interpret_mesh].
    flip_eps :
        Probe offset passed to
        [orient_faces_by_winding][meshmash.clean.orient_faces_by_winding].
    max_component_faces :
        Component-size skip threshold passed to
        [orient_faces_by_winding][meshmash.clean.orient_faces_by_winding].
    verbose :
        Verbosity passed to
        [orient_faces_by_winding][meshmash.clean.orient_faces_by_winding].

    Returns
    -------
    :
        The mesh as a ``(vertices, faces)`` tuple with consistent, outward-
        pointing faces.  Vertices are unchanged.
    """
    mesh = orient_faces_by_adjacency(mesh)
    return orient_faces_by_winding(
        mesh,
        flip_eps=flip_eps,
        max_component_faces=max_component_faces,
        verbose=verbose,
    )


def remove_degenerate_faces(mesh: Mesh, min_area: float = 1e-2) -> Mesh:
    """Drop faces whose area is at or below a threshold.

    Sliver/degenerate triangles (near-zero area) cause problems for many
    geometry operations (normals, winding numbers, Laplacians), so they are
    removed here based on their computed area.  Vertices are left untouched, so
    some may become unreferenced.

    Parameters
    ----------
    mesh :
        Input mesh accepted by [interpret_mesh][meshmash.types.interpret_mesh].
    min_area :
        Faces with area less than or equal to this value are removed.

    Returns
    -------
    :
        The mesh as a ``(vertices, faces)`` tuple with degenerate faces removed.
    """
    vertices, faces = interpret_mesh(mesh)
    areas = pcu.mesh_face_areas(vertices, faces)
    mask = areas > min_area
    faces = faces[mask]
    return vertices, faces


def remove_fins(mesh: Mesh) -> Mesh:
    """Remove "fin" triangles that hang on by a single, singly-referenced vertex.

    A fin is a triangle attached to the rest of the mesh only through one of its
    vertices, where that vertex is referenced by no other face.  Such triangles
    dangle off the surface and are dropped here.  The now-unreferenced vertices
    are then removed and the faces reindexed so the returned mesh is compact.

    Parameters
    ----------
    mesh :
        Input mesh accepted by [interpret_mesh][meshmash.types.interpret_mesh].

    Returns
    -------
    :
        The mesh as a ``(vertices, faces)`` tuple with fin triangles and any
        resulting unreferenced vertices removed and faces reindexed.
    """
    vertices, faces = interpret_mesh(mesh)
    vertex_counts = np.bincount(faces.ravel(), minlength=len(vertices))
    face_mask = ~np.any(vertex_counts[faces] == 1, axis=1)
    return mask_mesh_by_faces((vertices, faces), face_mask)


def clean_mesh(
    mesh: Mesh,
    tolerance: float = 50.0,
    fill_holes_size: float = 10000.0,
    lines_to_points: bool = True,
    point_merging: bool = True,
) -> Mesh:
    """Merge nearby vertices, remove degenerate cells, and fill small holes.

    Wraps [pyvista.PolyData.clean][] (absolute tolerance) followed by
    [pyvista.PolyDataFilters.fill_holes][].  Merging welds coincident/near-
    coincident vertices (closing cracks from unmerged duplicates), while
    hole-filling triangulates open boundary loops up to a given size.

    Parameters
    ----------
    mesh :
        Input mesh accepted by [mesh_to_poly][meshmash.utils.mesh_to_poly].
    tolerance :
        Absolute distance below which vertices are merged, in mesh units.
    fill_holes_size :
        Maximum hole size to fill, as passed to
        [pyvista.PolyDataFilters.fill_holes][].
    lines_to_points :
        Whether degenerate lines are converted to points during cleaning.
    point_merging :
        Whether to merge coincident points during cleaning.

    Returns
    -------
    :
        The cleaned mesh as a ``(vertices, faces)`` tuple.
    """
    poly = (
        mesh_to_poly(mesh)
        .clean(
            absolute=True,
            tolerance=tolerance,
            lines_to_points=lines_to_points,
            point_merging=point_merging,
        )
        .fill_holes(fill_holes_size)
    )
    return poly_to_mesh(poly)


def compute_face_winding_numbers(
    mesh: Mesh,
    sampling: Literal["constant", "edge_adaptive"] = "edge_adaptive",
    epsilon: float = 25.0,
    edge_scale: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample the winding number just outside and just inside each face.

    For every face, two probe points are placed a small distance off the face
    center along the ``+`` and ``-`` normal directions, and the mesh's fast
    winding number is evaluated at each.  For a correctly outward-oriented,
    watertight surface the ``+`` normal ("outside") probe reads ``~0`` and the
    ``-`` normal ("inside") probe reads ``~1``.  Faces where *both* sides read as
    inside are buried inside the solid; faces where *either* side reads as
    outside lie on a surface.

    Assumes faces are oriented outward (e.g. via
    [orient_mesh][meshmash.clean.orient_mesh]).

    Parameters
    ----------
    mesh :
        Input mesh accepted by [interpret_mesh][meshmash.types.interpret_mesh].
    sampling :
        How to choose the probe offset distance.  ``"constant"`` uses a fixed
        ``epsilon`` for every face; ``"edge_adaptive"`` scales the offset per
        face by ``edge_scale`` times the face's mean edge length (more robust
        across meshes with varying resolution).
    epsilon :
        Probe offset distance, in mesh units, used when ``sampling="constant"``.
    edge_scale :
        Fraction of each face's mean edge length used as the probe offset when
        ``sampling="edge_adaptive"``.

    Returns
    -------
    winding_outside :
        Winding number at the ``+`` normal probe of each face, shape ``(F,)``.
    winding_inside :
        Winding number at the ``-`` normal probe of each face, shape ``(F,)``.
    """
    vertices, faces = interpret_mesh(mesh)

    normals = pcu.estimate_mesh_face_normals(vertices, faces)
    centers = vertices[faces].mean(axis=1)

    if sampling == "constant":
        offset = epsilon
    elif sampling == "edge_adaptive":
        tri = vertices[faces]
        edge_lengths = np.linalg.norm(tri - np.roll(tri, -1, axis=1), axis=2)
        offset = edge_scale * edge_lengths.mean(axis=1)[:, None]
    else:
        raise ValueError(
            f"Unknown sampling strategy {sampling!r}; expected 'constant' or "
            "'edge_adaptive'."
        )

    points_outside = centers + normals * offset
    points_inside = centers - normals * offset

    query_points = np.concatenate([points_outside, points_inside], axis=0)
    winding = fast_winding_number(query_points, vertices, faces)

    winding_outside = winding[: len(points_outside)]
    winding_inside = winding[len(points_outside) :]
    return winding_outside, winding_inside


def graphcut_face_mask(
    mesh: Mesh,
    winding_outside: np.ndarray,
    winding_inside: np.ndarray,
    data_weight: float = 0.1,
    pairwise_weight: float = 1.0,
) -> np.ndarray:
    """Regularize a keep/remove face labeling with a binary graph cut.

    Turns the per-face winding-number evidence from
    [compute_face_winding_numbers][meshmash.clean.compute_face_winding_numbers]
    into a coherent keep/remove mask by solving a min-cut on the face-adjacency
    (dual) graph.  The cut balances a **data** term (agreement with the winding
    evidence) against a **smoothness** term (adjacent faces should share a
    label, weighted by shared-edge length), suppressing the salt-and-pepper
    misclassifications a hard threshold produces near ``winding ~ 0.5``.

    With ``pairwise_weight=0`` this reproduces the hard threshold
    ``min(winding_outside, winding_inside) <= 0.5``.

    !!! note
        Requires [PyMaxflow](https://pmneila.github.io/PyMaxflow/) (imported as
        ``maxflow``), which is GPL-licensed.  Install with
        ``pip install meshmash[graphcut]``.

    Parameters
    ----------
    mesh :
        Input mesh accepted by [interpret_mesh][meshmash.types.interpret_mesh].
    winding_outside :
        Per-face ``+`` normal winding numbers, shape ``(F,)``.
    winding_inside :
        Per-face ``-`` normal winding numbers, shape ``(F,)``.
    data_weight :
        Strength of the winding-number evidence (unary term).
    pairwise_weight :
        Strength of the coherence prior (smoothness term).

    Returns
    -------
    keep_mask :
        Boolean array of length ``F``; ``True`` marks faces to keep (surface),
        ``False`` marks buried faces to remove.
    """
    try:
        import maxflow
    except ImportError as e:
        raise ImportError(
            "graphcut_face_mask requires PyMaxflow (imported as `maxflow`). "
            "Install with `pip install meshmash[graphcut]` or "
            "`pip install PyMaxflow`. Note: PyMaxflow is GPL-licensed."
        ) from e

    vertices, faces = interpret_mesh(mesh)
    n_faces = len(faces)

    # --- data term (unary) --------------------------------------------------
    # m = min(w_out, w_in): a face is "buried" (remove) only when BOTH probes
    # read inside (m > 0.5), and kept when either probe is outside (m < 0.5).
    m = np.minimum(winding_outside, winding_inside)
    d_keep = data_weight * np.maximum(0.0, m - 0.5)  # cost of KEEPING a buried face
    d_remove = data_weight * np.maximum(0.0, 0.5 - m)  # cost of REMOVING a surface face

    # --- pairwise term (smoothness) -----------------------------------------
    # Face adjacency from shared (manifold) edges, each dual edge weighted by its
    # shared-edge length so cuts prefer to run along short edges / creases.
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    edges_sorted = np.sort(edges, axis=1)
    face_of_edge = np.tile(np.arange(n_faces), 3)

    order = np.lexsort((edges_sorted[:, 1], edges_sorted[:, 0]))
    edges_sorted = edges_sorted[order]
    face_of_edge = face_of_edge[order]

    # consecutive identical (sorted) edges are shared by two faces -> a dual edge
    same = np.all(edges_sorted[1:] == edges_sorted[:-1], axis=1)
    adj_a = face_of_edge[:-1][same]
    adj_b = face_of_edge[1:][same]
    shared_edge = edges_sorted[:-1][same]
    edge_len = np.linalg.norm(
        vertices[shared_edge[:, 0]] - vertices[shared_edge[:, 1]], axis=1
    )
    valid = adj_a != adj_b
    adj_a, adj_b, edge_len = adj_a[valid], adj_b[valid], edge_len[valid]
    w_pair = pairwise_weight * (edge_len / edge_len.mean())

    # --- build and solve the min-cut ----------------------------------------
    # source = KEEP (label 0), sink = REMOVE (label 1); get_segment == 0 -> keep.
    g = maxflow.Graph[float](n_faces, len(adj_a))
    node_ids = g.add_nodes(n_faces)
    g.add_grid_tedges(node_ids, d_remove, d_keep)
    for i, j, w in zip(adj_a, adj_b, w_pair):
        g.add_edge(int(i), int(j), float(w), float(w))
    g.maxflow()

    seg = np.fromiter(
        (g.get_segment(i) for i in range(n_faces)), dtype=np.int8, count=n_faces
    )
    keep_mask = seg == 0  # source side = keep
    return keep_mask


def remove_interior_faces(
    mesh: Mesh,
    sampling: Literal["constant", "edge_adaptive"] = "edge_adaptive",
    epsilon: float = 25.0,
    edge_scale: float = 0.5,
    graph_cut: bool = True,
    data_weight: float = 0.1,
    pairwise_weight: float = 1.0,
    return_mask: bool = False,
) -> Union[Mesh, np.ndarray]:
    """Remove buried/interior faces using winding numbers.

    Convenience wrapper that runs
    [compute_face_winding_numbers][meshmash.clean.compute_face_winding_numbers]
    to classify each face as surface or buried, then either applies a hard
    threshold or (default) regularizes the labeling with
    [graphcut_face_mask][meshmash.clean.graphcut_face_mask] before dropping the
    buried faces via [mask_mesh_by_faces][meshmash.utils.mask_mesh_by_faces].

    Assumes faces are oriented outward (e.g. via
    [orient_mesh][meshmash.clean.orient_mesh]).

    Parameters
    ----------
    mesh :
        Input mesh accepted by [interpret_mesh][meshmash.types.interpret_mesh].
    sampling :
        Probe-offset strategy passed to
        [compute_face_winding_numbers][meshmash.clean.compute_face_winding_numbers].
    epsilon :
        Constant probe offset passed to
        [compute_face_winding_numbers][meshmash.clean.compute_face_winding_numbers].
    edge_scale :
        Edge-adaptive probe scale passed to
        [compute_face_winding_numbers][meshmash.clean.compute_face_winding_numbers].
    graph_cut :
        If ``True`` (default), regularize the labeling with
        [graphcut_face_mask][meshmash.clean.graphcut_face_mask] (requires
        PyMaxflow).  If ``False``, use the hard threshold
        ``min(winding_outside, winding_inside) <= 0.5``.
    data_weight :
        Graph-cut data-term weight (only used when ``graph_cut=True``).
    pairwise_weight :
        Graph-cut smoothness-term weight (only used when ``graph_cut=True``).
    return_mask :
        If ``True``, return the boolean per-face keep mask instead of the
        cleaned mesh.

    Returns
    -------
    :
        The cleaned ``(vertices, faces)`` tuple with buried faces removed and
        vertices reindexed, or the boolean keep mask if ``return_mask=True``.
    """
    winding_outside, winding_inside = compute_face_winding_numbers(
        mesh, sampling=sampling, epsilon=epsilon, edge_scale=edge_scale
    )
    if graph_cut:
        keep_mask = graphcut_face_mask(
            mesh,
            winding_outside,
            winding_inside,
            data_weight=data_weight,
            pairwise_weight=pairwise_weight,
        )
    else:
        keep_mask = (winding_outside <= 0.5) | (winding_inside <= 0.5)

    if return_mask:
        return keep_mask
    return mask_mesh_by_faces(mesh, keep_mask)
