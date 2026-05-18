import numpy as np

from .utils import mesh_to_poly, poly_to_mesh


def clean_mesh(mesh, tolerance=5):
    poly = mesh_to_poly(mesh)
    poly.clean(
        tolerance=tolerance,  # Merge vertices within this distance
        absolute=True,
        lines_to_points=True,
        point_merging=True,
    )
    return poly_to_mesh(poly)


def node_bbox(node_id, cv, adjust_draco=True):
    """Get the bounding box in nm for a node ID in the chunkedgraph.

    Accounts for dataset voxel offset, mip resolution, chunk size,
    and draco grid quantization.

    Parameters
    ----------
    node_id : int
        A node ID in the chunkedgraph (e.g. a level-2 or level-4 ID).
    cv : cloudvolume.CloudVolume
        CloudVolume object for the segmentation.
    adjust_draco : bool
        Whether to snap bounds to the draco grid.

    Returns
    -------
    bbox : np.ndarray
        (2, 3) array where bbox[0] is the start and bbox[1] is the end, in nm.
    """
    chunk_grid = np.array(cv.meta.decode_chunk_position(node_id))
    layer = cv.meta.decode_layer_id(node_id)

    if getattr(cv.meta, "chunks_start_at_voxel_offset", False):
        base_location = cv.meta.voxel_offset(0) * cv.mip_resolution(0)
    else:
        base_location = np.array([0, 0, 0])

    layer_scale = 2 ** (layer - 2)
    chunk_start = (
        base_location
        + chunk_grid * cv.meta.graph_chunk_size * cv.mip_resolution(0) * layer_scale
    )

    chunk_dims = cv.meta.graph_chunk_size * cv.mip_resolution(0) * layer_scale
    chunk_end = chunk_start + chunk_dims

    if adjust_draco:
        draco_size = cv.meta.get_draco_grid_size(0)
        chunk_start = draco_size * np.ceil(chunk_start / draco_size)
        chunk_end = draco_size * np.floor(chunk_end / draco_size)

    return np.stack([chunk_start, chunk_end])


# ---------------------------------------------------------------------------
# Phase 1: Boundary detection
# ---------------------------------------------------------------------------


def get_chunk_grid_params(cv):
    """Extract chunk grid parameters from CloudVolume metadata.

    Returns
    -------
    chunk_size : np.ndarray
        (3,) L2 chunk size in nm.
    draco_grid_size : float
        Draco quantization grid size in nm.
    offset : np.ndarray
        (3,) offset where chunk grid starts, in nm.
    """
    chunk_size = np.array(cv.meta.graph_chunk_size) * np.array(cv.mip_resolution(0))
    draco_grid_size = cv.meta.get_draco_grid_size(0)
    if getattr(cv.meta, "chunks_start_at_voxel_offset", False):
        offset = np.array(cv.meta.voxel_offset(0)) * np.array(cv.mip_resolution(0))
    else:
        offset = np.array([0, 0, 0], dtype=float)
    return chunk_size, draco_grid_size, offset


def classify_vertices_by_chunk_boundary(vertices, node_id, cv, outer_face_tol=None):
    """Classify vertices as external/internal/no chunk boundary.

    External boundaries are defined as vertices near the raw node bbox faces.
    Internal boundaries are chunk-aligned vertices not on raw bbox faces.

    Parameters
    ----------
    vertices : np.ndarray
        (N, 3) vertex positions in nm.
    node_id : int
        Node id for bbox lookup.
    cv : cloudvolume.CloudVolume
        CloudVolume object for chunk metadata.
    outer_face_tol : float or None
        Absolute tolerance in nm for outer bbox face membership.
        Defaults to draco_grid_size when None.

    Returns
    -------
    dict
        Mapping with boolean arrays:
        - is_any_boundary
        - is_external_boundary
        - is_internal_boundary
        - is_no_boundary
    """
    chunk_size, draco_grid_size, offset = get_chunk_grid_params(cv)

    shifted = vertices - offset
    dist_to_chunk_behind = np.mod(shifted, chunk_size)
    dist_to_chunk_ahead = chunk_size - dist_to_chunk_behind

    # Match CloudVolume draco chunk alignment logic.
    is_on_chunk_behind = np.any(dist_to_chunk_behind < (draco_grid_size / 2), axis=1)
    is_on_chunk_ahead = np.any(dist_to_chunk_ahead <= (draco_grid_size / 2), axis=1)
    is_any_boundary = np.logical_or(is_on_chunk_behind, is_on_chunk_ahead)

    if outer_face_tol is None:
        outer_face_tol = draco_grid_size

    bbox_raw = node_bbox(node_id, cv, adjust_draco=False)
    is_on_outer_face = np.zeros(len(vertices), dtype=bool)
    for axis in range(3):
        is_on_outer_face |= (
            np.abs(vertices[:, axis] - bbox_raw[0, axis]) < outer_face_tol
        )
        is_on_outer_face |= (
            np.abs(vertices[:, axis] - bbox_raw[1, axis]) < outer_face_tol
        )

    is_external_boundary = is_on_outer_face
    is_internal_boundary = is_any_boundary & ~is_on_outer_face
    is_no_boundary = ~(is_external_boundary | is_internal_boundary)

    return {
        "is_any_boundary": is_any_boundary,
        "is_external_boundary": is_external_boundary,
        "is_internal_boundary": is_internal_boundary,
        "is_no_boundary": is_no_boundary,
    }


def deduplicate_chunk_boundaries(mesh, chunk_size, draco_grid_size, offset):
    """Merge duplicate vertices at internal chunk boundaries.

    Replicates CloudVolume's deduplication logic: merges vertex pairs that
    (a) have exactly matching positions and (b) are on a chunk boundary.

    Parameters
    ----------
    mesh : tuple
        (vertices, faces) where vertices is (N, 3) and faces is (M, 3).
    chunk_size : np.ndarray
        (3,) L2 chunk size in nm.
    draco_grid_size : float
        Draco quantization grid size in nm.
    offset : np.ndarray
        (3,) chunk grid offset in nm.

    Returns
    -------
    tuple
        (new_vertices, new_faces) with boundary duplicates merged.
    """
    vertices, faces = mesh

    # Identify chunk-aligned vertices (same logic as cloudvolume)
    shifted = vertices - offset
    dist_behind = np.mod(shifted, chunk_size)
    dist_ahead = chunk_size - dist_behind
    is_on_behind = np.any(dist_behind < (draco_grid_size / 2), axis=1)
    is_on_ahead = np.any(dist_ahead <= (draco_grid_size / 2), axis=1)
    is_chunk_aligned = is_on_behind | is_on_ahead

    # Find vertices with exactly 2 copies at the same position
    _, unique_inverse, counts = np.unique(
        vertices, return_inverse=True, return_counts=True, axis=0
    )
    only_double = np.where(counts == 2)[0]
    is_doubled = np.isin(unique_inverse, only_double)

    # Merge only vertices that are both doubled AND chunk-aligned
    do_merge = is_doubled & is_chunk_aligned

    if not np.any(do_merge):
        return mesh

    # CloudVolume's merge trick: append a unique column, set to -1 for merge targets,
    # then np.unique collapses matching rows.
    n = len(vertices)
    tag_col = np.arange(n, dtype=np.float64).reshape(-1, 1)
    tag_col[do_merge] = -1
    tagged = np.hstack([vertices, tag_col])

    # Remap faces through the tagged vertices
    face_verts = tagged[faces.flatten()]
    new_verts_4d, new_faces_flat = np.unique(face_verts, return_inverse=True, axis=0)
    new_vertices = new_verts_4d[:, :3].astype(vertices.dtype)
    new_faces = new_faces_flat.astype(np.uint32).reshape(-1, 3)

    return (new_vertices, new_faces)


def classify_vertices_to_chunk_planes(vertices, chunk_size, draco_grid_size, offset):
    """Per-axis chunk-boundary detection for all vertices.

    For each vertex and each axis, determines whether the vertex lies within
    half a draco_grid_size of a chunk boundary plane on that axis, and if so,
    which plane value it snaps to.

    Parameters
    ----------
    vertices : np.ndarray
        (N, 3) vertex positions in nm.
    chunk_size : np.ndarray
        (3,) L2 chunk size in nm.
    draco_grid_size : float
        Draco quantization grid size.
    offset : np.ndarray
        (3,) chunk grid offset in nm.

    Returns
    -------
    on_plane : np.ndarray
        (N, 3) bool — True if vertex is on a chunk plane for that axis.
    plane_values : np.ndarray
        (N, 3) float — the chunk plane value for that axis (only meaningful
        where on_plane is True).
    """
    tol = draco_grid_size / 2
    shifted = vertices - offset

    dist_behind = np.mod(shifted, chunk_size)  # (N, 3)
    dist_ahead = chunk_size - dist_behind  # (N, 3)

    # Draco rounds up: behind uses <, ahead uses <=
    on_behind = dist_behind < tol  # (N, 3)
    on_ahead = dist_ahead <= tol  # (N, 3)

    on_plane = on_behind | on_ahead  # (N, 3)

    # Compute actual plane values: snap to nearest chunk boundary
    # For "behind" vertices: plane is at vertex - dist_behind (round down to plane)
    # For "ahead" vertices: plane is at vertex + dist_ahead (round up to plane)
    plane_behind = shifted - dist_behind + offset  # (N, 3)
    plane_ahead = shifted + dist_ahead + offset  # (N, 3)

    # Pick the closer plane for each vertex/axis
    plane_values = np.where(dist_behind < dist_ahead, plane_behind, plane_ahead)

    return on_plane, plane_values


def classify_vertex_to_chunk_plane(vertex, chunk_size, draco_grid_size, offset):
    """Classify a single vertex to its nearest chunk plane.

    Returns
    -------
    tuple or None
        (axis, plane_value) for the closest chunk plane within tolerance,
        or None if the vertex is not on any chunk plane.
    """
    tol = draco_grid_size / 2
    shifted = vertex - offset

    dist_behind = np.mod(shifted, chunk_size)  # (3,)
    dist_ahead = chunk_size - dist_behind  # (3,)

    # Find minimum distance to any plane on any axis
    best_axis = None
    best_dist = np.inf
    best_plane_value = None

    for axis in range(3):
        d_behind = dist_behind[axis]
        d_ahead = dist_ahead[axis]

        # Check behind (strict <)
        if d_behind < tol and d_behind < best_dist:
            best_dist = d_behind
            best_axis = axis
            best_plane_value = vertex[axis] - d_behind

        # Check ahead (<= for draco asymmetry)
        if d_ahead <= tol and d_ahead < best_dist:
            best_dist = d_ahead
            best_axis = axis
            best_plane_value = vertex[axis] + d_ahead

    if best_axis is None:
        return None
    return (best_axis, best_plane_value)


# ---------------------------------------------------------------------------
# Phase 2: Generalized capping
# ---------------------------------------------------------------------------


def _extract_boundary_loops(faces):
    """Extract ordered boundary loops from a triangle mesh.

    Boundary edges are edges that appear in exactly one face.

    Returns a list of loops, each a list of vertex indices in order.
    """
    from collections import defaultdict

    edge_count = defaultdict(list)
    for fi, face in enumerate(faces):
        for i in range(3):
            e = tuple(sorted((face[i], face[(i + 1) % 3])))
            edge_count[e].append(fi)

    boundary_edges = {e for e, flist in edge_count.items() if len(flist) == 1}

    if not boundary_edges:
        return []

    adj = defaultdict(set)
    for a, b in boundary_edges:
        adj[a].add(b)
        adj[b].add(a)

    visited_edges = set()
    loops = []
    for start in adj:
        if all(tuple(sorted((start, n))) in visited_edges for n in adj[start]):
            continue
        loop = [start]
        prev = start
        current = None
        for n in adj[start]:
            if tuple(sorted((start, n))) not in visited_edges:
                current = n
                break
        if current is None:
            continue

        visited_edges.add(tuple(sorted((prev, current))))
        loop.append(current)

        while current != start:
            next_v = None
            for n in adj[current]:
                if n == prev:
                    continue
                if tuple(sorted((current, n))) not in visited_edges:
                    next_v = n
                    break
            if next_v is None:
                break
            visited_edges.add(tuple(sorted((current, next_v))))
            prev = current
            current = next_v
            if current != start:
                loop.append(current)

        if current == start and len(loop) >= 3:
            loops.append(loop)

    return loops


def _intersect_segment_plane(v1, v2, axis, plane_value):
    """Compute where the segment v1->v2 crosses an axis-aligned plane.

    Returns the 3D intersection point, or None if parallel or out of range.
    """
    d = v2[axis] - v1[axis]
    if abs(d) < 1e-12:
        return None
    t = (plane_value - v1[axis]) / d
    if t < 0 or t > 1:
        return None
    return v1 + t * (v2 - v1)


def _split_loop_by_plane(loop_vertices, chunk_size, draco_grid_size, offset):
    """Split a boundary loop into per-plane segments with corner geometry inserted.

    Walks the loop, classifying each vertex to its chunk plane. Handles three
    transition cases:
      1. Same plane — accumulate vertex into current segment
      2. Different axes — insert intersection points + corner vertex
      3. Off-plane (None) — split point, end current segment

    Returns
    -------
    dict
        Mapping (axis, plane_value) -> list of (N_i, 3) vertex arrays.
        Each array is an ordered polygon to be triangulated on that plane.
    """
    n = len(loop_vertices)
    classifications = [
        classify_vertex_to_chunk_plane(
            loop_vertices[i], chunk_size, draco_grid_size, offset
        )
        for i in range(n)
    ]

    # If all vertices are on the same plane, return immediately
    non_none = [c for c in classifications if c is not None]
    if not non_none:
        return {}
    if len(set(non_none)) == 1 and all(c is not None for c in classifications):
        return {non_none[0]: [loop_vertices.copy()]}

    # Walk the loop and build segments per plane
    plane_segments = {}  # (axis, plane_value) -> list of segment lists
    current_plane = None
    current_segment = []

    def _flush_segment():
        nonlocal current_segment, current_plane
        if current_plane is not None and len(current_segment) >= 1:
            if current_plane not in plane_segments:
                plane_segments[current_plane] = []
            plane_segments[current_plane].append(current_segment)
        current_segment = []

    # Find first on-plane vertex to start cleanly
    start_offset = 0
    for i in range(n):
        if classifications[i] is not None:
            start_offset = i
            break
    else:
        return {}

    # Reorder so we start at an on-plane vertex
    order = [(start_offset + i) % n for i in range(n)]

    current_plane = classifications[order[0]]
    current_segment = [loop_vertices[order[0]]]

    for idx in range(len(order)):
        i = order[idx]
        next_idx = (idx + 1) % n
        j = order[next_idx]

        this_class = classifications[i]
        next_class = classifications[j]

        if this_class is None:
            # Current vertex is off-plane — should not be in a segment
            # (we started at an on-plane vertex, so this means we just transitioned)
            continue

        if next_class is None:
            # Next vertex is off-plane — end current segment
            _flush_segment()
            current_plane = None
            continue

        if this_class == next_class:
            # Same plane — add next vertex to segment
            if next_idx != 0:  # Don't re-add start vertex
                current_segment.append(loop_vertices[j])
            continue

        # Transition between different planes
        v1 = loop_vertices[i]
        v2 = loop_vertices[j]
        this_axis, this_pv = this_class
        next_axis, next_pv = next_class

        if this_axis == next_axis:
            # Same axis, different plane value — shouldn't happen per our analysis,
            # but handle gracefully with a midpoint split
            mid = (v1 + v2) / 2
            current_segment.append(mid)
            _flush_segment()
            current_plane = next_class
            current_segment = [mid]
            if next_idx != 0:
                current_segment.append(loop_vertices[j])
        else:
            # Different axes — corner transition
            # Find where segment crosses each plane
            cross_next = _intersect_segment_plane(v1, v2, next_axis, next_pv)
            cross_this = _intersect_segment_plane(v1, v2, this_axis, this_pv)

            if cross_this is not None and cross_next is not None:
                # Two crossings — insert corner vertex
                d_this = np.linalg.norm(cross_this - v1)
                d_next = np.linalg.norm(cross_next - v1)

                if d_this < d_next:
                    # cross_this is closer to v1: exit current plane first
                    corner = np.copy(cross_this)
                    corner[next_axis] = next_pv
                    current_segment.append(cross_this)
                    current_segment.append(corner)
                    _flush_segment()
                    current_plane = next_class
                    current_segment = [corner, cross_next]
                else:
                    # cross_next is closer to v1
                    corner = np.copy(cross_next)
                    corner[this_axis] = this_pv
                    current_segment.append(cross_next)
                    current_segment.append(corner)
                    _flush_segment()
                    current_plane = next_class
                    current_segment = [corner, cross_this]
            else:
                # Only one crossing found (or none) — use what we have
                cross = cross_next if cross_next is not None else cross_this
                if cross is None:
                    cross = (v1 + v2) / 2
                current_segment.append(cross)
                _flush_segment()
                current_plane = next_class
                current_segment = [cross]

            if next_idx != 0:
                current_segment.append(loop_vertices[j])

    # Flush the last segment
    _flush_segment()

    def _polygonize_segments_on_plane(segments, axis, plane_value):
        """Polygonize split segments on one plane using Shapely."""
        from shapely.geometry import LineString
        from shapely.ops import polygonize, unary_union

        axes_2d = [a for a in range(3) if a != axis]
        lines = []

        for segment in segments:
            pts = np.array(segment, dtype=np.float64)
            if len(pts) < 2:
                continue

            # Remove near-duplicate consecutive points.
            diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            keep = np.concatenate([[True], diffs > 1e-6])
            pts = pts[keep]
            if len(pts) < 2:
                continue

            line = LineString(pts[:, axes_2d])
            if not line.is_empty and line.length > 1e-6:
                lines.append(line)

        if not lines:
            return []

        try:
            merged_lines = unary_union(lines)
            polygons_2d = list(polygonize(merged_lines))
        except Exception:
            return []

        polygons_3d = []
        for poly in polygons_2d:
            if poly.is_empty or poly.area <= 1e-6:
                continue
            coords_2d = np.array(poly.exterior.coords[:-1], dtype=np.float64)
            if len(coords_2d) < 3:
                continue

            polygon_3d = np.zeros((len(coords_2d), 3), dtype=np.float64)
            polygon_3d[:, axes_2d] = coords_2d
            polygon_3d[:, axis] = plane_value

            # Remove duplicate closing point if present.
            if (
                len(polygon_3d) > 1
                and np.linalg.norm(polygon_3d[0] - polygon_3d[-1]) < 1e-6
            ):
                polygon_3d = polygon_3d[:-1]

            if len(polygon_3d) >= 3:
                polygons_3d.append(polygon_3d)

        return polygons_3d

    # Merge segments on the same plane into closed polygons
    merged = {}
    for plane_key, segments in plane_segments.items():
        axis, plane_value = plane_key
        polygons_3d = _polygonize_segments_on_plane(segments, axis, plane_value)

        # Fallback to previous behavior if polygonization yielded nothing.
        if not polygons_3d:
            all_pts = np.concatenate([np.array(s) for s in segments], axis=0)
            diffs = np.linalg.norm(np.diff(all_pts, axis=0), axis=1)
            keep = np.concatenate([[True], diffs > 1e-6])
            all_pts = all_pts[keep]
            if len(all_pts) > 1 and np.linalg.norm(all_pts[0] - all_pts[-1]) < 1e-6:
                all_pts = all_pts[:-1]
            if len(all_pts) >= 3:
                polygons_3d = [all_pts]

        if polygons_3d:
            merged[plane_key] = polygons_3d

    return merged


def _triangulate_face_polygon(polygon_3d, axis, plane_value):
    """Triangulate a polygon on a chunk plane using constrained Delaunay.

    Projects to 2D by dropping the normal axis, triangulates, returns indices.
    """
    import triangle as tr

    axes_2d = [a for a in range(3) if a != axis]
    pts_2d = polygon_3d[:, axes_2d]

    n = len(pts_2d)
    if n < 3:
        return np.empty((0, 3), dtype=int)

    # Remove near-duplicate consecutive vertices (e.g. from plane snapping)
    # which create zero-length segments that crash the triangle C library.
    keep = [0]
    for i in range(1, n):
        if np.linalg.norm(pts_2d[i] - pts_2d[keep[-1]]) > 1e-6:
            keep.append(i)
    if len(keep) > 1 and np.linalg.norm(pts_2d[keep[-1]] - pts_2d[keep[0]]) < 1e-6:
        keep = keep[:-1]
    keep = np.array(keep)

    if len(keep) < 3:
        return np.empty((0, 3), dtype=int)

    pts_2d = pts_2d[keep]
    n = len(pts_2d)

    # Detect non-consecutive duplicate 2D points (pinched polygon).
    # These cause the triangle C library to segfault.
    from scipy.spatial import cKDTree

    tree = cKDTree(pts_2d)
    pairs = tree.query_pairs(r=1e-6)
    if pairs:
        return np.empty((0, 3), dtype=int)

    segments = np.array([[i, (i + 1) % n] for i in range(n)])

    tri_input = {"vertices": pts_2d.astype(np.float64), "segments": segments}
    tri_output = tr.triangulate(tri_input, "p")

    if "triangles" not in tri_output:
        return np.empty((0, 3), dtype=int)

    triangles = tri_output["triangles"]

    # Triangle can introduce additional vertices for pathological polygons
    # (e.g. self-intersections). In that case, indices no longer map to
    # the original input polygon, so skip this polygon robustly.
    if np.any(triangles < 0) or np.any(triangles >= n):
        return np.empty((0, 3), dtype=int)

    return keep[triangles]


def cap_mesh_at_chunk_boundaries(mesh, cv, debug_stats=None):
    """Cap open boundary holes at any chunk boundary plane.

    Detects boundary loops whose vertices lie on chunk planes (internal or
    external), splits them by plane (inserting corner geometry at cross-axis
    transitions), and fills each sub-polygon with constrained Delaunay
    triangulation.

    Parameters
    ----------
    mesh : tuple
        (vertices, faces) where vertices is (N, 3) and faces is (M, 3).
    cv : cloudvolume.CloudVolume
        CloudVolume object for chunk metadata.

    Returns
    -------
    tuple
        (new_vertices, new_faces) with cap triangles appended.
    """
    vertices, faces = mesh
    chunk_size, draco_grid_size, offset = get_chunk_grid_params(cv)
    tol = draco_grid_size / 2

    # Classify all vertices to chunk planes (per-axis)
    on_plane, _ = classify_vertices_to_chunk_planes(
        vertices, chunk_size, draco_grid_size, offset
    )
    # A vertex is "on a boundary" if it's on a chunk plane on any axis
    on_any_plane = np.any(on_plane, axis=1)

    loops = _extract_boundary_loops(faces)

    if debug_stats is not None:
        debug_stats.setdefault("loops_total", 0)
        debug_stats.setdefault("loops_skipped_low_boundary_signal", 0)
        debug_stats.setdefault("loops_with_plane_polygons", 0)
        debug_stats.setdefault("loops_without_plane_polygons", 0)
        debug_stats.setdefault("plane_polygons_total", 0)
        debug_stats.setdefault("plane_polygons_triangulated", 0)
        debug_stats.setdefault("plane_polygons_skipped_no_triangles", 0)
        debug_stats.setdefault("cap_faces_added", 0)
        debug_stats.setdefault("caps_geometry_checked", 0)
        debug_stats.setdefault("caps_geometry_suspect", 0)
        debug_stats.setdefault("caps_geometry_ratio_min", np.inf)
        debug_stats.setdefault("caps_geometry_ratio_max", 0.0)
        debug_stats.setdefault("caps_geometry_examples", [])
        debug_stats["loops_total"] += len(loops)

    new_verts_list = [vertices]
    new_faces_list = [faces]
    next_vert_idx = len(vertices)

    for loop in loops:
        loop_arr = np.array(loop)

        # Require at least a minimal chunk-plane signal; splitting logic below
        # will discard loops that still cannot be represented on chunk planes.
        on_plane_count = np.sum(on_any_plane[loop_arr])
        if on_plane_count < 3:
            if debug_stats is not None:
                debug_stats["loops_skipped_low_boundary_signal"] += 1
            continue

        loop_verts = vertices[loop_arr]

        plane_polygons = _split_loop_by_plane(
            loop_verts, chunk_size, draco_grid_size, offset
        )

        if debug_stats is not None:
            if plane_polygons:
                debug_stats["loops_with_plane_polygons"] += 1
            else:
                debug_stats["loops_without_plane_polygons"] += 1

        for (axis, plane_value), polygons in plane_polygons.items():
            for polygon_3d in polygons:
                if debug_stats is not None:
                    debug_stats["plane_polygons_total"] += 1
                if len(polygon_3d) < 3:
                    continue

                # Snap all polygon vertices to the plane
                polygon_3d = polygon_3d.copy()
                polygon_3d[:, axis] = plane_value

                tri_indices = _triangulate_face_polygon(polygon_3d, axis, plane_value)

                if debug_stats is not None:
                    # Geometry sanity check: compare realized cap area to a
                    # characteristic circular area from mean loop radius.
                    axes_2d = [a for a in range(3) if a != axis]
                    pts_2d = polygon_3d[:, axes_2d]
                    centroid_2d = np.mean(pts_2d, axis=0)
                    mean_radius = np.mean(np.linalg.norm(pts_2d - centroid_2d, axis=1))
                    expected_area = np.pi * (mean_radius**2)

                    cap_area = 0.0
                    if len(tri_indices) > 0 and expected_area > 1e-12:
                        tri_pts = pts_2d[tri_indices]  # (T, 3, 2)
                        v1 = tri_pts[:, 1] - tri_pts[:, 0]
                        v2 = tri_pts[:, 2] - tri_pts[:, 0]
                        cap_area = (
                            0.5
                            * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]).sum()
                        )

                    area_ratio = (
                        cap_area / expected_area if expected_area > 1e-12 else 0.0
                    )
                    debug_stats["caps_geometry_checked"] += 1
                    debug_stats["caps_geometry_ratio_min"] = min(
                        debug_stats["caps_geometry_ratio_min"], area_ratio
                    )
                    debug_stats["caps_geometry_ratio_max"] = max(
                        debug_stats["caps_geometry_ratio_max"], area_ratio
                    )

                    # Broad thresholds: flag severe mismatch only.
                    if area_ratio < 0.10 or area_ratio > 1.80:
                        debug_stats["caps_geometry_suspect"] += 1
                        if len(debug_stats["caps_geometry_examples"]) < 8:
                            debug_stats["caps_geometry_examples"].append(
                                {
                                    "axis": int(axis),
                                    "plane_value": float(plane_value),
                                    "polygon_n": int(len(polygon_3d)),
                                    "area_ratio": float(area_ratio),
                                    "mean_radius": float(mean_radius),
                                    "cap_area": float(cap_area),
                                }
                            )

                if len(tri_indices) == 0:
                    if debug_stats is not None:
                        debug_stats["plane_polygons_skipped_no_triangles"] += 1
                    continue
                if debug_stats is not None:
                    debug_stats["plane_polygons_triangulated"] += 1

                # Map polygon vertices back to mesh vertex indices
                poly_to_mesh_idx = np.empty(len(polygon_3d), dtype=int)
                for pi, coord in enumerate(polygon_3d):
                    dists = np.linalg.norm(vertices[loop_arr] - coord, axis=1)
                    min_dist_idx = np.argmin(dists)
                    if dists[min_dist_idx] < tol:
                        poly_to_mesh_idx[pi] = loop_arr[min_dist_idx]
                    else:
                        new_verts_list.append(coord.reshape(1, 3))
                        poly_to_mesh_idx[pi] = next_vert_idx
                        next_vert_idx += 1

                cap_faces = poly_to_mesh_idx[tri_indices]
                new_faces_list.append(cap_faces)
                if debug_stats is not None:
                    debug_stats["cap_faces_added"] += len(cap_faces)

    new_vertices = np.concatenate(new_verts_list, axis=0).astype(np.float32)
    new_faces = np.concatenate(new_faces_list, axis=0).astype(np.uint32)

    if debug_stats is not None and debug_stats["caps_geometry_checked"] == 0:
        debug_stats["caps_geometry_ratio_min"] = np.nan

    return new_vertices, new_faces


# ---------------------------------------------------------------------------
# Mesh repair utilities
# ---------------------------------------------------------------------------


def fill_holes(mesh, hole_size=10000):
    poly = mesh_to_poly(mesh)
    filled = poly.fill_holes(hole_size)
    return poly_to_mesh(filled)


def fix_mesh(mesh):
    import pymeshfix

    mf = pymeshfix.MeshFix(mesh[0].astype(np.float64), mesh[1].astype(np.int32))
    mf.clean()
    out = poly_to_mesh(mf.mesh)
    if mesh[0].dtype == np.float32:
        out = (out[0].astype(np.float32), out[1])
    return out


def remove_degenerate_faces(mesh):
    """Remove faces with repeated vertex indices (zero-area degenerate triangles)."""
    verts, faces = mesh
    degenerate = (
        (faces[:, 0] == faces[:, 1])
        | (faces[:, 1] == faces[:, 2])
        | (faces[:, 0] == faces[:, 2])
    )
    if degenerate.any():
        faces = faces[~degenerate]
    return (verts, faces)


def is_valid_mesh(mesh):
    """Check if a mesh has enough geometry to potentially become watertight."""
    verts, faces = mesh
    if faces.shape[0] < 4:
        return False
    extent = verts.max(axis=0) - verts.min(axis=0)
    if np.any(extent == 0):
        return False
    return True


def is_watertight(mesh):
    import trimesh

    trimesh_mesh = trimesh.Trimesh(*mesh, process=False)
    return trimesh_mesh.is_watertight


def clean_chunkedgraph_mesh(mesh, cv, debug_stats=None):
    chunk_size, draco_grid_size, offset = get_chunk_grid_params(cv)

    mesh = remove_degenerate_faces(mesh)
    mesh = deduplicate_chunk_boundaries(mesh, chunk_size, draco_grid_size, offset)
    mesh = clean_mesh(mesh, tolerance=5)
    mesh = cap_mesh_at_chunk_boundaries(mesh, cv, debug_stats=debug_stats)
    mesh = fill_holes(mesh)

    return mesh
