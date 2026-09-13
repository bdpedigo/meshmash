# %% [markdown]
# # Splitting a Mesh into Chunks
#
# Many operations in `meshmash` run on one piece of a mesh at a time, so a large
# mesh must first be cut into chunks of a workable size. There are two ways to
# make that cut:
#
# - `fit_mesh_split` bisects a chunk again and again along the Fiedler vector of
#   the graph Laplacian, which is the spectral cut.
# - `fit_mesh_split_geodesic` picks seed vertices spread across a chunk, then
#   gives every vertex to the nearest seed along the surface, which is the
#   geodesic Voronoi cut.
#
# This tutorial runs both on the same mesh and compares them.
#
# 1. Loading a sample dendrite mesh
# 2. Running each splitter
# 3. Comparing chunk counts, chunk sizes, run time, and cut length
# 4. Visualizing the two splits side by side
# 5. Computing on the chunks with `MeshStitcher`, and stitching the results back
#    into one array over the whole mesh

# %%
import html
import time
import warnings

import numpy as np
import pandas as pd
import pyvista as pv
from IPython.display import HTML, display

from meshmash import (
    MeshStitcher,
    apply_mesh_split,
    fetch_sample_mesh,
    fit_mesh_split,
    fit_mesh_split_geodesic,
    mesh_to_adjacency,
)

# %% [markdown]
# ## Load a sample mesh
#
# `fetch_sample_mesh` downloads a pre-packaged dendrite mesh from the
# [MICrONs dataset](https://www.microns-explorer.org/) (cached locally after
# the first call). This is the smaller of the two sample meshes, so both cuts
# below finish in under a second.

# %%
vertices, faces = fetch_sample_mesh("microns_dendrite_sample")
mesh = (vertices, faces)
print(f"Vertices: {len(vertices):,}  Faces: {len(faces):,}")

# %% [markdown]
# ## Build the adjacency once
#
# Both splitters take a mesh, and both start by turning it into a sparse
# adjacency matrix weighted by edge length. Build that matrix once and pass it
# to each splitter, so the timings below measure the cut and not the setup.

# %%
adjacency = mesh_to_adjacency(mesh)
print(f"Edges: {adjacency.nnz:,}")

# %% [markdown]
# ## Cut with recursive spectral bisection
#
# `fit_mesh_split` splits a chunk in two along the Fiedler vector, which is the
# eigenvector of the second smallest eigenvalue of the graph Laplacian. That
# vector separates the two ends of an elongated shape, so the cut tends to fall
# at a narrow place. The splitter repeats the bisection until every chunk holds
# at most `max_vertex_threshold` vertices.
#
# The sample mesh has about 32,000 vertices, so the default threshold of 20,000
# gives only two chunks. Use a smaller threshold to get a split worth looking at.

# %%
MAX_VERTICES = 5_000

start = time.perf_counter()
spectral_labels = fit_mesh_split(adjacency, max_vertex_threshold=MAX_VERTICES)
spectral_seconds = time.perf_counter() - start

print(f"Chunks: {spectral_labels.max() + 1}  Seconds: {spectral_seconds:.2f}")

# %% [markdown]
# Both splitters return the same thing: one integer label per vertex. Labels run
# from 0 upward, ordered from the largest chunk to the smallest. A vertex in a
# connected component smaller than `min_vertex_threshold` gets a label of -1.

# %%
print(spectral_labels[:20])
print(f"Unassigned vertices: {(spectral_labels == -1).sum()}")

# %% [markdown]
# ## Cut with geodesic Voronoi cells
#
# `fit_mesh_split_geodesic` cuts a chunk into several cells at once instead of
# two. It picks seeds by farthest-point sampling, then gives every vertex to the
# seed that reaches it first along the mesh surface.
#
# The number of seeds comes from `target_vertices`: a chunk of `n` vertices is
# cut with `ceil(n / target_vertices)` seeds. So `target_vertices` sets the chunk
# size you aim for, and `max_vertex_threshold` sets the size you refuse to
# exceed. Ask for chunks near 4,000 vertices to match the sizes the bisection
# produced above.

# %%
start = time.perf_counter()
geodesic_labels = fit_mesh_split_geodesic(
    adjacency, max_vertex_threshold=MAX_VERTICES, target_vertices=4_000
)
geodesic_seconds = time.perf_counter() - start

print(f"Chunks: {geodesic_labels.max() + 1}  Seconds: {geodesic_seconds:.2f}")

# %% [markdown]
# ## Compare the two splits


# %%
def summarize(labels, seconds):
    sizes = np.bincount(labels[labels != -1])
    kept = sum(
        len(submesh_faces) for _, submesh_faces in apply_mesh_split(mesh, labels)
    )
    return {
        "chunks": len(sizes),
        "smallest chunk": sizes.min(),
        "median chunk": int(np.median(sizes)),
        "largest chunk": sizes.max(),
        "seconds": round(seconds, 2),
        "faces on a cut": len(faces) - kept,
    }


comparison = pd.DataFrame(
    [
        summarize(spectral_labels, spectral_seconds),
        summarize(geodesic_labels, geodesic_seconds),
    ],
    index=["spectral", "geodesic"],
)
comparison

# %% [markdown]
# ## Visualize both splits
#
# Color every vertex by its chunk label, and draw one view per method. Drag either view
# to rotate it.

# %%
poly = pv.make_tri_mesh(vertices, faces)


def render(scalars, cmap="tab20"):
    """Draw the mesh by one value per vertex, and return an interactive frame.

    The scene goes in the ``srcdoc`` attribute, which is what PyVista itself
    does for a notebook. A ``data:`` URL is the other way to carry it, but
    Chrome drops any URL over 2 MB, and a scene this size passes that.
    """
    plotter = pv.Plotter(off_screen=True, window_size=(1000, 500))
    plotter.add_mesh(poly.copy(), scalars=scalars, cmap=cmap, show_scalar_bar=False)
    plotter.view_xy()
    scene = html.escape(plotter.export_html(filename=None).getvalue(), quote=True)

    return f'<iframe srcdoc="{scene}" style="width:100%;height:500px;border:none;"></iframe>'


def shuffle_labels(labels):
    """Shuffle chunk labels, so that two neighbors rarely get two near colors."""
    return np.random.default_rng(8888).permutation(labels.max() + 1)[labels]


with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    display(
        HTML(
            "<b>spectral</b>"
            + render(shuffle_labels(spectral_labels))
            + "<b>geodesic</b>"
            + render(shuffle_labels(geodesic_labels))
        )
    )

# %% [markdown]
# ## Computing on the chunks with MeshStitcher
#
# Most of the time the labels are not what you are after. You want one value per
# vertex, computed chunk by chunk, because the whole mesh is too large to
# compute on at once. `MeshStitcher` runs that loop for you:
#
# 1. `split_mesh` cuts the mesh with either method above, through its `method`
#    argument.
# 2. It grows every chunk by `overlap_distance`, so each submesh carries an
#    overlapping region of vertices from its neighbors.
# 3. `apply` runs your function on every submesh, in parallel when `n_jobs` is
#    not 1.
# 4. It writes each result back to the vertex that owns it, and drops the
#    overlap.
#
# The overlap is what makes the result match a computation on the whole mesh. A
# vertex at the border of a chunk has neighbors in the next chunk.
#
# Your function takes one submesh and returns one row per submesh vertex. Here a simple
# example is one that computes the unit surface normal at each vertex.


# %%
def vertex_normals(submesh):
    """The unit normal at every vertex of one submesh."""
    poly = pv.make_tri_mesh(*submesh)
    normals = poly.compute_normals(point_normals=True, cell_normals=False)["Normals"]
    return np.asarray(normals)


# %% [markdown]
# `overlap_distance` is a distance along the mesh, in the units of the vertex
# coordinates. These are nanometers, and this dendrite is about 46 micrometers
# from end to end, so an overlap of 500 nanometers is a thin one.

# %%
stitcher = MeshStitcher(mesh, n_jobs=1)
stitcher.split_mesh(
    max_vertex_threshold=MAX_VERTICES,
    method="geodesic",
    target_vertices=4_000,
    overlap_distance=500,
)

normals = stitcher.apply(vertex_normals)
print(f"Normals: {normals.shape}  Submeshes: {len(stitcher.submeshes)}")

# %% [markdown]
# ## Check the stitched result
#
# This mesh is small enough to compute on in one piece, so compare the stitched
# normals against normals computed on the whole mesh, and count the vertices
# where the two disagree by more than one degree.
#
# Two vertices of this mesh have a normal of zero, because the faces around them
# cancel out. An angle means nothing for those, so leave them out of the count.

# %%
whole_normals = vertex_normals(mesh)
defined = np.linalg.norm(whole_normals, axis=1) > 0


def degrees_off(stitched):
    dot = (stitched[defined] * whole_normals[defined]).sum(axis=1)
    return np.degrees(np.arccos(np.clip(dot, -1, 1)))


print(f"Vertices off by more than one degree: {(degrees_off(normals) > 1).sum()}")

# %% [markdown]
# Now shrink the overlap to 100 nanometers and run the same computation again.

# %%
thin_stitcher = MeshStitcher(mesh, n_jobs=1)
thin_stitcher.split_mesh(
    max_vertex_threshold=MAX_VERTICES,
    method="geodesic",
    target_vertices=4_000,
    overlap_distance=100,
)
thin_normals = thin_stitcher.apply(vertex_normals)

thin_error = degrees_off(thin_normals)
print(f"Vertices off by more than one degree: {(thin_error > 1).sum()}")
print(f"Worst vertex: {thin_error.max():.1f} degrees")

# %% [markdown]
# Those are the vertices on the chunk borders, where the overlapping region was
# too thin to reach every neighbor. Give the overlap enough room and the error
# goes away, as the first run shows.
#
# ## Plot the stitched result
#
# Color the mesh by the x component of the stitched normal, which says how much
# each patch of surface faces along the length of the dendrite. The result is
# one array over the whole mesh, with no seam at the chunk borders, even though
# a separate submesh produced each part of it.

# %%
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    display(HTML(render(normals[:, 0], cmap="coolwarm")))
