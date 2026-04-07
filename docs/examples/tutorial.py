# %% [markdown]
# # Condensed HKS Pipeline
#
# The [Heat Kernel Signature (HKS)](https://en.wikipedia.org/wiki/Heat_kernel_signature)
# is a spectral shape descriptor that captures local geometry at multiple spatial
# scales. `condensed_hks_pipeline` computes HKS features across a mesh, then
# agglomerates nearby vertices into a compact graph of "condensed nodes" — each
# with a feature vector summarizing HKS values within its region.
#
# This tutorial walks through:
# 1. Loading a sample dendrite mesh
# 2. Running `condensed_hks_pipeline`
# 3. Inspecting the condensed feature output
# 4. Visualizing a single HKS feature on the mesh

# %%
import base64
import warnings

import pyvista as pv
from IPython.display import HTML

from meshmash import condensed_hks_pipeline, fetch_sample_mesh

# %% [markdown]
# ## Load a sample mesh
#
# `fetch_sample_mesh` downloads a pre-packaged dendrite mesh from the
# [MICrONs dataset](https://www.microns-explorer.org/) (cached locally after
# the first call).

# %%
vertices, faces = fetch_sample_mesh("microns_dendrite_sample")
print(f"Vertices: {len(vertices):,}  Faces: {len(faces):,}")

# %% [markdown]
# ## Run the pipeline

# %%
result = condensed_hks_pipeline((vertices, faces), verbose=True)

# %% [markdown]
# ## Inspect condensed features
#
# `result.condensed_features` is a DataFrame with one row per condensed node
# and one column per HKS timescale (short to long).

# %%
result.condensed_features.head()

# %% [markdown]
# ## Visualize on the mesh
#
# Map one feature column back to the simplified mesh vertices using
# `result.simple_labels` — an integer array (one entry per vertex) indicating
# which condensed node each vertex belongs to.

# %%
poly = pv.make_tri_mesh(*result.simple_mesh)
col = result.condensed_features.columns[10]
scalars = result.condensed_features[col].reindex(result.simple_labels).values

plotter = pv.Plotter(off_screen=True, window_size=(800, 500))
plotter.add_mesh(poly, scalars=scalars, cmap="coolwarm", show_scalar_bar=False)
plotter.view_isometric()
plotter.export_html("hks_feature.html")

b64 = base64.b64encode(open("hks_feature.html", "rb").read()).decode()
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    display(
        HTML(
            f'<iframe src="data:text/html;base64,{b64}" style="width:100%;height:500px;border:none;"></iframe>'
        )
    )

# %% [markdown]
# ## Using CloudVolume for any neuron
#
# To run on an arbitrary neuron from a segmentation dataset, fetch the mesh
# with [CloudVolume](https://github.com/seung-lab/cloud-volume) and pass it to
# `condensed_hks_pipeline` in the same way:
#
# ```python
# from cloudvolume import CloudVolume
#
# cv = CloudVolume(
#     "graphene://https://minnie.microns-daf.com/segmentation/table/minnie3_v1",
#     progress=False,
# )
# root_id = 864691136618675163  # MICrONs pyramidal neuron
# raw = cv.mesh.get(root_id)[root_id]
# raw = raw.deduplicate_vertices(is_chunk_aligned=True)
# vertices, faces = raw.vertices, raw.faces
#
# result = condensed_hks_pipeline((vertices, faces), verbose=True)
# ```
#
# Full neurons are considerably larger than the dendrite sample; the pipeline
# will take longer and use more RAM, but the call is otherwise identical.
