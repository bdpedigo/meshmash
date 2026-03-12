"""Sample datasets for meshmash.

Use ``pip install meshmash[datasets]`` to enable downloading.
"""

import numpy as np

_GITHUB_BASE = (
    "https://github.com/bdpedigo/meshmash/releases/download/data-v1/{filename}"
)

# To verify: shasum -a 256 microns_neuron_sample.ply
_REGISTRY: dict[str, str | None] = {
    "microns_neuron_sample.ply": "84489fa274d7806c00507eb8212150d608e77fd78fe03c555a04f3791eae99ee",  # root_id: 864691136371550856
    "microns_dendrite_sample.ply": "e13f0284b23396570c0d59b6912355c762d813e578aef16a4ce17c5c95058ed6",  # small piece of dendrite from above
}


def fetch_sample_mesh(
    name: str = "microns_dendrite_sample",
) -> tuple[np.ndarray, np.ndarray]:
    """Download (once) and return a sample mesh as (vertices, faces).

    The file is cached in the OS-appropriate user cache directory
    (e.g. ``~/.cache/meshmash`` on Linux/macOS) and reused on subsequent calls.

    Parameters
    ----------
    name:
        Dataset name, without file extension. Available meshes:

        - ``"microns_neuron_sample"`` — full neuron mesh (MICrONs dataset,
          root_id 864691136371550856).
        - ``"microns_dendrite_sample"`` — small dendrite piece from the same
          neuron.

    Returns
    -------
    vertices : np.ndarray, shape (V, 3)
        Vertex positions.
    faces : np.ndarray, shape (F, 3)
        Triangle face indices.

    Examples
    --------
    >>> from meshmash import fetch_sample_mesh
    >>> vertices, faces = fetch_sample_mesh()
    >>> vertices.shape
    (N, 3)
    """
    try:
        import pooch
    except ImportError as exc:
        raise ImportError(
            "pooch is required to download sample datasets. "
            "Install it with:  pip install meshmash[datasets]"
        ) from exc

    try:
        import meshio
    except ImportError as exc:
        raise ImportError(
            "meshio is required to load sample meshes. "
            "It should be installed with meshmash by default."
        ) from exc

    filename = f"{name}.ply"
    if filename not in _REGISTRY:
        available = [k.removesuffix(".ply") for k in _REGISTRY]
        raise ValueError(f"Unknown sample mesh {name!r}. Available: {available}")

    known_hash = _REGISTRY[filename]

    url = _GITHUB_BASE.format(filename=filename)
    cache_dir = pooch.os_cache("meshmash")

    path = pooch.retrieve(
        url=url,
        known_hash=known_hash,
        path=cache_dir,
        fname=filename,
        progressbar=True,
    )

    mesh = meshio.read(path)
    vertices = mesh.points
    faces = mesh.cells_dict["triangle"]
    return vertices, faces
