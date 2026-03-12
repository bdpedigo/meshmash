from typing import Any, Tuple, Union

import numpy as np
import pandas as pd

type Mesh = Union[tuple[np.ndarray, np.ndarray], Any]

type ArrayLike = Union[np.ndarray, list, tuple, pd.Series, pd.DataFrame, pd.Index]


def interpret_mesh(mesh) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(mesh, tuple):
        return mesh
    elif hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
        return mesh.vertices, mesh.faces
    else:
        raise ValueError(
            "Mesh should be tuple of vertices and faces or object with `vertices` and `faces` attributes"
        )
