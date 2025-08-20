import logging
from io import BytesIO
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from cloudfiles import CloudFiles

HEADER_FILE_NAME = "header.txt"


def interpret_path(path: Union[str, Path], **kwargs) -> Path:
    if isinstance(path, str):
        path = Path(path)

    if path.suffix == "":
        file_name = None
        path = path
    else:
        file_name = path.name
        path = path.parent

    path_str = str(path)
    if path_str.startswith("gs:/"):
        path_str = "gs://" + path_str[4:]
    elif path_str.startswith("file:/"):
        path_str = "file://" + path_str[6:]

    cf = CloudFiles(path_str, **kwargs)

    return cf, file_name


def _read_header(cf: CloudFiles, header_file_name=HEADER_FILE_NAME) -> list[str]:
    header_bytes = cf.get(header_file_name)
    if header_bytes is None:
        return None
    header = header_bytes.decode()
    columns = header.split("\t")
    return columns


def _check_header(
    cf: CloudFiles,
    columns: list[str],
    check_header: bool = True,
    header_file_name: str = HEADER_FILE_NAME,
) -> None:
    if check_header:
        if cf.exists(header_file_name):
            read_columns = _read_header(cf, header_file_name)
            if not np.array_equal(columns, read_columns):
                raise ValueError(
                    f"Columns in {header_file_name} do not match columns in features: {columns} != {read_columns}"
                )
        else:
            header = "\t".join(columns)
            header_bytes = header.encode()
            cf.put(header_file_name, header_bytes)


def save_condensed_features(
    path: Union[str, Path],
    features: pd.DataFrame,
    labels: np.ndarray,
    feature_dtype: type = np.float32,
    label_dtype: type = np.int32,
    check_header: bool = True,
):
    cf, file_name = interpret_path(path)

    columns: list = features.columns.tolist()

    # look for a header file with the name header.txt in the same directory

    if check_header:
        if cf.exists(HEADER_FILE_NAME):
            read_columns = _read_header(cf)
            if not np.array_equal(columns, read_columns):
                raise ValueError(
                    f"Columns in header.txt do not match columns in features: {columns} != {read_columns}"
                )
        else:
            header = "\t".join(columns)
            header_bytes = header.encode()
            cf.put(HEADER_FILE_NAME, header_bytes)

    pre_len = len(features)
    features = features.reindex(np.arange(-1, len(features) - 1), copy=False)
    post_len = len(features)
    if pre_len != post_len:
        logging.warning("Feature rows were missing.")

    X = features.values

    with BytesIO() as bio:
        np.savez_compressed(
            bio,
            X=X.astype(feature_dtype),
            labels=labels.astype(label_dtype),
        )

        cf.put(file_name, bio.getvalue())


def read_condensed_features(path: Union[str, Path]) -> tuple[pd.DataFrame, np.ndarray]:
    cf, file_name = interpret_path(path)

    with BytesIO(cf.get(file_name)) as bio:
        data = np.load(bio)
        X = data["X"]
        labels = data["labels"]

    columns = _read_header(cf)
    index = np.arange(-1, len(X) - 1)
    features = pd.DataFrame(X, columns=columns, index=index)

    return features, labels


def save_condensed_edges(
    path: Union[str, Path], edges: pd.DataFrame, check_header: bool = True
):
    cf, file_name = interpret_path(path)

    edge_list = edges[["source", "target"]].values.astype(np.int32)
    edge_features = edges.drop(columns=["source", "target"])
    columns = edge_features.columns.tolist()
    edge_features = edge_features.values.astype(np.float32)

    if check_header:
        if cf.exists(HEADER_FILE_NAME):
            read_columns = _read_header(cf)
            if not np.array_equal(columns, read_columns):
                raise ValueError(
                    f"Columns in header.txt do not match columns in features: {columns} != {read_columns}"
                )
        else:
            header = "\t".join(columns)
            header_bytes = header.encode()
            cf.put(HEADER_FILE_NAME, header_bytes)

    with BytesIO() as bio:
        np.savez_compressed(
            bio,
            edges=edge_list,
            edge_features=edge_features,
        )

        cf.put(file_name, bio.getvalue())


def read_condensed_edges(path: Union[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cf, file_name = interpret_path(path)

    with BytesIO(cf.get(file_name)) as bio:
        data = np.load(bio)
        edges = pd.DataFrame(data["edges"], columns=["source", "target"])
        edge_features = pd.DataFrame(data["edge_features"], columns=_read_header(cf))

    edges = pd.concat([edges, edge_features], axis=1)

    return edges


def save_condensed_graph(
    path: Union[str, Path],
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    nodes_dtype=np.float32,
    edges_dtype=np.int32,
    check_header: bool = True,
):
    cf, file_name = interpret_path(path)

    _check_header(
        cf,
        nodes.columns,
        check_header=check_header,
        header_file_name="nodes_header.txt",
    )
    _check_header(
        cf,
        edges.columns,
        check_header=check_header,
        header_file_name="edges_header.txt",
    )

    pre_len = len(nodes)
    nodes = nodes.reindex(np.arange(len(nodes)), copy=False)
    post_len = len(nodes)
    if pre_len != post_len:
        logging.warning("Node rows were missing.")

    with BytesIO() as bio:
        np.savez_compressed(
            bio, nodes=nodes.astype(nodes_dtype), edges=edges.astype(edges_dtype)
        )

        cf.put(file_name, bio.getvalue())


def read_condensed_graph(path: Union[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cf, file_name = interpret_path(path)

    with BytesIO(cf.get(file_name)) as bio:
        data = np.load(bio)
        nodes = data["nodes"]
        node_columns = _read_header(cf, header_file_name="nodes_header.txt")
        nodes = pd.DataFrame(
            nodes,
            columns=node_columns,
        )
        edges = data["edges"]
        edge_columns = _read_header(cf, header_file_name="edges_header.txt")
        edges = pd.DataFrame(edges, columns=edge_columns)

    return nodes, edges


def save_id_to_mesh_map(path: Union[str, Path], id_to_mesh_map: np.ndarray):
    assert id_to_mesh_map.shape[1] == 2

    cf, file_name = interpret_path(path)

    with BytesIO() as bio:
        np.savez_compressed(bio, id_to_mesh_map=id_to_mesh_map)

        cf.put(file_name, bio.getvalue())


def read_id_to_mesh_map(path: Union[str, Path]) -> np.ndarray:
    cf, file_name = interpret_path(path)

    with BytesIO(cf.get(file_name)) as bio:
        # try:
        #     data = np.load(bio)
        # except EOFError:
        #     # TODO add a retry here or not?
        #     return np.empty((0, 2), dtype=np.int32)
        data = np.load(bio)
        id_to_mesh_map = data["id_to_mesh_map"]

    return id_to_mesh_map


def save_array(path: Union[str, Path], array: np.ndarray):
    cf, file_name = interpret_path(path)

    with BytesIO() as bio:
        np.savez_compressed(bio, array=array)

        cf.put(file_name, bio.getvalue())


def read_array(path: Union[str, Path]) -> np.ndarray:
    cf, file_name = interpret_path(path)

    with BytesIO(cf.get(file_name)) as bio:
        data = np.load(bio)
        array = data["array"]

    return array

