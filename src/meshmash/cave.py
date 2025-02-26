from typing import Optional

import numpy as np
from caveclient import CAVEclient

from .types import Mesh
from .utils import project_points_to_mesh


def find_nucleus_point(root_id: int, client: CAVEclient) -> Optional[np.ndarray]:
    current_root_id = client.chunkedgraph.suggest_latest_roots(root_id)
    nuc_table = client.materialize.views.nucleus_detection_lookup_v1(
        pt_root_id=current_root_id
    ).query(
        split_positions=True,
        desired_resolution=[1, 1, 1],
    )
    if len(nuc_table) > 1:  # find correct nucleus, hopefully one is a neuron
        cell_table = client.materialize.query_table(
            "aibs_metamodel_mtypes_v661_v2",
            filter_in_dict={"target_id": nuc_table["id"].values},
        )
        if len(cell_table) == 1:
            neuron_nuc_id = cell_table["id_ref"].values[0]
            nuc_table = nuc_table.set_index("id").loc[[neuron_nuc_id]]
        else:
            raise ValueError(f"Found more than one neuron nucleus for root {root_id}")
    elif len(nuc_table) == 0:
        raise ValueError(f"Found no nucleus for root {root_id}")
    else:
        pass  # has one nucleus

    nuc_coords = nuc_table[["pt_position_x", "pt_position_y", "pt_position_z"]].values

    if nuc_coords.shape != (1, 3):
        raise ValueError(f"Error finding nucleus for root {root_id}")

    return nuc_coords


def get_synapse_mapping(
    root_id: int,
    mesh: Mesh,
    client: CAVEclient,
    distance_threshold: Optional[float] = None,
    mapping_column: str = "ctr_pt_position",
) -> np.ndarray:
    post_synapses = client.materialize.query_table(
        "synapses_pni_2",
        filter_equal_dict={"post_pt_root_id": root_id},
        log_warning=False,
        split_positions=True,
        desired_resolution=[1, 1, 1],
    )
    post_synapses.query("pre_pt_root_id != post_pt_root_id", inplace=True)
    post_synapses.set_index("id", inplace=True)

    synapse_locs = post_synapses[
        [f"{mapping_column}_x", f"{mapping_column}_y", f"{mapping_column}_z"]
    ].values

    indices = project_points_to_mesh(
        synapse_locs,
        mesh,
        distance_threshold=distance_threshold,
        return_distances=False,
    )

    post_synapses["mesh_index"] = indices.astype("int32")
    post_synapses.query("mesh_index != -1", inplace=True)

    out = post_synapses["mesh_index"]
    out = out.to_frame().reset_index().values
    return out
