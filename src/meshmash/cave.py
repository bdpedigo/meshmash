from typing import Optional, Union

import numpy as np
from caveclient import CAVEclient

from .types import Mesh
from .utils import project_points_to_mesh


def find_nucleus_point(
    root_id: int,
    client: CAVEclient,
    update_root_id: Union[bool, str] = True,
) -> Optional[np.ndarray]:
    if isinstance(update_root_id, bool) and update_root_id:
        current_root_id = client.chunkedgraph.suggest_latest_roots(root_id)
    elif update_root_id == "check":
        is_current = client.chunkedgraph.is_latest_roots([root_id])[0]
        if not is_current:
            raise ValueError(f"Root {root_id} is not latest")
        else:
            current_root_id = root_id
    if "minnie" in client.datastack_name:
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
                raise ValueError(
                    f"Found more than one neuron nucleus for root {root_id}"
                )
        elif len(nuc_table) == 0:
            raise ValueError(f"Found no nucleus for root {root_id}")
        else:
            pass  # has one nucleus``

    elif "v1dd" in client.datastack_name:
        nuc_table = client.materialize.views.nucleus_alternative_lookup(
            pt_root_id=current_root_id
        ).query(
            split_positions=True,
            desired_resolution=[1, 1, 1],
        )

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
    side: str = "post",
) -> np.ndarray:
    synapse_table_name = client.info.get_datastack_info()["synapse_table"]
    synapses = client.materialize.query_table(
        synapse_table_name,
        filter_equal_dict={f"{side}_pt_root_id": root_id},
        log_warning=False,
        split_positions=True,
        desired_resolution=[1, 1, 1],
    )
    synapses.query("pre_pt_root_id != post_pt_root_id", inplace=True)
    synapses.set_index("id", inplace=True)

    if len(synapses) == 0:
        return np.empty(shape=(0, 2), dtype="int32")

    synapse_locs = synapses[
        [f"{mapping_column}_x", f"{mapping_column}_y", f"{mapping_column}_z"]
    ].values

    indices = project_points_to_mesh(
        synapse_locs,
        mesh,
        distance_threshold=distance_threshold,
        return_distances=False,
    )

    synapses["mesh_index"] = indices.astype("int32")
    synapses.query("mesh_index != -1", inplace=True)

    out = synapses["mesh_index"]
    out = out.to_frame().reset_index().values
    return out
