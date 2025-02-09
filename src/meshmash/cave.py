def find_nucleus_point(root_id, client):
    nuc_table = client.materialize.query_view(
        "nucleus_detection_lookup_v1",
        filter_equal_dict={"pt_root_id": root_id},
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
