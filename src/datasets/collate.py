
import torch
from torch_geometric.data import Batch


def collate_fn(batch):
    graphs = Batch.from_data_list([
        item["graph"] for item in batch
    ])

    line_graphs = Batch.from_data_list([
        item["line_graph"] for item in batch
    ])

    first_distance = batch[0].get(
        "distance_graphs",
        batch[0].get("distance_graph"),
    )

    if isinstance(first_distance, (list, tuple)):
        distance_graphs = []

        for cutoff in range(len(first_distance)):
            cutoff_graphs = [
                item.get(
                    "distance_graphs",
                    item.get("distance_graph"),
                )[cutoff]
                for item in batch
            ]

            distance_graphs.append(
                Batch.from_data_list(cutoff_graphs)
            )
    else:
        distance_graphs = Batch.from_data_list([
            item.get(
                "distance_graphs",
                item.get("distance_graph"),
            )
            for item in batch
        ])

    descriptors = torch.stack([
        item["descriptor"] for item in batch
    ])

    if "geometry" in batch[0]:
        geometry = torch.stack([
            item["geometry"] for item in batch
        ])
    else:
        geometry = None

    labels = torch.stack([
        item["label"] for item in batch
    ])

    return {
        "graph": graphs,
        "line_graph": line_graphs,
        "distance_graph": distance_graphs,
        "distance_graphs": distance_graphs,
        "descriptor": descriptors,
        "geometry": geometry,
        "label": labels,
    }

