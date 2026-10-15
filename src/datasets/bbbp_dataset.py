
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.paths import SPLITS_DIR, require


FORBIDDEN_DESCRIPTOR_COLUMNS = {"label", "labels", "logbb", "bbb+/bbb-", "bbb"}


class BBBDataset(Dataset):
    def __init__(
        self,
        graph_path,
        line_graph_path,
        distance_graph_path,
        descriptor_path,
        label_path,
        geometry_path=None,
        desc_mean=None,
        desc_std=None,
        geom_mean=None,
        geom_std=None,
    ):
        require(
            graph_path,
            line_graph_path,
            distance_graph_path,
            descriptor_path,
            label_path,
        )

        with open(graph_path, "rb") as file:
            self.graphs = pickle.load(file)

        with open(line_graph_path, "rb") as file:
            self.line_graphs = pickle.load(file)

        with open(distance_graph_path, "rb") as file:
            self.distance_graphs = pickle.load(file)

        descriptor_df = pd.read_csv(descriptor_path)
        columns = {column.lower() for column in descriptor_df.columns}
        bad_columns = columns & FORBIDDEN_DESCRIPTOR_COLUMNS

        assert not bad_columns, f"Target columns found in descriptors: {bad_columns}"
        assert descriptor_df.shape[1] == 7, "Expected 7 descriptors"

        descriptors = descriptor_df.to_numpy(dtype=np.float32)

        if desc_mean is not None:
            descriptors = (descriptors - desc_mean) / desc_std

        self.descriptors = descriptors

        self.geometry = None

        if geometry_path is not None and Path(geometry_path).exists():
            with open(geometry_path, "rb") as file:
                geometry_data = pickle.load(file)

            geometry_values = []

            for item in geometry_data:
                if isinstance(item, dict):
                    geometry_values.append(item["combined"])
                else:
                    geometry_values.append(item)

            geometry_values = np.array(geometry_values, dtype=np.float32)

            if geom_mean is not None:
                geometry_values = (geometry_values - geom_mean) / geom_std

            self.geometry = geometry_values

        label_df = pd.read_csv(label_path)
        self.labels = label_df["label"].to_numpy(dtype=np.float32)

        count = len(self.labels)

        assert len(self.graphs) == count
        assert len(self.line_graphs) == count
        assert len(self.distance_graphs) == count
        assert len(self.descriptors) == count

        if self.geometry is not None:
            assert len(self.geometry) == count

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            "graph": self.graphs[idx],
            "line_graph": self.line_graphs[idx],
            "distance_graph": self.distance_graphs[idx],
            "distance_graphs": self.distance_graphs[idx],
            "descriptor": torch.from_numpy(self.descriptors[idx]),
            "label": torch.tensor(self.labels[idx], dtype=torch.float),
        }

        if self.geometry is not None:
            item["geometry"] = torch.from_numpy(self.geometry[idx])
        else:
            item["geometry"] = torch.zeros(24, dtype=torch.float)

        return item


def compute_descriptor_stats(descriptor_path, train_idx):
    descriptors = pd.read_csv(descriptor_path).to_numpy(dtype=np.float32)
    train_descriptors = descriptors[train_idx]

    mean = train_descriptors.mean(axis=0)
    std = train_descriptors.std(axis=0)
    std[std < 1e-6] = 1.0

    return mean, std


def compute_geometry_stats(geometry_path, train_idx):
    if not Path(geometry_path).exists():
        return None, None

    with open(geometry_path, "rb") as file:
        geometry_data = pickle.load(file)

    geometry_values = []

    for item in geometry_data:
        if isinstance(item, dict):
            geometry_values.append(item["combined"])
        else:
            geometry_values.append(item)

    geometry_values = np.array(geometry_values, dtype=np.float32)
    train_geometry = geometry_values[train_idx]

    mean = train_geometry.mean(axis=0)
    std = train_geometry.std(axis=0)
    std[std < 1e-6] = 1.0

    return mean, std


def load_split_indices(split="scaffold"):
    split_files = [
        SPLITS_DIR / split / f"{part}.csv"
        for part in ["train", "val", "test"]
    ]

    require(*split_files)

    indices = []

    for file_path in split_files:
        data = pd.read_csv(file_path)
        indices.append(data["idx"].tolist())

    return indices
