import pandas as pd
from torch.utils.data import Dataset

from graphs.molecular_graph import smiles_to_graph


class ChEMBLDataset(Dataset):
    """ChEMBL molecules as 2D graphs (for masked-atom pretraining)."""

    def __init__(self, csv_path, max_molecules=None):
        smiles = pd.read_csv(csv_path)["smiles"].tolist()
        if max_molecules is not None:
            smiles = smiles[:max_molecules]

        self.graphs = []
        for smi in smiles:
            try:
                self.graphs.append(smiles_to_graph(smi))
            except Exception:
                continue
        print("ChEMBL graphs:", len(self.graphs))

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]
