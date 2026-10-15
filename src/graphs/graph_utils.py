import pickle
import pandas as pd

from utils.paths import (
    CLEANED_B3DB,
    CONFORMERS,
    MOL_GRAPHS,
    LINE_GRAPHS,
    DIST_GRAPHS,
    GEOM_FEATURES,
    require,
)
from graphs.molecular_graph import smiles_to_graph
from graphs.line_graph import smiles_to_line_graph
from graphs.adaptive_distance_graph import (
    mol_to_multiscale_distance_graphs,
    DEFAULT_CUTOFFS,
)
from graphs.geometric_features import extract_geometric_features


def build_all_graphs(cutoffs=DEFAULT_CUTOFFS):
    require(
        CLEANED_B3DB,
        CONFORMERS,
        hint="python main.py preprocess --stage conformers",
    )

    dataframe = pd.read_csv(CLEANED_B3DB)

    with open(CONFORMERS, "rb") as file:
        conformers = pickle.load(file)

    assert len(dataframe) == len(conformers), (
        "Cleaned data and conformers have different lengths."
    )

    molecular_graphs = []
    line_graphs = []
    distance_graphs = []
    geometry_features = []

    for smiles, item in zip(dataframe["smiles"], conformers):
        assert smiles == item["smiles"], "Molecule order mismatch."

        mol = item["mol"]
        molecular_graphs.append(smiles_to_graph(smiles))
        line_graphs.append(smiles_to_line_graph(smiles))
        distance_graphs.append(
            mol_to_multiscale_distance_graphs(mol, cutoffs)
        )
        geometry_features.append(extract_geometric_features(mol))

    outputs = [
        (molecular_graphs, MOL_GRAPHS),
        (line_graphs, LINE_GRAPHS),
        (distance_graphs, DIST_GRAPHS),
        (geometry_features, GEOM_FEATURES),
    ]

    for data, path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as file:
            pickle.dump(data, file)

    print(f"Graphs and geometric features saved: {len(molecular_graphs)}")