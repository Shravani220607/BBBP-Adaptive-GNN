
from collections import defaultdict

import pandas as pd
from sklearn.model_selection import train_test_split

from utils.paths import CLEANED_B3DB, SPLITS_DIR, require


def split_scaffold_groups(groups, total, train_fraction=0.8, val_fraction=0.1):
    groups = sorted(groups, key=lambda group: (-len(group), group[0]))
    train, val, test = [], [], []

    for group in groups:
        if len(train) + len(group) <= train_fraction * total:
            train.extend(group)
        elif len(val) + len(group) <= val_fraction * total:
            val.extend(group)
        else:
            test.extend(group)

    return train, val, test


def scaffold_split(smiles_list):
    from rdkit.Chem.Scaffolds import MurckoScaffold

    groups = defaultdict(list)

    for index, smiles in enumerate(smiles_list):
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                smiles=smiles,
                includeChirality=False,
            )
        except Exception:
            scaffold = ""

        groups[scaffold].append(index)

    return split_scaffold_groups(list(groups.values()), len(smiles_list))


def random_split(labels, seed=42):
    indices = list(range(len(labels)))

    train, remaining = train_test_split(
        indices,
        test_size=0.2,
        random_state=seed,
        stratify=labels,
    )

    val, test = train_test_split(
        remaining,
        test_size=0.5,
        random_state=seed,
        stratify=[labels[i] for i in remaining],
    )

    return train, val, test


def save_split(df, name, train, val, test):
    folder = SPLITS_DIR / name
    folder.mkdir(parents=True, exist_ok=True)

    for part, indices in [
        ("train", train),
        ("val", val),
        ("test", test),
    ]:
        indices = sorted(indices)
        subset = df.iloc[indices].copy()
        subset.insert(0, "idx", indices)
        subset.to_csv(folder / f"{part}.csv", index=False)

    print(f"{name}: train={len(train)}, val={len(val)}, test={len(test)}")


def make_splits():
    require(CLEANED_B3DB, hint="python main.py preprocess --stage conformers")
    df = pd.read_csv(CLEANED_B3DB)

    save_split(df, "scaffold", *scaffold_split(df["smiles"].tolist()))
    save_split(df, "random", *random_split(df["label"].tolist()))
