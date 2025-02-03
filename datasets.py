import os
from pathlib import Path

import numpy as np

# Dataset names
NAMES = [
    "abalone",
    "auto_mpg",
    "bank32nh",
    "compactive",
    "concrete",
    "dee",
    "ele_2",
    "elevators",
    "kinematics32nh",
    "kinematics8nm",
    "laser",
    "machineCPU",
    "pumadyn32nh",
    "pyramidines",
    "stock",
    "TF1",
    "TF2",
    "TF3",
    "TF4",
    "TF5-5",
    "TF5",
    "triazines",
    "treasury",
    "wizmir"    
]


# PATH = Path(".") / "data"


def load_dataset(name):
    """Load dataset by name."""
    
    base_path = os.path.dirname(__file__)
    PATH = Path(base_path) / "data"
    
    if not PATH.exists():
        raise FileNotFoundError(f"Dataset folder {PATH} not found.")
    
    if name not in NAMES:
        raise ValueError(f"Dataset {name} not found.")

    train = PATH / f"{name}_trn.csv"
    test = PATH / f"{name}_tst.csv"

    # Dataset loading
    train = np.genfromtxt(train, delimiter=",")
    test = np.genfromtxt(test, delimiter=",")
    dataset = {
        "train_input": train[:, :-1],
        "train_label": train[:, [-1]].ravel(),
        "test_input": test[:, :-1],
        "test_label": test[:, [-1]].ravel(),
    }
    return dataset

def show_datasets():
    """Show available datasets statistics."""
    # print(f"{'name':<20}|{'n_vars':<10}|{'n_samples_train':<20}|{'n_samples_test':<20}")
    print(f"{'Nb.':<3}|{'Name':<20}|{'n_vars':<10}|{'n_samples_train':<20}|{'n_samples_test':<20}")
    print("-" * 80)
    
    for i, name in enumerate(NAMES, 1):
        dataset = load_dataset(name)
        n_vars = dataset["train_input"].shape[1]
        n_samples_train = dataset["train_input"].shape[0]
        n_samples_test = dataset["test_input"].shape[0]
        print(f"{i:<3}|{name:<20}|{n_vars:<10}|{n_samples_train:<20}|{n_samples_test:<20}")
        print("-" * 80)
        
if __name__ == "__main__":
    show_datasets()