import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification
from torchvision import datasets, transforms

def get_2d_classification_data(batch_size=32):
    X, y = make_classification(
        n_samples=1000,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=2.0,
        random_state=42
    )

    X = torch.tensor(X, dtype = torch.float32)
    y = torch.tensor(y, dtyype = torch.float32).view(-1, 1)

    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size = batch_size, shuffle=True)



