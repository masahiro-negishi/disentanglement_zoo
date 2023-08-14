import torch
from torch.utils.data import Dataset

from .shapes3d import Shapes3D


class TrainSet(Dataset):
    """Class of train set"""

    def __init__(self, dataset, indices: torch.Tensor):
        """initialize training set

        Args:
            dataset (child class of GroundTruthDataset): full dataset
            indices (torch.Tensor): indices for train set
        """
        self.images = (
            torch.tensor(dataset.images[indices.tolist()]) / 255
        )  # (train_size, shape of input image)
        self.labels = torch.tensor(
            dataset.labels[indices.tolist()]
        )  # (train_size, num_factors)
        self._num_factors = dataset.num_factors
        self._factors_num_values = dataset.factors_num_values
        self._observation_shape = dataset.observation_shape

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.images)

    @property
    def num_factors(self):
        return self._num_factors

    @property
    def factors_num_values(self):
        return self._factors_num_values

    @property
    def observation_shape(self):
        return self._observation_shape


def prepare_dataloader(dataset: str, train_size: int, batch_size: int, seed: int):
    """prepare dataloader for training

    Args:
        dataset (str): dataset name
        train_size (int): # of samples in train set
        batch_size (int): batch size for training
        seed (int): random seed

    Retruns:
        trainloader (torch.utils.data.DataLoader): train set
    """
    # select dataset
    if dataset == "shapes3d":
        fullset = Shapes3D()
    else:
        raise ValueError(f"Dataset {dataset} is not supported")

    # indices for train set
    assert train_size <= len(
        fullset
    ), "size of train set must be smaller than or equal to # of all samples"
    torch.manual_seed(seed)
    indices = torch.randperm(len(fullset))
    train_indices, _ = torch.sort(indices[:train_size])

    # train set
    trainset = TrainSet(fullset, train_indices)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )

    # close fullset
    fullset.close()

    return trainloader
