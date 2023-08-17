import torch
from torch.utils.data import Dataset

from .shapes3d import Shapes3D


class Subset(Dataset):
    """Class of train/eval set"""

    def __init__(self, dataset, indices: torch.Tensor):
        """initialize the set

        Args:
            dataset (child class of GroundTruthDataset): full dataset
            indices (torch.Tensor): indices for train set
        """
        self.images = (
            torch.permute(torch.tensor(dataset.images[indices.tolist()]), (0, 3, 1, 2))
            / 255
        )
        # (size, C, H, W)
        self.labels = torch.tensor(
            dataset.labels[indices.tolist()]
        )  # (size, num_factors)
        self._num_factors = dataset.num_factors
        self._factors_num_values = dataset.factors_num_values
        self._observation_shape = [
            dataset.observation_shape[2]
        ] + dataset.observation_shape[:2]

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
        return self._observation_shape  # (C, H, W)


def prepare_dataloader(dataset: str, train_size: int, eval_size: int, batch_size: int):
    """prepare dataloader for training

    Args:
        dataset (str): dataset name
        train_size (int): # of samples in train set
        eval_size (int): # of samples in eval set
        batch_size (int): batch size for training

    Retruns:
        trainloader (torch.utils.data.DataLoader): train set
    """
    # select dataset
    if dataset == "shapes3d":
        fullset = Shapes3D()
    else:
        raise ValueError(f"Dataset {dataset} is not supported")

    # indices for train set
    assert train_size + eval_size <= len(
        fullset
    ), "size of train set + test set must be smaller than or equal to # of all samples"
    indices = torch.randperm(len(fullset))
    train_indices, _ = torch.sort(indices[:train_size])
    eval_indices, _ = torch.sort(indices[train_size : train_size + eval_size])

    # train set
    trainset = Subset(fullset, train_indices)
    evalset = Subset(fullset, eval_indices)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )
    evalloader = torch.utils.data.DataLoader(
        evalset, batch_size=batch_size, shuffle=False
    )

    # close fullset
    fullset.close()

    return trainloader, evalloader
