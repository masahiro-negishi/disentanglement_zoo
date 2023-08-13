import pytest
from dataset import prepare_dataloader


@pytest.mark.parametrize(('dataset', 'train_size', 'batch_size', 'seed', 'input_shape', 'num_factors'), [
    ('shapes3d', 100, 10, 0, (64, 64, 3), 6),
    ('shapes3d', 200, 15, 1, (64, 64, 3), 6),
])
def test_prepare_dataloader(dataset: str, train_size: int, batch_size: int, seed: int, input_shape: list, num_factors: int):
    trainloader = prepare_dataloader(dataset=dataset, train_size=train_size, batch_size=batch_size, seed=seed)
    for images, labels in trainloader:
        assert images.shape[0] == batch_size
        assert images.shape[1:] == input_shape
        assert labels.shape[0] == batch_size
        assert labels.shape[1] == num_factors
        break
    assert len(trainloader) == train_size // batch_size if train_size % batch_size == 0 else train_size // batch_size + 1

@pytest.mark.parametrize(('dataset', 'train_size', 'batch_size', 'seed'), [
    ('shapes3d', 480001, 10, 0),
])
def test_prepare_dataloader_exception(dataset: str, train_size: int, batch_size: int, seed: int):
    with pytest.raises(AssertionError) as e:
        _ = prepare_dataloader(dataset=dataset, train_size=train_size, batch_size=batch_size, seed=seed)
    assert str(e.value) == "size of train set must be smaller than or equal to # of all samples"