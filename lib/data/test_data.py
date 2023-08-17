import pytest

from .dataset import prepare_dataloader


@pytest.mark.parametrize(
    (
        "dataset",
        "train_size",
        "eval_size",
        "batch_size",
        "seed",
        "only_initial_shuffle_train",
        "input_shape",
        "num_factors",
    ),
    [
        ("shapes3d", 100, 20, 10, 0, False, (3, 64, 64), 6),
        ("shapes3d", 200, 30, 15, 1, True, (3, 64, 64), 6),
    ],
)
def test_prepare_dataloader(
    dataset: str,
    train_size: int,
    eval_size: int,
    batch_size: int,
    seed: int,
    only_initial_shuffle_train: bool,
    input_shape: list,
    num_factors: int,
):
    trainloader, evalloader = prepare_dataloader(
        dataset=dataset,
        train_size=train_size,
        eval_size=eval_size,
        batch_size=batch_size,
        seed=seed,
        only_initial_shuffle_train=only_initial_shuffle_train,
    )
    for images, labels in trainloader:
        assert images.shape[0] == batch_size
        assert images.shape[1:] == input_shape
        assert labels.shape[0] == batch_size
        assert labels.shape[1] == num_factors
        break
    assert (
        len(trainloader) == train_size // batch_size
        if train_size % batch_size == 0
        else train_size // batch_size + 1
    )
    for images, labels in evalloader:
        assert images.shape[0] == batch_size
        assert images.shape[1:] == input_shape
        assert labels.shape[0] == batch_size
        assert labels.shape[1] == num_factors
        break
    assert (
        len(evalloader) == eval_size // batch_size
        if eval_size % batch_size == 0
        else eval_size // batch_size + 1
    )


@pytest.mark.parametrize(
    (
        "dataset",
        "train_size",
        "eval_size",
        "batch_size",
        "seed",
        "only_initial_shuffle_train",
    ),
    [
        ("shapes3d", 320000, 160001, 10, 0, False),
    ],
)
def test_prepare_dataloader_exception(
    dataset: str,
    train_size: int,
    eval_size: int,
    batch_size: int,
    seed: int,
    only_initial_shuffle_train: bool,
):
    with pytest.raises(AssertionError) as e:
        _ = prepare_dataloader(
            dataset=dataset,
            train_size=train_size,
            eval_size=eval_size,
            batch_size=batch_size,
            seed=seed,
            only_initial_shuffle_train=only_initial_shuffle_train,
        )
    assert (
        str(e.value)
        == "size of train set + test set must be smaller than or equal to # of all samples"
    )
