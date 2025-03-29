from torch.utils.data import DataLoader, Dataset


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool = True,
    shuffle: bool = True,
):
    """
    Create a DataLoader for a given dataset.

    Args:
        dataset (Dataset): The dataset to load data from.
        batch_size (int): The batch size for the DataLoader.
        num_workers (int): The number of worker threads to use.
        pin_memory (bool, optional): Whether to pin memory. Defaults to True.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.

    Returns:
        DataLoader: A DataLoader for the given dataset.
    """
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
    )
