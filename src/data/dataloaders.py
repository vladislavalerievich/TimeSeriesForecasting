from torch.utils.data import DataLoader, IterableDataset

from src.data.synthetic.synthetic_generation import generate_sine_waves


def train_val_loader(config, cpus_available):
    class SyntheticDataset(IterableDataset):
        def __init__(self, mode="train"):
            self.mode = mode
            self.config = config
            if self.mode == "val":
                # Pre-generate validation data with fixed seeds
                self.pregen_data = [
                    generate_sine_waves(
                        batch_size=config["batch_size"],
                        context_length=config["context_len"],
                        pred_length=config["pred_len"],
                        random_seed=42 + i,
                    )
                    for i in range(config["validation_rounds"])
                ]

        def __iter__(self):
            while True:
                if self.mode == "val":
                    yield from self.pregen_data
                else:
                    yield generate_sine_waves(
                        batch_size=config["batch_size"],
                        context_length=config["context_len"],
                        pred_length=config["pred_len"],
                    )

    train_loader = DataLoader(
        SyntheticDataset(mode="train"), batch_size=None, num_workers=cpus_available
    )

    val_loader = DataLoader(
        SyntheticDataset(mode="val"), batch_size=None, num_workers=cpus_available
    )

    return train_loader, val_loader
