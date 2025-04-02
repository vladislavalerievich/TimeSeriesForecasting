from torch.utils.data import DataLoader, Dataset

from src.data_handling.synthetic.synthetic_generation import generate_sine_waves


def train_val_loader(config, cpus_available):
    class SyntheticDataset(Dataset):
        def __init__(self, mode="train"):
            self.mode = mode
            self.config = config
            if mode == "val":
                # Pre-generate validation batches
                self.data = [
                    generate_sine_waves(
                        config["batch_size"],
                        config["context_len"],
                        config["pred_len"],
                        random_seed=42 + i,
                    )
                    for i in range(config["validation_rounds"])
                ]

        def __len__(self):
            return (
                config["training_rounds"]
                if self.mode == "train"
                else config["validation_rounds"]
            )

        def __getitem__(self, idx):
            if self.mode == "train":
                return generate_sine_waves(
                    config["batch_size"], config["context_len"], config["pred_len"]
                )
            return self.data[idx]

    train_loader = DataLoader(
        SyntheticDataset(mode="train"),
        batch_size=None,
        num_workers=cpus_available,
        shuffle=True,
    )

    val_loader = DataLoader(
        SyntheticDataset(mode="val"), batch_size=None, num_workers=cpus_available
    )

    return train_loader, val_loader
