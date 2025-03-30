import numpy as np
import torch


def generate_sine_waves(
    batch_size, context_length, pred_length, random_seed=None, noise_std=0.1
):
    if random_seed is not None:
        np.random.seed(random_seed)

    # Random parameters for each sample in batch
    amplitudes = np.random.uniform(0.1, 5.0, size=batch_size)
    frequencies = np.random.uniform(0.01, 0.1, size=batch_size)
    phases = np.random.uniform(0, 2 * np.pi, size=batch_size)
    noises = np.random.uniform(0, noise_std, size=batch_size)

    # Time feature generation
    def get_time_features(t):
        minutes = t % 60
        hours = (t // 60) % 24
        day = (t // (60 * 24)) % 31 + 1  # Simple 31-day month
        month = (t // (60 * 24 * 31)) % 12 + 1
        year = 2023 + (t // (60 * 24 * 31 * 12))
        dow = (t // (60 * 24)) % 7  # Day of week
        doy = (t // (60 * 24)) % 365 + 1  # Day of year
        return [year, month, day, dow, doy, hours, minutes]

    # Generate full sequences
    total_length = context_length + pred_length
    time_features = np.array([get_time_features(t) for t in range(total_length)])

    # Batch storage
    batch_ts = []
    batch_history = []
    batch_target_dates = []
    batch_target = []

    for i in range(batch_size):
        # Generate sine wave
        t = np.arange(total_length)
        sine = amplitudes[i] * np.sin(2 * np.pi * frequencies[i] * t + phases[i])
        noise = np.random.normal(0, noises[i], total_length)
        series = sine + noise

        # Split into context and prediction
        history = series[:context_length]
        target = series[context_length:]

        # Store time features
        batch_ts.append(time_features[:context_length])
        batch_target_dates.append(time_features[context_length:])
        batch_history.append(history)
        batch_target.append(target)

    # Convert to tensors and create proper dictionary structure
    return {
        "ts": torch.FloatTensor(np.array(batch_ts)),
        "history": torch.FloatTensor(np.array(batch_history)).unsqueeze(-1),
        "target_dates": torch.FloatTensor(np.array(batch_target_dates)),
        "task": torch.zeros((batch_size, pred_length), dtype=torch.long),
        "target_values": torch.FloatTensor(np.array(batch_target)).unsqueeze(-1),
        "complete_target": torch.FloatTensor(np.array(batch_target)).unsqueeze(-1),
    }
