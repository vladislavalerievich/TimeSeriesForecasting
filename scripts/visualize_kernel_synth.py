import argparse
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt

from src.synthetic_generation.kernel_synth.kernel_synth import KernelSynthGenerator


def plot_series(
    series: Dict,
    title: str = "Synthetic Time Series",
    output_file: Optional[Path] = None,
) -> None:
    """
    Plot and optionally save a synthetic time series.

    Parameters
    ----------
    series : dict
        Dictionary with keys 'timestamps' and 'values'.
    title : str
        Plot title.
    output_file : Path, optional
        If provided, save plot to this file.
    """
    timestamps = series["timestamps"]
    values = series["values"]
    plt.figure(figsize=(12, 4))
    plt.plot(timestamps, values, linewidth=1.2)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-L", "--length", type=int, default=192)
    parser.add_argument("-J", "--max_kernels", type=int, default=5)
    parser.add_argument("--plot", action="store_true", help="Plot a sample series")
    parser.add_argument(
        "--save_plot",
        type=str,
        default="outputs/plots/kernel_synth.png",
        help="Path to save the plot image",
    )
    args = parser.parse_args()

    out = Path(__file__).parent / "independent-kernelsynth.arrow"
    gen = KernelSynthGenerator(length=args.length, max_kernels=args.max_kernels)

    sample = gen.generate_time_series()
    plot_series(
        sample,
        title="Sample Synthetic Series",
        output_file=Path(args.save_plot) if args.save_plot else None,
    )
