import re
from collections import Counter

import matplotlib.pyplot as plt


def parse_dataset_info(text):
    """Parse the dataset information from the text file."""
    context_lengths = []
    forecast_lengths = []

    # Split by terms
    sections = re.split(r"(LONG TERM:|MEDIUM TERM:|SHORT TERM:)", text)

    for i, section in enumerate(sections):
        if "Context lengths" in section and "Forecast lengths" in section:
            # Extract context lengths
            context_matches = re.findall(
                r"Context lengths \([^)]+\): \[([^\]]+)\]", section
            )
            for match in context_matches:
                lengths = [int(x.strip()) for x in match.split(",")]
                context_lengths.extend(lengths)

            # Extract forecast lengths
            forecast_matches = re.findall(
                r"Forecast lengths \([^)]+\): \[([^\]]+)\]", section
            )
            for match in forecast_matches:
                lengths = [int(x.strip()) for x in match.split(",")]
                forecast_lengths.extend(lengths)

    return context_lengths, forecast_lengths


def visualize_lengths(context_lengths, forecast_lengths, save_plots=True):
    """Create and save bar plots for context and forecast lengths."""

    # Get unique lengths and their counts
    unique_context = Counter(context_lengths)
    unique_forecast = Counter(forecast_lengths)

    # Sort by length value
    context_sorted = sorted(unique_context.items())
    forecast_sorted = sorted(unique_forecast.items())

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 14))

    # Plot 1: Context Lengths (Log scale for better visualization)
    context_vals, context_counts = zip(*context_sorted) if context_sorted else ([], [])
    bars1 = ax1.bar(
        range(len(context_vals)), context_counts, alpha=0.7, color="steelblue"
    )
    ax1.set_title(
        "Distribution of Unique Context Lengths Across All Datasets",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax1.set_xlabel("Context Length", fontsize=14)
    ax1.set_ylabel("Frequency", fontsize=14)

    # Use log scale for x-axis labels to handle wide range
    ax1.set_xticks(range(len(context_vals)))

    # Create smart labeling - show every nth label based on total count
    n_labels = len(context_vals)
    if n_labels > 20:
        step = max(1, n_labels // 15)  # Show ~15 labels max
        indices_to_show = list(range(0, n_labels, step)) + [
            n_labels - 1
        ]  # Always show first and last
    else:
        indices_to_show = list(range(n_labels))

    # Set labels only for selected indices
    labels = [""] * n_labels
    for i in indices_to_show:
        if context_vals[i] >= 1000:
            labels[i] = (
                f"{context_vals[i] / 1000:.1f}K"  # Convert to K format for readability
            )
        else:
            labels[i] = str(context_vals[i])

    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax1.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars (only for higher bars to avoid clutter)
    max_count = max(context_counts) if context_counts else 0
    for i, (bar, count) in enumerate(zip(bars1, context_counts)):
        if (
            count >= max_count * 0.3
        ):  # Only label bars that are at least 30% of max height
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.05,
                f"{count}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    # Plot 2: Forecast Lengths (Linear scale - smaller range)
    forecast_vals, forecast_counts = (
        zip(*forecast_sorted) if forecast_sorted else ([], [])
    )
    bars2 = ax2.bar(
        range(len(forecast_vals)), forecast_counts, alpha=0.7, color="coral"
    )
    ax2.set_title(
        "Distribution of Unique Forecast Lengths Across All Datasets",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax2.set_xlabel("Forecast Length", fontsize=14)
    ax2.set_ylabel("Frequency", fontsize=14)

    # For forecast lengths (smaller range), show all labels
    ax2.set_xticks(range(len(forecast_vals)))
    ax2.set_xticklabels(
        [str(v) for v in forecast_vals], rotation=45, ha="right", fontsize=11
    )
    ax2.grid(True, alpha=0.3, axis="y")

    # Add value labels on all forecast bars
    for bar, count in zip(bars2, forecast_counts):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.05,
            f"{count}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Add summary text boxes
    context_summary = f"Range: {min(context_vals):,} - {max(context_vals):,}\nTotal unique: {len(unique_context)}"
    forecast_summary = f"Range: {min(forecast_vals)} - {max(forecast_vals)}\nTotal unique: {len(unique_forecast)}"

    ax1.text(
        0.02,
        0.98,
        context_summary,
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
    )
    ax2.text(
        0.02,
        0.98,
        forecast_summary,
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.8),
    )

    plt.tight_layout()

    if save_plots:
        plt.savefig(
            "outputs/plots/dataset_lengths_distribution.png",
            dpi=300,
            bbox_inches="tight",
        )
        print("Plot saved as 'dataset_lengths_distribution.png'")

    plt.show()

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    print(f"\nCONTEXT LENGTHS:")
    print(f"Total unique context lengths: {len(unique_context)}")
    print(f"Range: {min(context_lengths):,} - {max(context_lengths):,}")
    print(f"Most common context lengths:")
    for length, count in unique_context.most_common(10):
        print(f"  {length:,}: {count} occurrences")

    print(f"\nFORECAST LENGTHS:")
    print(f"Total unique forecast lengths: {len(unique_forecast)}")
    print(f"Range: {min(forecast_lengths)} - {max(forecast_lengths)}")
    print(f"Most common forecast lengths:")
    for length, count in unique_forecast.most_common(10):
        print(f"  {length}: {count} occurrences")

    return unique_context, unique_forecast


# Example usage with the provided data
dataset_text = """
LONG TERM: 31 datasets. Unique frequencies: 10S, 10T, 15T, 5T, D, H, M, W-SUN, W-THU, W-TUE, W-WED
  - LOOP_SEATTLE/5T (freq: 5T)
    Context lengths (val): [94320]
    Context lengths (test): [94320]
    Forecast lengths (val): [720]
    Forecast lengths (test): [720]
  - LOOP_SEATTLE/H (freq: H)
    Context lengths (val): [7320]
    Context lengths (test): [7320]
    Forecast lengths (val): [720]
    Forecast lengths (test): [720]
  - M_DENSE/H (freq: H)
    Context lengths (val): [15360]
    Context lengths (test): [15360]
    Forecast lengths (val): [720]
    Forecast lengths (test): [720]
  - SZ_TAXI/15T (freq: 15T)
    Context lengths (val): [2256]
    Context lengths (test): [2256]
    Forecast lengths (val): [720]
    Forecast lengths (test): [720]
  - bitbrains_fast_storage/5T (freq: 5T)
    Context lengths (val): [7200]
    Context lengths (test): [7200]
    Forecast lengths (val): [720]
    Forecast lengths (test): [720]
  - bitbrains_rnd/5T (freq: 5T)
    Context lengths (val): [7200]
    Context lengths (test): [7200]
    Forecast lengths (val): [720]
    Forecast lengths (test): [720]
  - bizitobs_application (freq: 10S)
    Context lengths (val): [7934]
    Context lengths (test): [7934]
    Forecast lengths (val): [900]
    Forecast lengths (test): [900]
  - bizitobs_l2c/5T (freq: 5T)
    Context lengths (val): [28368]
    Context lengths (test): [28368]
    Forecast lengths (val): [720]
    Forecast lengths (test): [720]
  - bizitobs_l2c/H (freq: H)
    Context lengths (val): [1944]
    Context lengths (test): [1944]
    Forecast lengths (val): [720]
    Forecast lengths (test): [720]
  - bizitobs_service (freq: 10S)
    Context lengths (val): [7935]
    Context lengths (test): [7935]
    Forecast lengths (val): [900]
    Forecast lengths (test): [900]
  - electricity/15T (freq: 15T)
    Context lengths (val): [125856]
    Context lengths (test): [125856]
    Forecast lengths (val): [720]
    Forecast lengths (test): [720]
  - electricity/D (freq: D)
    Context lengths (val): [1011]
    Context lengths (test): [1011]
    Forecast lengths (val): [450]
    Forecast lengths (test): [450]
  - electricity/H (freq: H)
    Context lengths (val): [31464]
    Context lengths (test): [31464]
    Forecast lengths (val): [720]
    Forecast lengths (test): [720]
  - ett1/15T (freq: 15T)
    Context lengths (val): [62480]
    Context lengths (test): [62480]
    Forecast lengths (val): [720]
    Forecast lengths (test): [720]
  - ett1/H (freq: H)
    Context lengths (val): [15260]
    Context lengths (test): [15260]
    Forecast lengths (val): [720]
    Forecast lengths (test): [720]
  - ett2/15T (freq: 15T)
    Context lengths (val): [62480]
    Context lengths (test): [62480]
    Forecast lengths (val): [720]
    Forecast lengths (test): [720]
  - ett2/H (freq: H)
    Context lengths (val): [15260]
    Context lengths (test): [15260]
    Forecast lengths (val): [720]
    Forecast lengths (test): [720]
  - hierarchical_sales/D (freq: D)
    Context lengths (val): [1375]
    Context lengths (test): [1375]
    Forecast lengths (val): [450]
    Forecast lengths (test): [450]
  - hierarchical_sales/W (freq: W-WED)
    Context lengths (val): [140]
    Context lengths (test): [140]
    Forecast lengths (val): [120]
    Forecast lengths (test): [120]
  - jena_weather/10T (freq: 10T)
    Context lengths (val): [46944]
    Context lengths (test): [46944]
    Forecast lengths (val): [720]
    Forecast lengths (test): [720]
  - jena_weather/H (freq: H)
    Context lengths (val): [7344]
    Context lengths (test): [7344]
    Forecast lengths (val): [720]
    Forecast lengths (test): [720]
  - kdd_cup_2018_with_missing/H (freq: H)
    Context lengths (val): [9458]
    Context lengths (test): [9458]
    Forecast lengths (val): [720]
    Forecast lengths (test): [720]
  - m4_daily (freq: D)
    Context lengths (val): [810]
    Context lengths (test): [810]
    Forecast lengths (val): [210]
    Forecast lengths (test): [210]
  - m4_weekly (freq: W-SUN)
    Context lengths (val): [1997]
    Context lengths (test): [1997]
    Forecast lengths (val): [195]
    Forecast lengths (test): [195]
  - saugeenday/D (freq: D)
    Context lengths (val): [21041]
    Context lengths (test): [21041]
    Forecast lengths (val): [450]
    Forecast lengths (test): [450]
  - saugeenday/M (freq: M)
    Context lengths (val): [600]
    Context lengths (test): [600]
    Forecast lengths (val): [180]
    Forecast lengths (test): [180]
  - saugeenday/W (freq: W-THU)
    Context lengths (val): [3031]
    Context lengths (test): [3031]
    Forecast lengths (val): [120]
    Forecast lengths (test): [120]
  - solar/10T (freq: 10T)
    Context lengths (val): [46800]
    Context lengths (test): [46800]
    Forecast lengths (val): [720]
    Forecast lengths (test): [720]
  - solar/H (freq: H)
    Context lengths (val): [7320]
    Context lengths (test): [7320]
    Forecast lengths (val): [720]
    Forecast lengths (test): [720]
  - us_births/D (freq: D)
    Context lengths (val): [6405]
    Context lengths (test): [6405]
    Forecast lengths (val): [450]
    Forecast lengths (test): [450]
  - us_births/W (freq: W-TUE)
    Context lengths (val): [923]
    Context lengths (test): [923]
    Forecast lengths (val): [120]
    Forecast lengths (test): [120]

MEDIUM TERM: 38 datasets. Unique frequencies: 10S, 10T, 15T, 5T, D, H, M, W-FRI, W-SUN, W-THU, W-TUE, W-WED
  - LOOP_SEATTLE/5T (freq: 5T)
    Context lengths (val): [95520]
    Context lengths (test): [95520]
    Forecast lengths (val): [480]
    Forecast lengths (test): [480]
  - LOOP_SEATTLE/H (freq: H)
    Context lengths (val): [7800]
    Context lengths (test): [7800]
    Forecast lengths (val): [480]
    Forecast lengths (test): [480]
  - M_DENSE/D (freq: D)
    Context lengths (val): [430]
    Context lengths (test): [430]
    Forecast lengths (val): [300]
    Forecast lengths (test): [300]
  - M_DENSE/H (freq: H)
    Context lengths (val): [15600]
    Context lengths (test): [15600]
    Forecast lengths (val): [480]
    Forecast lengths (test): [480]
  - SZ_TAXI/15T (freq: 15T)
    Context lengths (val): [2496]
    Context lengths (test): [2496]
    Forecast lengths (val): [480]
    Forecast lengths (test): [480]
  - bitbrains_fast_storage/5T (freq: 5T)
    Context lengths (val): [7680]
    Context lengths (test): [7680]
    Forecast lengths (val): [480]
    Forecast lengths (test): [480]
  - bitbrains_rnd/5T (freq: 5T)
    Context lengths (val): [7680]
    Context lengths (test): [7680]
    Forecast lengths (val): [480]
    Forecast lengths (test): [480]
  - bizitobs_application (freq: 10S)
    Context lengths (val): [7634]
    Context lengths (test): [7634]
    Forecast lengths (val): [600]
    Forecast lengths (test): [600]
  - bizitobs_l2c/5T (freq: 5T)
    Context lengths (val): [28608]
    Context lengths (test): [28608]
    Forecast lengths (val): [480]
    Forecast lengths (test): [480]
  - bizitobs_l2c/H (freq: H)
    Context lengths (val): [2184]
    Context lengths (test): [2184]
    Forecast lengths (val): [480]
    Forecast lengths (test): [480]
  - bizitobs_service (freq: 10S)
    Context lengths (val): [7635]
    Context lengths (test): [7635]
    Forecast lengths (val): [600]
    Forecast lengths (test): [600]
  - electricity/15T (freq: 15T)
    Context lengths (val): [130656]
    Context lengths (test): [130656]
    Forecast lengths (val): [480]
    Forecast lengths (test): [480]
  - electricity/D (freq: D)
    Context lengths (val): [1161]
    Context lengths (test): [1161]
    Forecast lengths (val): [300]
    Forecast lengths (test): [300]
  - electricity/H (freq: H)
    Context lengths (val): [31224]
    Context lengths (test): [31224]
    Forecast lengths (val): [480]
    Forecast lengths (test): [480]
  - electricity/W (freq: W-FRI)
    Context lengths (val): [128]
    Context lengths (test): [128]
    Forecast lengths (val): [80]
    Forecast lengths (test): [80]
  - ett1/15T (freq: 15T)
    Context lengths (val): [62480]
    Context lengths (test): [62480]
    Forecast lengths (val): [480]
    Forecast lengths (test): [480]
  - ett1/D (freq: D)
    Context lengths (val): [425]
    Context lengths (test): [425]
    Forecast lengths (val): [300]
    Forecast lengths (test): [300]
  - ett1/H (freq: H)
    Context lengths (val): [15500]
    Context lengths (test): [15500]
    Forecast lengths (val): [480]
    Forecast lengths (test): [480]
  - ett2/15T (freq: 15T)
    Context lengths (val): [62480]
    Context lengths (test): [62480]
    Forecast lengths (val): [480]
    Forecast lengths (test): [480]
  - ett2/D (freq: D)
    Context lengths (val): [425]
    Context lengths (test): [425]
    Forecast lengths (val): [300]
    Forecast lengths (test): [300]
  - ett2/H (freq: H)
    Context lengths (val): [15500]
    Context lengths (test): [15500]
    Forecast lengths (val): [480]
    Forecast lengths (test): [480]
  - hierarchical_sales/D (freq: D)
    Context lengths (val): [1525]
    Context lengths (test): [1525]
    Forecast lengths (val): [300]
    Forecast lengths (test): [300]
  - hierarchical_sales/W (freq: W-WED)
    Context lengths (val): [180]
    Context lengths (test): [180]
    Forecast lengths (val): [80]
    Forecast lengths (test): [80]
  - jena_weather/10T (freq: 10T)
    Context lengths (val): [47424]
    Context lengths (test): [47424]
    Forecast lengths (val): [480]
    Forecast lengths (test): [480]
  - jena_weather/H (freq: H)
    Context lengths (val): [7824]
    Context lengths (test): [7824]
    Forecast lengths (val): [480]
    Forecast lengths (test): [480]
  - kdd_cup_2018_with_missing/H (freq: H)
    Context lengths (val): [9938]
    Context lengths (test): [9938]
    Forecast lengths (val): [480]
    Forecast lengths (test): [480]
  - m4_daily (freq: D)
    Context lengths (val): [880]
    Context lengths (test): [880]
    Forecast lengths (val): [140]
    Forecast lengths (test): [140]
  - m4_monthly (freq: M)
    Context lengths (val): [307]
    Context lengths (test): [307]
    Forecast lengths (val): [180]
    Forecast lengths (test): [180]
  - m4_weekly (freq: W-SUN)
    Context lengths (val): [2062]
    Context lengths (test): [2062]
    Forecast lengths (val): [130]
    Forecast lengths (test): [130]
  - saugeenday/D (freq: D)
    Context lengths (val): [21341]
    Context lengths (test): [21341]
    Forecast lengths (val): [300]
    Forecast lengths (test): [300]
  - saugeenday/M (freq: M)
    Context lengths (val): [660]
    Context lengths (test): [660]
    Forecast lengths (val): [120]
    Forecast lengths (test): [120]
  - saugeenday/W (freq: W-THU)
    Context lengths (val): [2991]
    Context lengths (test): [2991]
    Forecast lengths (val): [80]
    Forecast lengths (test): [80]
  - solar/10T (freq: 10T)
    Context lengths (val): [47280]
    Context lengths (test): [47280]
    Forecast lengths (val): [480]
    Forecast lengths (test): [480]
  - solar/H (freq: H)
    Context lengths (val): [7800]
    Context lengths (test): [7800]
    Forecast lengths (val): [480]
    Forecast lengths (test): [480]
  - temperature_rain_with_missing (freq: D)
    Context lengths (val): [425]
    Context lengths (test): [425]
    Forecast lengths (val): [300]
    Forecast lengths (test): [300]
  - us_births/D (freq: D)
    Context lengths (val): [6405]
    Context lengths (test): [6405]
    Forecast lengths (val): [300]
    Forecast lengths (test): [300]
  - us_births/M (freq: M)
    Context lengths (val): [120]
    Context lengths (test): [120]
    Forecast lengths (val): [120]
    Forecast lengths (test): [120]
  - us_births/W (freq: W-TUE)
    Context lengths (val): [883]
    Context lengths (test): [883]
    Forecast lengths (val): [80]
    Forecast lengths (test): [80]

SHORT TERM: 55 datasets. Unique frequencies: 10S, 10T, 15T, 5T, A-DEC, D, H, M, Q-DEC, W-FRI, W-SUN, W-THU, W-TUE, W-WED
  - LOOP_SEATTLE/5T (freq: 5T)
    Context lengths (val): [104160]
    Context lengths (test): [104160]
    Forecast lengths (val): [48]
    Forecast lengths (test): [48]
  - LOOP_SEATTLE/D (freq: D)
    Context lengths (val): [305]
    Context lengths (test): [305]
    Forecast lengths (val): [30]
    Forecast lengths (test): [30]
  - LOOP_SEATTLE/H (freq: H)
    Context lengths (val): [7848]
    Context lengths (test): [7848]
    Forecast lengths (val): [48]
    Forecast lengths (test): [48]
  - M_DENSE/D (freq: D)
    Context lengths (val): [640]
    Context lengths (test): [640]
    Forecast lengths (val): [30]
    Forecast lengths (test): [30]
  - M_DENSE/H (freq: H)
    Context lengths (val): [16560]
    Context lengths (test): [16560]
    Forecast lengths (val): [48]
    Forecast lengths (test): [48]
  - SZ_TAXI/15T (freq: 15T)
    Context lengths (val): [2640]
    Context lengths (test): [2640]
    Forecast lengths (val): [48]
    Forecast lengths (test): [48]
  - SZ_TAXI/H (freq: H)
    Context lengths (val): [648]
    Context lengths (test): [648]
    Forecast lengths (val): [48]
    Forecast lengths (test): [48]
  - bitbrains_fast_storage/5T (freq: 5T)
    Context lengths (val): [7776]
    Context lengths (test): [7776]
    Forecast lengths (val): [48]
    Forecast lengths (test): [48]
  - bitbrains_fast_storage/H (freq: H)
    Context lengths (val): [625]
    Context lengths (test): [625]
    Forecast lengths (val): [48]
    Forecast lengths (test): [48]
  - bitbrains_rnd/5T (freq: 5T)
    Context lengths (val): [7776]
    Context lengths (test): [7776]
    Forecast lengths (val): [48]
    Forecast lengths (test): [48]
  - bitbrains_rnd/H (freq: H)
    Context lengths (val): [624]
    Context lengths (test): [624]
    Forecast lengths (val): [48]
    Forecast lengths (test): [48]
  - bizitobs_application (freq: 10S)
    Context lengths (val): [7934]
    Context lengths (test): [7934]
    Forecast lengths (val): [60]
    Forecast lengths (test): [60]
  - bizitobs_l2c/5T (freq: 5T)
    Context lengths (val): [31008]
    Context lengths (test): [31008]
    Forecast lengths (val): [48]
    Forecast lengths (test): [48]
  - bizitobs_l2c/H (freq: H)
    Context lengths (val): [2376]
    Context lengths (test): [2376]
    Forecast lengths (val): [48]
    Forecast lengths (test): [48]
  - bizitobs_service (freq: 10S)
    Context lengths (val): [7935]
    Context lengths (test): [7935]
    Forecast lengths (val): [60]
    Forecast lengths (test): [60]
  - car_parts_with_missing (freq: M)
    Context lengths (val): [39]
    Context lengths (test): [39]
    Forecast lengths (val): [12]
    Forecast lengths (test): [12]
  - covid_deaths (freq: D)
    Context lengths (val): [182]
    Context lengths (test): [182]
    Forecast lengths (val): [30]
    Forecast lengths (test): [30]
  - electricity/15T (freq: 15T)
    Context lengths (val): [139296]
    Context lengths (test): [139296]
    Forecast lengths (val): [48]
    Forecast lengths (test): [48]
  - electricity/D (freq: D)
    Context lengths (val): [1311]
    Context lengths (test): [1311]
    Forecast lengths (val): [30]
    Forecast lengths (test): [30]
  - electricity/H (freq: H)
    Context lengths (val): [34104]
    Context lengths (test): [34104]
    Forecast lengths (val): [48]
    Forecast lengths (test): [48]
  - electricity/W (freq: W-FRI)
    Context lengths (val): [184]
    Context lengths (test): [184]
    Forecast lengths (val): [8]
    Forecast lengths (test): [8]
  - ett1/15T (freq: 15T)
    Context lengths (val): [68720]
    Context lengths (test): [68720]
    Forecast lengths (val): [48]
    Forecast lengths (test): [48]
  - ett1/D (freq: D)
    Context lengths (val): [635]
    Context lengths (test): [635]
    Forecast lengths (val): [30]
    Forecast lengths (test): [30]
  - ett1/H (freq: H)
    Context lengths (val): [16460]
    Context lengths (test): [16460]
    Forecast lengths (val): [48]
    Forecast lengths (test): [48]
  - ett1/W (freq: W-THU)
    Context lengths (val): [87]
    Context lengths (test): [87]
    Forecast lengths (val): [8]
    Forecast lengths (test): [8]
  - ett2/15T (freq: 15T)
    Context lengths (val): [68720]
    Context lengths (test): [68720]
    Forecast lengths (val): [48]
    Forecast lengths (test): [48]
  - ett2/D (freq: D)
    Context lengths (val): [635]
    Context lengths (test): [635]
    Forecast lengths (val): [30]
    Forecast lengths (test): [30]
  - ett2/H (freq: H)
    Context lengths (val): [16460]
    Context lengths (test): [16460]
    Forecast lengths (val): [48]
    Forecast lengths (test): [48]
  - ett2/W (freq: W-THU)
    Context lengths (val): [87]
    Context lengths (test): [87]
    Forecast lengths (val): [8]
    Forecast lengths (test): [8]
  - hierarchical_sales/D (freq: D)
    Context lengths (val): [1615]
    Context lengths (test): [1615]
    Forecast lengths (val): [30]
    Forecast lengths (test): [30]
  - hierarchical_sales/W (freq: W-WED)
    Context lengths (val): [228]
    Context lengths (test): [228]
    Forecast lengths (val): [8]
    Forecast lengths (test): [8]
  - hospital (freq: M)
    Context lengths (val): [72]
    Context lengths (test): [72]
    Forecast lengths (val): [12]
    Forecast lengths (test): [12]
  - jena_weather/10T (freq: 10T)
    Context lengths (val): [51744]
    Context lengths (test): [51744]
    Forecast lengths (val): [48]
    Forecast lengths (test): [48]
  - jena_weather/D (freq: D)
    Context lengths (val): [306]
    Context lengths (test): [306]
    Forecast lengths (val): [30]
    Forecast lengths (test): [30]
  - jena_weather/H (freq: H)
    Context lengths (val): [7872]
    Context lengths (test): [7872]
    Forecast lengths (val): [48]
    Forecast lengths (test): [48]
  - kdd_cup_2018_with_missing/D (freq: D)
    Context lengths (val): [395]
    Context lengths (test): [395]
    Forecast lengths (val): [30]
    Forecast lengths (test): [30]
  - kdd_cup_2018_with_missing/H (freq: H)
    Context lengths (val): [9938]
    Context lengths (test): [9938]
    Forecast lengths (val): [48]
    Forecast lengths (test): [48]
  - m4_daily (freq: D)
    Context lengths (val): [1006]
    Context lengths (test): [1006]
    Forecast lengths (val): [14]
    Forecast lengths (test): [14]
  - m4_hourly (freq: H)
    Context lengths (val): [700]
    Context lengths (test): [700]
    Forecast lengths (val): [48]
    Forecast lengths (test): [48]
  - m4_monthly (freq: M)
    Context lengths (val): [469]
    Context lengths (test): [469]
    Forecast lengths (val): [18]
    Forecast lengths (test): [18]
  - m4_quarterly (freq: Q-DEC)
    Context lengths (val): [25]
    Context lengths (test): [25]
    Forecast lengths (val): [8]
    Forecast lengths (test): [8]
  - m4_weekly (freq: W-SUN)
    Context lengths (val): [2179]
    Context lengths (test): [2179]
    Forecast lengths (val): [13]
    Forecast lengths (test): [13]
  - m4_yearly (freq: A-DEC)
    Context lengths (val): [31]
    Context lengths (test): [31]
    Forecast lengths (val): [6]
    Forecast lengths (test): [6]
  - restaurant (freq: D)
    Context lengths (val): [436]
    Context lengths (test): [436]
    Forecast lengths (val): [30]
    Forecast lengths (test): [30]
  - saugeenday/D (freq: D)
    Context lengths (val): [23141]
    Context lengths (test): [23141]
    Forecast lengths (val): [30]
    Forecast lengths (test): [30]
  - saugeenday/M (freq: M)
    Context lengths (val): [696]
    Context lengths (test): [696]
    Forecast lengths (val): [12]
    Forecast lengths (test): [12]
  - saugeenday/W (freq: W-THU)
    Context lengths (val): [3231]
    Context lengths (test): [3231]
    Forecast lengths (val): [8]
    Forecast lengths (test): [8]
  - solar/10T (freq: 10T)
    Context lengths (val): [51600]
    Context lengths (test): [51600]
    Forecast lengths (val): [48]
    Forecast lengths (test): [48]
  - solar/D (freq: D)
    Context lengths (val): [305]
    Context lengths (test): [305]
    Forecast lengths (val): [30]
    Forecast lengths (test): [30]
  - solar/H (freq: H)
    Context lengths (val): [7848]
    Context lengths (test): [7848]
    Forecast lengths (val): [48]
    Forecast lengths (test): [48]
  - solar/W (freq: W-FRI)
    Context lengths (val): [44]
    Context lengths (test): [44]
    Forecast lengths (val): [8]
    Forecast lengths (test): [8]
  - temperature_rain_with_missing (freq: D)
    Context lengths (val): [635]
    Context lengths (test): [635]
    Forecast lengths (val): [30]
    Forecast lengths (test): [30]
  - us_births/D (freq: D)
    Context lengths (val): [6705]
    Context lengths (test): [6705]
    Forecast lengths (val): [30]
    Forecast lengths (test): [30]
  - us_births/M (freq: M)
    Context lengths (val): [216]
    Context lengths (test): [216]
    Forecast lengths (val): [12]
    Forecast lengths (test): [12]
  - us_births/W (freq: W-TUE)
    Context lengths (val): [931]
    Context lengths (test): [931]
    Forecast lengths (val): [8]
    Forecast lengths (test): [8]
"""

# Parse the data and create visualizations
context_lengths, forecast_lengths = parse_dataset_info(dataset_text)
unique_context, unique_forecast = visualize_lengths(context_lengths, forecast_lengths)

print(
    f"\nTotal datasets analyzed: {len(context_lengths) // 2}"
)  # Divide by 2 since we have val and test for each
print(f"Total context length entries: {len(context_lengths)}")
print(f"Total forecast length entries: {len(forecast_lengths)}")
