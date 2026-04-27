"""
Usage examples:
    # Plot all curves from one or more JSON files
    python Plot.py --files dqn_variants_results.json all_results.json

    # Select specific labels from each file
    python Plot.py --files dqn_variants_results.json all_results.json \
                   --labels "Replay+Target DQN" REINFORCE AC A2C_MC

    # Custom title and output filename
    python Plot.py --files dqn_variants_results.json all_results.json \
                   --title "DQN vs PG Methods" --out comparison.png

    # Apply smoothing
    python Plot.py --files dqn_variants_results.json all_results.json --smooth 11
"""

import json
import argparse
import numpy as np
from Helpers import LearningCurvePlot, smooth


def load_json(path):
    with open(path) as f:
        return json.load(f)


def plot_from_jsons(json_paths, selected_labels=None, title="Comparison", out="comparison.png",
                    smoothing_window=None, ylim=(0, 520), optimal_line=500):

    plot = LearningCurvePlot(title=title)
    plot.set_ylim(*ylim)

    for path in json_paths:
        entries = load_json(path)
        for entry in entries:
            label = entry["label"]

            if selected_labels and label not in selected_labels:
                continue

            params = entry["params"]
            timesteps = list(range(
                params["evaluate_every"],
                params["total_steps"] + 1,
                params["evaluate_every"],
            ))

            mean = np.array(entry["mean_returns"])
            std = np.array(entry["std_returns"])

            # trim timesteps to match curve length (in case of early stopping)
            timesteps = timesteps[:len(mean)]

            if smoothing_window is not None and len(mean) > smoothing_window:
                mean = smooth(mean, smoothing_window)
                std = smooth(std, smoothing_window)

            plot.add_curve(timesteps, mean, std=std, label=label)

    if optimal_line is not None:
        plot.add_hline(optimal_line, label=f"Optimal ({optimal_line})")

    plot.save(out)
    print(f"Saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot learning curves from experiment JSON files.")
    parser.add_argument("--files", nargs="+", required=True,
                        help="One or more result JSON files to load.")
    parser.add_argument("--labels", nargs="*", default=None,
                        help="Subset of labels to plot (default: all).")
    parser.add_argument("--title", default="Learning Curves", help="Plot title.")
    parser.add_argument("--out", default="comparison.png", help="Output filename.")
    parser.add_argument("--smooth", type=int, default=None, dest="smooth",
                        help="Savitzky-Golay smoothing window size (odd int, e.g. 11).")
    parser.add_argument("--ylim", nargs=2, type=float, default=[0, 520],
                        metavar=("LOWER", "UPPER"), help="Y-axis limits.")
    args = parser.parse_args()

    plot_from_jsons(
        json_paths=args.files,
        selected_labels=args.labels,
        title=args.title,
        out=args.out,
        smoothing_window=args.smooth,
        ylim=tuple(args.ylim),
    )
    
