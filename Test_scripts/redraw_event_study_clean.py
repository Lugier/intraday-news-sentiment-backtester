import json
import os
import sys
from typing import Dict, List

import matplotlib.pyplot as plt


def load_timeline_json(json_path: str) -> Dict:
    with open(json_path, "r") as f:
        return json.load(f)


def build_series(data: Dict, key: str, sentiments: List[str], windows: List[int]) -> Dict[str, List[float]]:
    series: Dict[str, List[float]] = {}
    for s in sentiments:
        values: List[float] = []
        sub = data.get(key, {}).get(s, {})
        for w in windows:
            v = sub.get(str(w))
            values.append(float(v) * 100 if v is not None else 0.0)
        series[s] = values
    return series


def plot_aar_only(timeline: Dict, out_path: str) -> None:
    sentiments = ["positive", "negative", "neutral"]
    colors = {
        "positive": "#2E7D32",  # green
        "negative": "#C62828",  # red
        "neutral": "#1565C0",   # blue
    }

    windows = timeline.get("available_windows") or [5, 15, 30, 60]
    windows = sorted([int(w) for w in windows])

    aar = build_series(timeline, "aar_means", sentiments, windows)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

    # AAR
    for i, s in enumerate(sentiments):
        ax1.plot(windows, aar[s], marker="o", linewidth=2, color=colors[s], label=f"{s.capitalize()}")
        for x, y in zip(windows, aar[s]):
            ax1.annotate(f"{y:.3f}%", (x, y), textcoords="offset points", xytext=(0, 6 + i * 2), ha="center", fontsize=8, color=colors[s])
    ax1.axhline(0, color="#666", linewidth=1, alpha=0.4)
    ax1.set_ylabel("AAR (%)")
    ax1.set_xlabel("Minuten nach der Nachricht")
    ax1.set_title("Durchschnittliche abnormale Renditen (AAR)")
    ax1.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_caar_only(timeline: Dict, out_path: str) -> None:
    sentiments = ["positive", "negative", "neutral"]
    colors = {
        "positive": "#2E7D32",
        "negative": "#C62828",
        "neutral": "#1565C0",
    }

    windows = timeline.get("available_windows") or [5, 15, 30, 60]
    windows = sorted([int(w) for w in windows])

    caar = build_series(timeline, "caar_means", sentiments, windows)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for i, s in enumerate(sentiments):
        ax.plot(windows, caar[s], marker="o", linewidth=2, color=colors[s], label=f"{s.capitalize()}")
        for x, y in zip(windows, caar[s]):
            ax.annotate(f"{y:.3f}%", (x, y), textcoords="offset points", xytext=(0, 6 + i * 2), ha="center", fontsize=8, color=colors[s])
    ax.axhline(0, color="#666", linewidth=1, alpha=0.4)
    ax.set_ylabel("CAAR (%)")
    ax.set_xlabel("Minuten nach der Nachricht")
    ax.set_title("Kumulierte durchschnittliche abnormale Renditen (CAAR)")
    ax.legend(frameon=False)

    ax.set_xticks(windows)
    ax.set_xticklabels([f"{w} Min." for w in windows])

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if len(sys.argv) < 2:
        print("Nutzung: python redraw_event_study_clean.py <timeline_json_path> [output_png_path] [titel_prefix]")
        sys.exit(1)

    json_path = sys.argv[1]
    if not os.path.exists(json_path):
        print(f"Datei nicht gefunden: {json_path}")
        sys.exit(1)

    output_path = (
        sys.argv[2]
        if len(sys.argv) >= 3
        else os.path.join(os.path.dirname(json_path), "LOG_portfolio_event_study_timeline_clean.png")
    )
    title_prefix = sys.argv[3] if len(sys.argv) >= 4 else ""

    timeline = load_timeline_json(json_path)

    # Basis-Pfad ableiten und zwei separate Grafiken schreiben
    base, ext = os.path.splitext(output_path)
    aar_path = f"{base}_AAR.png"
    caar_path = f"{base}_CAAR.png"

    plot_aar_only(timeline, aar_path)
    plot_caar_only(timeline, caar_path)
    print(f"Gespeichert: {aar_path}\nGespeichert: {caar_path}")


if __name__ == "__main__":
    main()


