import os
import json
import random
from typing import Dict, List, Tuple

import numpy as np


def collect_per_stock_values(run_dir: str, window: int = 60, sentiment: str = "positive") -> Tuple[List[float], List[float]]:
    aar_vals: List[float] = []
    caar_vals: List[float] = []
    for name in os.listdir(run_dir):
        ticker_dir = os.path.join(run_dir, name)
        event_file = os.path.join(ticker_dir, "event_study", f"{name}_event_study_results.json")
        if not os.path.exists(event_file):
            continue
        try:
            with open(event_file, "r") as f:
                data = json.load(f)
            node = data.get("aar_caar", {}).get(sentiment, {})
            aar_map = node.get("AAR", {})
            caar_map = node.get("CAAR", {})
            w = str(window)
            if w in aar_map and aar_map[w] is not None:
                aar_vals.append(float(aar_map[w]))
            if w in caar_map and caar_map[w] is not None:
                caar_vals.append(float(caar_map[w]))
        except Exception:
            continue
    return aar_vals, caar_vals


def bootstrap_ci(values: List[float], n_boot: int = 20000, alpha: float = 0.05, seed: int = 42) -> Tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=float)
    means = []
    n = arr.size
    for _ in range(n_boot):
        sample = arr[rng.integers(0, n, size=n, endpoint=False)]
        means.append(sample.mean())
    lower = float(np.percentile(means, 100 * (alpha / 2)))
    upper = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return float(arr.mean()), lower, upper


def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python compute_portfolio_bootstrap_ci.py <run_dir> [window] [sentiment]")
        sys.exit(1)
    run_dir = sys.argv[1]
    window = int(sys.argv[2]) if len(sys.argv) >= 3 else 60
    sentiment = sys.argv[3] if len(sys.argv) >= 4 else "positive"

    aar_vals, caar_vals = collect_per_stock_values(run_dir, window=window, sentiment=sentiment)
    aar_mean, aar_lo, aar_hi = bootstrap_ci(aar_vals)
    caar_mean, caar_lo, caar_hi = bootstrap_ci(caar_vals)

    to_pct = lambda x: x * 100 if not np.isnan(x) else float("nan")

    result: Dict[str, Dict[str, float]] = {
        "config": {"window": window, "sentiment": sentiment, "n_stocks": len(aar_vals)},
        "AAR": {
            "mean_pct": to_pct(aar_mean),
            "ci95_low_pct": to_pct(aar_lo),
            "ci95_high_pct": to_pct(aar_hi),
        },
        "CAAR": {
            "mean_pct": to_pct(caar_mean),
            "ci95_low_pct": to_pct(caar_lo),
            "ci95_high_pct": to_pct(caar_hi),
        },
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()



