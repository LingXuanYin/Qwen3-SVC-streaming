# coding=utf-8
"""Analyze grid search results using OLS response surface modeling."""

import json
import numpy as np
from itertools import product


def load_results(path="L:/DATASET/svc_output/grid_search/results.json"):
    with open(path) as f:
        results = json.load(f)
    return [r for r in results if r["status"] == "ok"]


def fit_response_surface(results, target="loss"):
    """Fit quadratic response surface: loss = b0 + b1*x1 + b2*x2 + b3*x3 + b11*x1^2 + b22*x2^2 + b33*x3^2 + b12*x1*x2 + ...

    Features: log(lr), rank, weight (normalized)
    """
    X_raw = []
    y = []
    for r in results:
        log_lr = np.log10(r["lr"])
        rank = r["lora_rank"]
        weight = r["sub_talker_loss_weight"]
        X_raw.append([log_lr, rank, weight])
        y.append(r[target])

    X_raw = np.array(X_raw)
    y = np.array(y)

    # Normalize features
    means = X_raw.mean(axis=0)
    stds = X_raw.std(axis=0)
    stds[stds == 0] = 1
    X_norm = (X_raw - means) / stds

    # Build quadratic features: [1, x1, x2, x3, x1^2, x2^2, x3^2, x1*x2, x1*x3, x2*x3]
    n = X_norm.shape[0]
    d = X_norm.shape[1]
    features = [np.ones(n)]  # intercept
    for i in range(d):
        features.append(X_norm[:, i])
    for i in range(d):
        features.append(X_norm[:, i] ** 2)
    for i in range(d):
        for j in range(i + 1, d):
            features.append(X_norm[:, i] * X_norm[:, j])
    X = np.column_stack(features)

    # OLS: beta = (X^T X)^{-1} X^T y
    beta = np.linalg.lstsq(X, y, rcond=None)[0]

    # Predict
    y_pred = X @ beta
    residuals = y - y_pred
    r2 = 1 - np.sum(residuals**2) / np.sum((y - y.mean())**2)

    return beta, means, stds, r2


def find_optimal(beta, means, stds, grid_ranges):
    """Search for optimal hyperparameters in continuous space."""
    best_loss = float("inf")
    best_params = None

    # Dense search in normalized space
    log_lr_range = np.linspace(np.log10(grid_ranges["lr"][0]), np.log10(grid_ranges["lr"][-1]), 50)
    rank_range = np.linspace(grid_ranges["lora_rank"][0], grid_ranges["lora_rank"][-1], 20)
    weight_range = np.linspace(grid_ranges["sub_talker_loss_weight"][0], grid_ranges["sub_talker_loss_weight"][-1], 20)

    for log_lr, rank, weight in product(log_lr_range, rank_range, weight_range):
        x = np.array([log_lr, rank, weight])
        x_norm = (x - means) / stds

        features = [1.0]
        for v in x_norm:
            features.append(v)
        for v in x_norm:
            features.append(v ** 2)
        for i in range(len(x_norm)):
            for j in range(i + 1, len(x_norm)):
                features.append(x_norm[i] * x_norm[j])

        pred = np.dot(features, beta)
        if pred < best_loss:
            best_loss = pred
            best_params = {"lr": 10 ** log_lr, "lora_rank": round(rank), "sub_talker_loss_weight": round(weight, 2)}

    return best_params, best_loss


def main():
    results = load_results()
    print(f"Loaded {len(results)} successful runs\n")

    # Print raw results
    print(f"{'lr':>8} {'rank':>5} {'w_sub':>5} | {'loss':>8} {'main':>8} {'sub':>8}")
    print("-" * 60)
    for r in sorted(results, key=lambda x: x["loss"]):
        print(f"{r['lr']:>8.0e} {r['lora_rank']:>5} {r['sub_talker_loss_weight']:>5.1f} | "
              f"{r['loss']:>8.4f} {r['main']:>8.4f} {r['sub']:>8.4f}")

    # Fit response surface
    for target in ["loss", "main"]:
        beta, means, stds, r2 = fit_response_surface(results, target=target)
        print(f"\n{'='*60}")
        print(f"Response surface for '{target}': R2 = {r2:.4f}")

        grid_ranges = {
            "lr": [3e-4, 8e-4],
            "lora_rank": [16, 32],
            "sub_talker_loss_weight": [0.1, 0.3],
        }

        opt_params, opt_loss = find_optimal(beta, means, stds, grid_ranges)
        print(f"Predicted optimal {target}: {opt_loss:.4f}")
        print(f"Optimal params: lr={opt_params['lr']:.2e}, rank={opt_params['lora_rank']}, "
              f"sub_weight={opt_params['sub_talker_loss_weight']}")

    # Best from grid
    best_grid = min(results, key=lambda x: x["loss"])
    print(f"\n{'='*60}")
    print(f"Best from grid: lr={best_grid['lr']:.0e}, rank={best_grid['lora_rank']}, "
          f"weight={best_grid['sub_talker_loss_weight']}, loss={best_grid['loss']:.4f}")


if __name__ == "__main__":
    main()
