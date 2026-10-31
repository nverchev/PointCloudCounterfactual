#!/usr/bin/env bash
set -euo pipefail

for config in modelnet_dt modelnet_bbcv; do
    uv run train_classifier.py --config-name "$config" final=true
    uv run evaluate_classifier.py --config-name "$config" final=true
    uv run train_autoencoder.py --config-name "$config" final=true
    uv run evaluate_autoencoder.py --config-name "$config" final=true
    uv run train_w_autoencoder.py --config-name "$config" final=true
    uv run evaluate_w_autoencoder.py --config-name "$config" final=true
    uv run evaluate_counterfactuals.py --config-name "$config" final=true
    uv run visualize_counterfactuals.py --config-name "$config" final=true
done
