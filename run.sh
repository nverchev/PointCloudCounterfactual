#!/usr/bin/env bash
set -euo pipefail

uv run python train_classifier.py final=True
uv run python train_autoencoder.py final=True
uv run python train_w_autoencoder.py final=True
uv run python evaluate_counterfactuals.py final=True

uv run python train_classifier.py final=True data/dataset=modelnet_bottle_bowl_cup_vase
uv run python train_autoencoder.py final=True data/dataset=modelnet_bottle_bowl_cup_vase
uv run python train_w_autoencoder.py final=True data/dataset=modelnet_bottle_bowl_cup_vase
uv run python evaluate_counterfactuals.py final=True data/dataset=modelnet_bottle_bowl_cup_vase
