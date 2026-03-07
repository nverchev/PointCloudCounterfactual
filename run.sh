#!/usr/bin/env bash
set -euo pipefail

uv run train_classifier.py final=True
uv run train_autoencoder.py final=True
uv run train_flow.py final=True
uv run evaluate_counterfactuals.py final=True

uv run train_classifier.py final=True data/dataset=modelnet_bottle_bowl_cup_vase
uv run train_autoencoder.py final=True data/dataset=modelnet_bottle_bowl_cup_vase
uv run train_flow.py final=True data/dataset=modelnet_bottle_bowl_cup_vase
uv run evaluate_counterfactuals.py final=True data/dataset=modelnet_bottle_bowl_cup_vase
