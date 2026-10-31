# Towards Point Cloud Counterfactual Explanations

Implementation of **Towards Point Cloud Counterfactual Explanations** (EUSIPCO 2025). This project extends the [VAEX](https://github.com/nverchev/VAEX) framework to 3D point clouds, building upon the generative architecture of [PCGen](https://github.com/nverchev/PCGen) and using the [DRYTorch](https://github.com/nverchev/drytorch) engine for experiment tracking, configuration, and model training.

---

## Overview

### What are Point Cloud Counterfactuals?

Visual counterfactual explanations answer the question: *"How should the 3D shape change to flip the classifier's prediction to a target class while preserving the identity of the original object?"*

3D point clouds represent unstructured, permutation-invariant geometric data widely used in safety-critical applications such as autonomous driving and robotics. However, existing explainability methods for point clouds rely largely on saliency heatmaps, point removal, or adversarial coordinate perturbations. These approaches only identify influential local points or test classifier fragility without providing realistic, semantically coherent geometric alterations that reveal underlying model biases.

### The Two-Stage Generative Architecture

Generating realistic counterfactuals on point clouds requires precise semantic control without corrupting geometric surface fidelity. This framework builds upon the double autoencoder architecture of [PCGen](https://github.com/nverchev/PCGen):

1. **Outer VQ-VAE (Geometric Building Blocks)**:
   - Turns a point cloud into discrete codes $\bar{\mathbf{w}}$ chosen from a learned codebook of geometric primitives.
   - The outer point generator is **unconditioned** ($p(\mathbf{x} \mid \bar{\mathbf{w}}, \mathbf{y}) = p(\mathbf{x} \mid \bar{\mathbf{w}})$). The classifier's predictions reach individual points only through discrete code selection, preventing high-frequency coordinate noise and preserving surface realism.

2. **Inner VAE (Disentangled Semantic Latent Space)**:
   - Compresses the discrete codes $\bar{\mathbf{w}}$ into a continuous latent space partitioned into two components:
     - $\mathbf{z}_u$: captures geometric variations unrelated to the classification, with a standard Gaussian prior.
     - $\mathbf{z}_y$: isolates the semantic features driving the classifier's decision, steered by a learnable conditional prior $p(\mathbf{z}_y \mid \mathbf{y})$ derived from the classifier's soft predictions.

### Counterfactual Intervention

Counterfactual generation follows causal abduction-action-prediction:
- **Abduction**: Encodes the input point cloud into the latent posteriors $q(\mathbf{z}_u \mid \mathbf{x})$ and $q(\mathbf{z}_y \mid \mathbf{x}, \mathbf{y})$.
- **Action**: Intervenes on the decision-relevant latent $\mathbf{do}(\mathbf{Z}_y = \mathbf{z}_y')$ by evaluating the posterior mean under a target evaluation:
  $$\mathbf{y}' = (1 - \tau)\,\mathbf{y} + \tau\,\mathbf{t}$$
  where $\mathbf{y}$ is the temperature-scaled classifier prediction, $\mathbf{t}$ is the one-hot target class, and $\tau \in [0, 1]$ is the counterfactual strength. Using the posterior mean avoids sampling noise while keeping the explanation faithful to the input.
- **Prediction**: Decodes $\mathbf{z}_y'$ and $\mathbf{z}_u$ through the inner decoder to select new discrete codes, which the outer generator renders into a realistic 3D point cloud.

---

## Project structure

```
PointCloudCounterfactual/
├── configs/
│   └── experiment/               # Hydra experiment configurations (modelnet_dt, modelnet_bbcv, modelnet40)
├── external/                     # CUDA matching extension for earth mover distance
├── src/
│   ├── config/                   # Typed specifications and Hydra configuration schemas
│   ├── data/                     # ModelNet dataset loading, splits, and data augmentations
│   ├── lib/                      # DRYTorch components: models, objectives, hooks, loaders
│   ├── module/                   # DGCNN classifier, VQ-VAE encoder/decoder, codeword VAE
│   ├── trackers/                 # Experiment tracking and PyVista 3D visualization
│   └── utils/                    # Distances, neighbour operations, counterfactual interventions
├── tests/                        # Unit tests
├── train_classifier.py           # Classifier training
├── evaluate_classifier.py        # Classifier evaluation
├── train_autoencoder.py          # Outer VQ-VAE training (phase one)
├── evaluate_autoencoder.py       # Outer autoencoder evaluation
├── train_w_autoencoder.py        # Inner codeword VAE training (phase two)
├── evaluate_w_autoencoder.py     # Double reconstruction evaluation
├── evaluate_counterfactuals.py   # Counterfactual metrics and tables
├── visualize_counterfactuals.py  # 3D rendering of counterfactuals
└── run.sh                        # One-shot pipeline runner for both subsets
```

---

## Installation

```bash
uv sync                      # minimal installation
uv sync --all-extras --dev   # full dependencies, trackers, visualization, and dev tools
uv sync --extra emd          # builds the CUDA matching extension, see external/README.md
```

The CUDA extension compiles the accelerated earth mover distance matching routines. Without it, the loss falls back on an exact matching that is too slow for training.

### Configure the paths

Paths come from a gitignored `.env` file at the repository root:

```dotenv
DATA_DIR=/path/to/datasets
ROOT_EXP_DIR=./experiments
```

ModelNet40 downloads and extracts automatically on first use.

---

## Usage

### Experiments

Experiments run on two challenging subsets of ModelNet40 identified by the classifier's confusion matrix:
- `modelnet_dt`: Desk vs. Table (2 classes).
- `modelnet_bbcv`: Bottle, Bowl, Cup, Vase (4 classes).
- `modelnet40`: Full 40-class ModelNet40 dataset used to train the baseline classifier.

### Training Pipeline

The multi-stage training pipeline trains the classifier, followed by the two generative stages:

```bash
# 1. Train and evaluate the point cloud classifier
uv run train_classifier.py --config-name modelnet_dt
uv run evaluate_classifier.py --config-name modelnet_dt

# 2. Train and evaluate the outer VQ-VAE (geometric reconstruction)
uv run train_autoencoder.py --config-name modelnet_dt
uv run evaluate_autoencoder.py --config-name modelnet_dt

# 3. Train the inner codeword VAE (semantic conditioning)
uv run train_w_autoencoder.py --config-name modelnet_dt
uv run evaluate_w_autoencoder.py --config-name modelnet_dt
```

Run each phase with matching overrides (apart from `user.*`), as they form the run name that subsequent phases use to locate checkpoints.

To train the classifier on the full 40-class dataset (whose confusion matrix identified the two subsets):
```bash
uv run train_classifier.py --config-name modelnet40
uv run evaluate_classifier.py --config-name modelnet40
```

### Visualisation and Evaluation

- **Evaluate counterfactual metrics and test tables:**
  ```bash
  uv run evaluate_counterfactuals.py --config-name modelnet_dt
  ```

- **Render 3D counterfactual explanations:**
  ```bash
  # Renders misclassified clouds towards their correct class
  uv run visualize_counterfactuals.py --config-name modelnet_dt

  # Renders specific clouds towards all target classes
  uv run visualize_counterfactuals.py --config-name modelnet_dt user.plot.sample_indices=[0,1,2]
  ```

### End-to-End Final Pipeline

The final models train on the combined training and validation sets and evaluate on the test split, reproducing the numbers in [Results](#results):

```bash
# End-to-end run for one subset
uv run train_classifier.py --config-name modelnet_dt final=true
uv run evaluate_classifier.py --config-name modelnet_dt final=true
uv run train_autoencoder.py --config-name modelnet_dt final=true
uv run evaluate_autoencoder.py --config-name modelnet_dt final=true
uv run train_w_autoencoder.py --config-name modelnet_dt final=true
uv run evaluate_w_autoencoder.py --config-name modelnet_dt final=true
uv run evaluate_counterfactuals.py --config-name modelnet_dt final=true
uv run visualize_counterfactuals.py --config-name modelnet_dt final=true
```

To run both subsets (`modelnet_dt` and `modelnet_bbcv`) end-to-end:
```bash
bash run.sh
```

---

## Results

Evaluated on the test split with `final=true`:

| Test | Description | DT published | DT obtained | BBCV published | BBCV obtained |
|---|---|---|---|---|---|
| `Original` | Accuracy of the classifier on real test clouds | 87 % | 86.6 % | 89 % | 88.3 % |
| `NoSemantics` | Accuracy after removing the semantics of the classification ($\bar{\mathbf{y}}$) | 73 % | 79.9 % | 43 % | 41.1 % |
| `Success_0.5` | Counterfactual success rate towards every target class ($\tau = 0.5$) | 80 % | 75.8 % | 86 % | 84.4 % |
| `Success_1` | Counterfactual success rate towards every target class ($\tau = 1.0$) | 99 % | 99.0 % | 100 % | 99.8 % |
| `Misclassified_0.5` | Success rate of turning misclassified clouds back towards their true class ($\tau = 0.5$) | 87 % | 78.4 % | 95 % | 75.7 % |
| `Misclassified_1` | Success rate of turning misclassified clouds back towards their true class ($\tau = 1.0$) | 97 % | 97.6 % | 100 % | 100.0 % |

- `Original`, `Reconstruction` and `NoSemantics` report `Accuracy` against the true label, while the others report `Success` against the target class requested.
- `Success` measures all clouds evaluated towards all classes (including the already assigned class). `Flip` measures only decisions that actively revert.
- `Reconstruction` gives classifier accuracy on double reconstructions through the entire architecture.

Results vary with the seed, which is not fixed by default. The classifier is a main source of this variation, because it directly affects the counterfactual success: its test accuracy on DT ranged from 80 % to 87 % between trainings. Each evaluation also draws new points for the generator, so the same weights give slightly different results, and the rates of the misclassified clouds, computed on few clouds, vary the most.

---

## Citation

```bibtex
@inproceedings{vercheval2025towards,
  author={Vercheval, Nicolas and Pi\v{z}urica, Aleksandra},
  booktitle={2025 33rd European Signal Processing Conference (EUSIPCO)},
  title={Towards Point Cloud Counterfactual Explanations},
  year={2025},
  pages={1922--1926},
  doi={10.23919/EUSIPCO.2025}
}
```
