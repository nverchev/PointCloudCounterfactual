# Towards Point Cloud Counterfactual Explanations

Code from the paper published at EUSIPCO 2025. To replicate the results, please follow the instructions:
- Specify the index for the correct cuda version in the uv.toml file (locally or in the uv global config):
```toml
[[index]]
name = "pypi"
url = "https://pypi.org/simple"
default = true

[[index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cu130"  # change this to match your cuda toolkit version
packages = ["torch", "torchvision", "torchaudio"]
```
- modify the hydra_conf/config_all/user/user_settings.yaml to select the trackers you want to use.
- install the project dependencies and the trackers. For example:
```console
uv sync
uv pip install tqdm
uv pip install tensorboard
```
- install the structural loss from the external folder:
```console
uv pip install external/pytorch_structural_losses/ --no-build-isolation --link-mode=copy
```
- set up local path directories by adding them in a .env file (the datasets should download automatically):
```.env
DATASET_DIR={your dataset directory}
ROOT_EXP_DIR={the directory for your experiments}
```
- execute run.sh
