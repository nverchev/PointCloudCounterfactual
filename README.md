# Towards Point Cloud Counterfactual Explanations

Code from the paper published at EUSIPCO 2025. To replicate the results, please follow the instructions:
- make sure that torch.cuda and the local toolkit version match, otherwise modify the pyproject.toml file.
- modify the hydra_conf/config_all/user/user_settings.yaml to select the trackers you want to use.
- install the project dependencies and the trackers . For example:
```console
uv sync
uv pip install tqdm
uv pip install tensorboard
```
- install the packages in the external folder:
```console
uv pip install external/pytorch_structural_losses/ --no-build-isolation
uv pip install external/emd/ --no-build-isolation
```
- set up local path directories by adding them in a .env file (the datasets should download automatically):
```.env
DATASET_DIR={your dataset directory} 
ROOT_EXP_DIR={the directory for your experiments} 
```
- execute run.sh