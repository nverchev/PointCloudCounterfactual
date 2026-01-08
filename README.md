# Towards Point Cloud Counterfactual Explanations

Code from the paper published at EUSIPCO 2025. To replicate the results, please follow the instructions:
- make sure that torch.cuda and the local toolkit version match, otherwise modify the pyproject.toml file.
- install the project dependencies.
- install the packages in the external folder.
- you can set up local path directories by adding them in a .env file as follows:
  DATASET_DIR={your dataset directory} 
  ROOT_EXP_DIR={the directory for your experiments} 
- modify the /hydra_conf/config_all/user/user_settings.yaml file to match your environment.
- the datasets should download automatically.
- execute run.sh