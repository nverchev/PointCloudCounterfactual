# Towards Point Cloud Counterfactual Explanations

Code from the paper published at EUSIPCO 2025. To replicate the results, please follow the instructions:
- make sure that torch.cuda and the local toolkit version match, otherwise modify the pyproject.toml file.
- install the project dependencies.
- install the packages in the external folder.
- write a .env file with:
  - DATASET_DIR={your dataset directory} 
  - EXPERIMENT_DIR={the directory for your experiments} 
  - METADATA_DIR=./dataset_metadata
- the datasets should download automatically.
- execute run.sh