# Towards Point Cloud Counterfactual Explanations

Code from the paper published at EUSIPCO 2025. To replicate the results, please follow the instructions:
- install the requirements.txt file.
- install the packages in the external folder.
- install drytorch from nverchev/drytorch (soon available on pypi)
- write a .env file with:
  - DATASET_DIR={your dataset directory} 
  - EXPERIMENT_DIR={the directory for your experiments} 
  - METADATA_DIR=./dataset_metadata
- the datasets should download automatically.
- execute run.sh