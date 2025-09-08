python train_classifier.py final=True
python train_autoencoder.py final=True
python train_w_autoencoder.py final=True
python evaluate_counterfactuals.py final=True

python train_classifier.py final=True data/dataset=modelnet_bottle_bowl_cup_vase
python train_autoencoder.py final=True data/dataset=modelnet_bottle_bowl_cup_vase
python train_w_autoencoder.py final=True data/dataset=modelnet_bottle_bowl_cup_vase
python evaluate_counterfactuals.py final=True data/dataset=modelnet_bottle_bowl_cup_vase user.counterfactual_value=1