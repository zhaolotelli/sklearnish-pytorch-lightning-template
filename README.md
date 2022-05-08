# sklearnish-pytorch-lightning-template

pytorch lightning model training in sklearnish way. 

## structure

model folder is written into a module providing `ClassifierModelBase` and `RegressorModelBase`, which are `pl.LightningModule`'s. 

base.py gives `BaseClassifier` and `BaseRegressor` in sklearnish way. 

main.py provides an example. 
