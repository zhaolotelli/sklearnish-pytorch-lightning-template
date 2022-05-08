import os
import sys
from model import ClassifierModelBase, RegressorModelBase
import torch
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from sklearn.base import ClassifierMixin, RegressorMixin, BaseEstimator


class BaseClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model_name, model_hparams, optimizer_name, optimizer_hparams, trainer_hparams):
        self.model = ClassifierModelBase(model_name, model_hparams, optimizer_name, optimizer_hparams)
        CHECKPOINT_PATH = trainer_hparams['ckp'] if 'ckp' in trainer_hparams.keys() else sys.path[0]
        device = trainer_hparams['device'] if 'device' in trainer_hparams.keys() else 'cpu'
        epoch = trainer_hparams['epoch'] if 'epoch' in trainer_hparams.keys() else 50
        self.batch_size = trainer_hparams['batch_size'] if 'batch_size' in trainer_hparams.keys() else 256
        self.num_workers = trainer_hparams['num_workers'] if 'num_workers' in trainer_hparams.keys() else 8
        self.trainer = pl.Trainer(
            default_root_dir=os.path.join(CHECKPOINT_PATH, model_name),  # Where to save models
            # We run on a single GPU (if possible)
            gpus=1 if str(device) == "cuda:0" else 0,
            # How many epochs to train for if no patience is set
            max_epochs=epoch,
            callbacks=[
                ModelCheckpoint(
                    save_weights_only=True, mode="max", monitor="val_acc"
                ),
                # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
                LearningRateMonitor("epoch"),
            ],  # Log learning rate every epoch
            progress_bar_refresh_rate=1000,
        )  # In case your notebook crashes due to the progress bar, consider increasing the refresh rate

    def fit(self, X, y, X_valid, y_valid):
        X, y = map(torch.tensor, (X, y))
        train_ds = TensorDataset(X, y)
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        X_valid, y_valid = map(torch.tensor, (X_valid, y_valid))
        val_ds = TensorDataset(X_valid, y_valid)
        val_dl = DataLoader(val_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        self.trainer.fit(self.model, train_dl, val_dl)
        self.model = ClassifierModelBase.load_from_checkpoint(
            self.trainer.checkpoint_callback.best_model_path
        )
        val_result = self.trainer.test(self.model, test_dataloaders=val_dl, verbose=False)
        print(val_result[0])

    def test(self, X_test, y_test):
        X_test, y_test = map(torch.tensor, (X_test, y_test))
        test_ds = TensorDataset(X_test, y_test)
        test_dl = DataLoader(test_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        test_result = self.trainer.test(self.model, test_dataloaders=test_dl, verbose=False)
        print(test_result[0])
        return test_result[0]

    def predict(self, X):
        self.model.freeze()
        X = torch.tensor(X)
        pred = self.model(X)
        ypred = torch.argmax(pred, dim=-1)  # classification
        self.model.unfreeze()
        return ypred.numpy()

    def load_from_checkpoint(self, ckp):
        if os.path.isfile(ckp):
            print(f"Found pretrained model at {ckp}, loading...")
            # Automatically loads the model with the saved hyper-parameters
            self.model = ClassifierModelBase.load_from_checkpoint(ckp)


class BaseRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, model_name, model_hparams, optimizer_name, optimizer_hparams, trainer_hparams):
        self.model = RegressorModelBase(model_name, model_hparams, optimizer_name, optimizer_hparams)
        CHECKPOINT_PATH = trainer_hparams['ckp'] if 'ckp' in trainer_hparams.keys() else sys.path[0]
        device = trainer_hparams['device'] if 'device' in trainer_hparams.keys() else 'cpu'
        epoch = trainer_hparams['epoch'] if 'epoch' in trainer_hparams.keys() else 50
        self.batch_size = trainer_hparams['batch_size'] if 'batch_size' in trainer_hparams.keys() else 256
        self.num_workers = trainer_hparams['num_workers'] if 'num_workers' in trainer_hparams.keys() else 8
        self.trainer = pl.Trainer(
            default_root_dir=os.path.join(CHECKPOINT_PATH, model_name),  # Where to save models
            # We run on a single GPU (if possible)
            gpus=1 if str(device) == "cuda:0" else 0,
            # How many epochs to train for if no patience is set
            max_epochs=epoch,
            callbacks=[
                ModelCheckpoint(
                    save_weights_only=True, mode="max", monitor="val_r2"
                ),
                # Save the best checkpoint based on the maximum val_r2 recorded. Saves only weights and not optimizer
                LearningRateMonitor("epoch"),
            ],  # Log learning rate every epoch
            progress_bar_refresh_rate=1000,
        )  # In case your notebook crashes due to the progress bar, consider increasing the refresh rate

    def fit(self, X, y, X_valid, y_valid):
        X, y = map(torch.tensor, (X, y))
        train_ds = TensorDataset(X, y)
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        X_valid, y_valid = map(torch.tensor, (X_valid, y_valid))
        val_ds = TensorDataset(X_valid, y_valid)
        val_dl = DataLoader(val_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        self.trainer.fit(self.model, train_dl, val_dl)
        self.model = RegressorModelBase.load_from_checkpoint(
            self.trainer.checkpoint_callback.best_model_path
        )
        val_result = self.trainer.test(self.model, test_dataloaders=val_dl, verbose=False)
        print(val_result[0])

    def test(self, X_test, y_test):
        X_test, y_test = map(torch.tensor, (X_test, y_test))
        test_ds = TensorDataset(X_test, y_test)
        test_dl = DataLoader(test_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        test_result = self.trainer.test(self.model, test_dataloaders=test_dl, verbose=False)
        print(test_result[0])
        return test_result[0]

    def predict(self, X):
        self.model.freeze()
        X = torch.tensor(X)
        pred = self.model(X)
        self.model.unfreeze()
        return pred.detach().numpy()

    def load_from_checkpoint(self, ckp):
        if os.path.isfile(ckp):
            print(f"Found pretrained model at {ckp}, loading...")
            # Automatically loads the model with the saved hyper-parameters
            self.model = RegressorModelBase.load_from_checkpoint(ckp)
