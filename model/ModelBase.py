import importlib
import torch
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl


class ClassifierModelBase(pl.LightningModule):
    def __init__(self, model_name, model_hparams, optimizer_name, optimizer_hparams):
        super().__init__()

        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # Create model
        self.model = self.create_model(model_name, model_hparams)
        # Create loss module
        self.loss_module = nn.CrossEntropyLoss()

    def forward(self, x):
        # Forward function that is run when visualizing the graph
        return self.model(x)

    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        if self.hparams.optimizer_name == "Adam":
            # AdamW is Adam with a correct implementation of weight decay (see here
            # for details: https://arxiv.org/pdf/1711.05101.pdf)
            optimizer = optim.AdamW(self.parameters(), **self.hparams.optimizer_hparams)
            # We will reduce the learning rate by 0.1 after 20 epochs
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        elif self.hparams.optimizer_name == "SGD":
            optimizer = optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
            # We will reduce the learning rate by 0.1 after 20 epochs
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        else:
            assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'

        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        pred = self.model(x)
        loss = self.loss_module(pred, y)
        ypred = torch.argmax(pred, dim = -1)
        acc = (ypred == y).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def evaluate(self, batch, stage = None):
        x, y = batch
        pred = self.model(x)
        loss = self.loss_module(pred, y)
        ypred = torch.argmax(pred, dim = -1)
        acc = (ypred == y).float().mean()

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, val_batch, batch_idx):
        self.evaluate(val_batch, stage="val")

    def test_step(self, test_batch, batch_idx):
        self.evaluate(test_batch, stage="test")

    def predict_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        ypred = torch.argmax(pred, dim = -1)
        return ypred

    def create_model(self, model_name, model_hparams):
        try:
            Model = getattr(importlib.import_module(
                '.'+model_name, package=__package__), model_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {model_name}!')
        return Model(**model_hparams)


class RegressorModelBase(pl.LightningModule):
    def __init__(self, model_name, model_hparams, optimizer_name, optimizer_hparams):
        super().__init__()

        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # Create model
        self.model = self.create_model(model_name, model_hparams)
        # Create loss module
        self.loss_module = nn.MSELoss(reduction='sum')

    def forward(self, x):
        # Forward function that is run when visualizing the graph
        return self.model(x)

    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        if self.hparams.optimizer_name == "Adam":
            # AdamW is Adam with a correct implementation of weight decay (see here
            # for details: https://arxiv.org/pdf/1711.05101.pdf)
            optimizer = optim.AdamW(self.parameters(), **self.hparams.optimizer_hparams)
            # We will reduce the learning rate by 0.1 after 10 epochs
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        elif self.hparams.optimizer_name == "SGD":
            optimizer = optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
            # We will reduce the learning rate by 0.1 after 10 epochs
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        else:
            assert False, f'Unknown optimizer: "{self.hparams.optimizer_name}"'

        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        pred = self.model(x).squeeze()
        loss = self.loss_module(pred, y)
        b = x.size(dim=0)
        r2 = 1 - torch.square(pred - y).sum() / torch.var(y) / b
        self.log("train_loss", loss)
        self.log("train_r2", r2)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        pred = self.model(x).squeeze()
        loss = self.loss_module(pred, y)
        b = x.size(dim=0)
        r2 = 1 - torch.square(pred - y).sum() / torch.var(y) / b

        if stage:
            self.log(f"b_{stage}_loss", loss, prog_bar=False)
            self.log(f"b_{stage}_r2", r2, prog_bar=False)

        return torch.stack((pred, y))

    def validation_step(self, val_batch, batch_idx):
        result = self.evaluate(val_batch, stage="val")
        return result

    def test_step(self, test_batch, batch_idx):
        result = self.evaluate(test_batch, stage="test")
        return result

    def validation_epoch_end(self, outputs):
        pred, y = torch.concat(outputs, 1)
        loss = self.loss_module(pred, y)
        N = y.size(dim=0)
        r2 = 1 - torch.square(pred - y).sum() / torch.var(y) / N
        self.log("val_r2", r2, prog_bar=True)

    def test_epoch_end(self, outputs):
        pred, y = torch.concat(outputs, 1)
        loss = self.loss_module(pred, y)
        N = y.size(dim=0)
        r2 = 1 - torch.square(pred - y).sum() / torch.var(y) / N
        self.log("test_r2", r2, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        return pred

    def create_model(self, model_name, model_hparams):
        try:
            Model = getattr(importlib.import_module(
                '.' + model_name, package=__package__), model_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {model_name}!')
        return Model(**model_hparams)
