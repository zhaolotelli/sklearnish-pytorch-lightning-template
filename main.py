from base import BaseClassifier


## ViT ##
model_name = "ViT"
model_hparams = {"image_size": (100, 40),
                 "patch_size": (10, 4),
                 "num_classes": 3,
                 "dim": 64,
                 "depth": 2,
                 "heads": 4,
                 "mlp_dim": 32,
                 "pool": 'cls',
                 "channels": 1,
                 "dim_head": 32}
optimizer_name = "Adam"
optimizer_hparams = {"lr": 0.0001,
                     "weight_decay": 0.1}
trainer_hparams = {"ckp": 'C:/Users/57232/Desktop/ustc/plmodels/CKP',
                   "epoch": 40,
                   "device": "cuda:0",
                   "batch_size": 256,
                   "num_workers": 8}
data_hparams = {}


## train ##
X, Y = get_data(**data_hparams, flag='train')
X_valid, Y_valid = get_data(**data_hparams, flag='valid')
X_test, Y_test = get_data(**data_hparams, flag='test')
clf = BaseClassifier(model_name=model_name,
                     model_hparams=model_hparams,
                     optimizer_name=optimizer_name,
                     optimizer_hparams=optimizer_hparams,
                     trainer_hparams=trainer_hparams)
clf.fit(X, Y, X_valid, Y_valid)
clf.test(X_test, Y_test)

## load pretrained model ##
# clf = BaseClassifier(model_name=model_name,
#                      model_hparams=model_hparams,
#                      optimizer_name=optimizer_name,
#                      optimizer_hparams=optimizer_hparams,
#                      trainer_hparams=trainer_hparams)
# ckp = '...'
# clf.load_from_checkpoint(ckp)
for i in range(10):
    print("true value: {}; predict value: {}".format(Y_test[i], clf.predict(X_test[[i], ])))
