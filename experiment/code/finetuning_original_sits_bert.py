import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassCohenKappa,
    MulticlassConfusionMatrix,
)
from torch.optim import Adam
import numpy as np

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import pytorch_lightning as pl

from print_color import print

from sits_siam.original import (
    OriginalSITSBert,
    OriginalSITSBertMissingMaskFix,
)
import random

from sits_siam.augment import (
    AddMissingMask,
    Pipeline,
    ToPytorchTensor,
)

from sits_siam.utils import SitsDatasetFromDataframe


def setup_seed():
    seed = 123

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def beautify_prints():
    torch.set_printoptions(precision=2, sci_mode=False, linewidth=200)
    np.set_printoptions(precision=2, suppress=True, linewidth=200)


setup_seed()
beautify_prints()

train_transforms = Pipeline(
    [
        # AddNDVIWeights(),
        # RandomChanSwapping(),
        # RandomChanRemoval(),
        # RandomAddNoise(0.02),
        # RandomTempSwapping(max_distance=3),
        # RandomTempShift(),
        AddMissingMask(),
        OriginalSITSBertMissingMaskFix(),
        # Normalize(
        #     # a=median,
        #     # b=iqd,
        # ),
        ToPytorchTensor(),
    ]
)

val_transforms = Pipeline(
    [
        # AddNDVIWeights(),
        AddMissingMask(),
        OriginalSITSBertMissingMaskFix(),
        # Normalize(
        #     # a=median,
        #     # b=iqd,
        # ),
        ToPytorchTensor(),
    ]
)

whole_df = pd.read_parquet("data/california_sits_bert_original.parquet")

train_df = whole_df[whole_df.use_bert == 0].reset_index(drop=True)
val_df = whole_df[whole_df.use_bert == 1].reset_index(drop=True)
test_df = whole_df[whole_df.use_bert == 2].reset_index(drop=True)

print(f"Train df={len(train_df)}, Val df={len(val_df)}, Test df={len(test_df)}")
train_dataset = SitsDatasetFromDataframe(train_df, max_seq_len=64, transform=train_transforms)
val_dataset = SitsDatasetFromDataframe(val_df, max_seq_len=64, transform=val_transforms)
test_dataset = SitsDatasetFromDataframe(test_df, max_seq_len=64, transform=val_transforms)

del train_df
del val_df
del test_df
del whole_df


class OriginalSITSBertTrainer(pl.LightningModule):
    def __init__(self, max_seq_len=40, num_classes=13):
        super(OriginalSITSBertTrainer, self).__init__()
        self.model = OriginalSITSBert(num_classes=num_classes, pretrained=True)

        self.criterion = nn.CrossEntropyLoss()
        self.train_oa = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.val_oa = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.test_oa = MulticlassAccuracy(num_classes=num_classes, average="micro")

        self.train_kappa = MulticlassCohenKappa(num_classes=num_classes)
        self.val_kappa = MulticlassCohenKappa(num_classes=num_classes)
        self.test_kappa = MulticlassCohenKappa(num_classes=num_classes)
        self.test_cm = MulticlassConfusionMatrix(
            num_classes=num_classes, normalize="true"
        )

    def forward(self, input):
        x = input["x"]
        doy = input["doy"]
        mask = input["mask"]

        outputs = self.model(x, doy, mask)

        return outputs

    def training_step(self, batch, batch_idx):
        targets = batch["y"].squeeze()
        outputs = self(batch)

        loss = self.criterion(outputs, targets)
        train_oa_score = self.train_oa(outputs, targets)
        train_kappa_score = self.train_kappa(outputs, targets)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_oa", train_oa_score, prog_bar=True)
        self.log("train_kappa", train_kappa_score, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        targets = batch["y"].squeeze()
        outputs = self(batch)

        loss = self.criterion(outputs, targets)
        val_oa_score = self.val_oa(outputs, targets)
        val_kappa_score = self.val_kappa(outputs, targets)

        # Log loss and oa score
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_oa", val_oa_score, prog_bar=True)
        self.log("val_kappa", val_kappa_score, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        targets = batch["y"].squeeze()
        outputs = self(batch)

        loss = self.criterion(outputs, targets)
        test_oa_score = self.test_oa(outputs, targets)
        test_kappa_score = self.test_kappa(outputs, targets)
        self.test_cm(outputs, targets)

        # Log loss and oa score
        self.log("test_loss", loss)
        self.log("test_oa", test_oa_score)
        self.log("test_kappa", test_kappa_score)

        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=2e-4)
        return optimizer


train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=1024, shuffle=False
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=1024, shuffle=False
)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1024, shuffle=False
)

early_stopping = EarlyStopping(monitor="val_loss", patience=2, mode="min")

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss", filename="best_model", save_top_k=1, mode="min"
)


trainer = pl.Trainer(
    max_epochs=-1,
    devices="auto",
    callbacks=[checkpoint_callback, early_stopping],
)

model = OriginalSITSBertTrainer()

trainer.fit(
    model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
)

model = OriginalSITSBertTrainer.load_from_checkpoint(
    checkpoint_callback.best_model_path
)
model = model.eval()

trainer.test(model=model, dataloaders=test_dataloader)
print(model.test_cm.compute())
