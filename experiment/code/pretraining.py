import pandas as pd
import pytorch_lightning as pl
import torch
import numpy as np

from lightly.loss import NegativeCosineSimilarity
from lightly.utils.debug import std_of_l2_normalized
from lightly.models.modules.heads import SimSiamPredictionHead, SimSiamProjectionHead

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from sits_siam.backbone import TransformerBackbone
from sits_siam.utils import SitsDatasetFromDataframe
from sits_siam.bottleneck import PoolingBottleneck
from sits_siam.augment import (
    RandomAddNoise,
    RandomTempSwapping,
    RandomTempShift,
    RandomTempRemoval,
    AddMissingMask,
    Normalize,
    Pipeline,
    ToPytorchTensor,
)

# disable scientific notation pytorch, keep 3 numbers after decimal
torch.set_printoptions(precision=3, sci_mode=False)

whole_df = pd.read_parquet("data/california_sits_bert_original.parquet")

median = [
    0.0656,
    0.0948,
    0.1094,
    0.1507,
    0.2372,
    0.2673,
    0.2866,
    0.2946,
    0.2679,
    0.1985,
]
iqd = [0.0456, 0.0536, 0.0946, 0.0769, 0.0851, 0.1053, 0.1066, 0.1074, 0.1428, 0.1376]


class FastSiamMultiViewTransform(object):
    def __init__(
        self,
        n_views: int = 2,
    ):
        self.n_views = n_views
        self.transform = Pipeline(
            [
                RandomAddNoise(0.02),
                RandomTempSwapping(max_distance=3),
                RandomTempShift(),
                RandomTempRemoval(),
                AddMissingMask(),
                Normalize(
                    a=median,
                    b=iqd,
                ),
                ToPytorchTensor(),
            ]
        )

    def __call__(self, sample: np.ndarray):
        return [
            self.transform({k: v.copy() for k, v in sample.items()})
            for _ in range(self.n_views)
        ]


whole_df = pd.read_parquet("data/california_sits_bert_original.parquet")

train_df = whole_df[whole_df.use_bert.isin([0, 2])].reset_index(drop=True)
val_df = whole_df[whole_df.use_bert == 1].reset_index(drop=True)

train_dataset = SitsDatasetFromDataframe(
    train_df, max_seq_len=45, transform=FastSiamMultiViewTransform()
)
val_dataset = SitsDatasetFromDataframe(
    val_df, max_seq_len=45, transform=FastSiamMultiViewTransform()
)


class TransformerClassifier(pl.LightningModule):
    def __init__(self, max_seq_len=45):
        super(TransformerClassifier, self).__init__()
        self.backbone = TransformerBackbone(max_seq_len=max_seq_len)
        self.bottleneck = PoolingBottleneck()
        self.projection_head = SimSiamProjectionHead(128, 512, 1024)
        self.prediction_head = SimSiamPredictionHead(1024, 512, 1024)

        self.criterion = NegativeCosineSimilarity()

    def forward(self, input):
        x = input["x"]
        doy = input["doy"]
        mask = input["mask"]

        f = self.backbone(x, doy, mask)
        f = self.bottleneck(f)
        z = self.projection_head(f)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p

    def training_step(self, batch, batch_idx):
        views = batch
        features = [self.forward(view) for view in views]
        zs = torch.stack([z for z, _ in features])
        ps = torch.stack([p for _, p in features])

        loss = 0.0
        for i in range(len(views)):
            mask = torch.arange(len(views), device=self.device) != i
            loss += self.criterion(ps[i], torch.mean(zs[mask], dim=0)) / len(views)

        self.log("train_loss", loss, prog_bar=True)
        self.log(
            "train_collapse",
            std_of_l2_normalized(ps[0].detach()),
            sync_dist=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        views = batch
        features = [self.forward(view) for view in views]
        zs = torch.stack([z for z, _ in features])
        ps = torch.stack([p for _, p in features])

        loss = 0.0
        for i in range(len(views)):
            mask = torch.arange(len(views), device=self.device) != i
            loss += self.criterion(ps[i], torch.mean(zs[mask], dim=0)) / len(views)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log(
            "val_collapse",
            std_of_l2_normalized(ps[0].detach()),
            sync_dist=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        views = batch
        features = [self.forward(view) for view in views]
        zs = torch.stack([z for z, _ in features])
        ps = torch.stack([p for _, p in features])

        loss = 0.0
        for i in range(len(views)):
            mask = torch.arange(len(views), device=self.device) != i
            loss += self.criterion(ps[i], torch.mean(zs[mask], dim=0)) / len(views)

        self.log("test_loss", loss, sync_dist=True)
        self.log("test_collapse", std_of_l2_normalized(ps[0].detach()), prog_bar=True)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4
        )
        return optim


train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=512, shuffle=True, num_workers=4
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=512, shuffle=False, num_workers=4
)

early_stopping = EarlyStopping(monitor="val_loss", patience=3, mode="min")

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss", filename="best_model", save_top_k=1, mode="min"
)

trainer = pl.Trainer(max_epochs=5, callbacks=[checkpoint_callback, early_stopping])
model = TransformerClassifier()


trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
model = TransformerClassifier.load_from_checkpoint(checkpoint_callback.best_model_path)

# Saving pytorch model backbone state dict
torch.save(model.backbone.state_dict(), "weights/fastsiam.pth")
