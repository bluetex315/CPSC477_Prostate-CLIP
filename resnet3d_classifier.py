#!/usr/bin/env python3
# finetune_classifier.py

import os
import datetime
import argparse
from collections import Counter
from typing import Callable

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm

import clip
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    Spacingd, ScaleIntensityRangePercentilesd,
    CenterSpatialCropd, SpatialPadd, RandFlipd,
    ToTensord
)
from monai.data import CacheDataset, Dataset as MonaiDataset
from monai.networks.nets.resnet import resnet18, resnet50

# import 3D CLIP definition
from main import CLIP3DContrastive
import pandas as pd
import torch
from torch.utils.data import DataLoader
from monai.data import CacheDataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    Spacingd, ScaleIntensityRangePercentilesd,
    CenterSpatialCropd, SpatialPadd, RandFlipd,
    ToTensord
)
from sklearn.model_selection import train_test_split


class Gleason3DFinetuneDataModule:
    """
    For downstream Gleason‐class cross‐entropy fine‐tuning.
    If split=='train' builds train+val loaders; otherwise only a test loader.
    """
    def __init__(
        self,
        metadata_csv_path: str,
        split: str = "train",       # "train" or "test"
        test_split: float = 0.2,
        val_split: float = 0.2,
        vol_size: tuple = (256,256,16),
        batch_size: int = 4,
        num_workers: int = 4,
        cache_rate: float = 1.0,
        random_seed: int = 42,
        device: str = "cuda",
        label_transform: Callable[[int], int] = lambda x: x,
    ):
        self.metadata_csv_path = metadata_csv_path
        self.split             = split
        self.test_split        = test_split
        self.val_split         = val_split
        self.vol_size          = vol_size
        self.batch_size        = batch_size
        self.num_workers       = num_workers
        self.cache_rate        = cache_rate
        self.seed              = random_seed
        self.device            = device
        self.label_transform   = label_transform

        # load & drop any rows where T2 or Gleason Class is NaN
        self.df = pd.read_csv(self.metadata_csv_path).dropna(subset=["T2","Gleason Class"])
        print(f"[DataModule] total examples: {len(self.df)}")

        # build simple dicts
        self.data_dicts = []

        for _, row in self.df.iterrows():
            orig_lbl = int(row["Gleason Class"])
            new_lbl  = self.label_transform(orig_lbl)
            self.data_dicts.append({
                "image": row["T2"],
                "label": new_lbl
            })

        # define transforms
        common = [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=(0.4,0.4,1.0), mode="bilinear"),
            ScaleIntensityRangePercentilesd(
                keys=["image"], lower=1, upper=99,
                b_min=0.0, b_max=1.0, clip=True
            ),
            CenterSpatialCropd(keys=["image"], roi_size=self.vol_size),
            SpatialPadd(keys=["image"], spatial_size=self.vol_size, mode="reflect"),
            ToTensord(keys=["image","label"])
        ]
        self.train_transforms = Compose(common + [RandFlipd(keys=["image"], prob=0.5, spatial_axis=0)])
        self.val_transforms   = Compose(common)
        self.test_transforms  = self.val_transforms

        # immediately split
        self._split_data()

    def _split_data(self):
        trainval, test = train_test_split(
            self.data_dicts,
            test_size=self.test_split,
            random_state=self.seed,
            shuffle=True
        )
        # now train/val
        rel_val = self.val_split / (1 - self.test_split)
        train, val = train_test_split(
            trainval,
            test_size=rel_val,
            random_state=self.seed,
            shuffle=True
        )
        self.train_dicts = train
        self.val_dicts   = val
        self.test_dicts  = test

        print(f"[DataModule] splits: train={len(train)}, val={len(val)}, test={len(test)}")

    def setup(self):
        """Instantiate datasets & dataloaders for the selected split."""
        if self.split == "train":
            self.train_ds = CacheDataset(
                data=self.train_dicts,
                transform=self.train_transforms,
                cache_rate=self.cache_rate
            )
            self.val_ds = CacheDataset(
                data=self.val_dicts,
                transform=self.val_transforms,
                cache_rate=self.cache_rate
            )
            self.train_loader = DataLoader(
                self.train_ds, batch_size=self.batch_size, shuffle=True,
                num_workers=self.num_workers, pin_memory=(self.device!="cpu")
            )
            self.val_loader   = DataLoader(
                self.val_ds,   batch_size=self.batch_size, shuffle=False,
                num_workers=self.num_workers, pin_memory=(self.device!="cpu")
            )
        else:
            self.test_ds = CacheDataset(
                data=self.test_dicts,
                transform=self.test_transforms,
                cache_rate=self.cache_rate
            )
            self.test_loader = DataLoader(
                self.test_ds, batch_size=self.batch_size, shuffle=False,
                num_workers=self.num_workers, pin_memory=(self.device!="cpu")
            )

    def get_loaders(self):
        if not hasattr(self, "train_loader") and self.split=="train":
            self.setup()
        if not hasattr(self, "test_loader") and self.split!="train":
            self.setup()
        if self.split=="train":
            return self.train_loader, self.val_loader
        else:
            return self.test_loader

class ClassifierHead(nn.Module):
    def __init__(self, backbone: CLIP3DContrastive, embed_dim: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        # freeze all of the pretrained CLIP3D weights
        for p in self.backbone.parameters():
            p.requires_grad = False
        # simple linear head
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, volume: torch.Tensor):
        # volume should be (B,1,D,H,W) already correctly ordered
        with torch.no_grad():
            emb = self.backbone.encode_image(volume)
        logits = self.fc(emb)
        return logits

class ResNet3DClassifier(nn.Module):
    def __init__(self, model_name, spatial_dims, in_channels, num_classes, pretrained=False):
        super().__init__()
        if model_name == 'resnet18':
            self.backbone = resnet18(
                spatial_dims=spatial_dims, n_input_channels=in_channels,
                pretrained=pretrained, feed_forward=False,
                shortcut_type='B', bias_downsample=False
            )
        else:
            self.backbone = resnet50(
                spatial_dims=spatial_dims, n_input_channels=in_channels,
                pretrained=pretrained, feed_forward=False,
                shortcut_type='B', bias_downsample=False
            )
        out_ch = self.backbone.layer4[-1].bn3.num_features
        # self.features = nn.Sequential(
        #     backbone,
        #     nn.AdaptiveAvgPool3d(1),
        #     nn.Flatten(1)
        # )
        self.classifier = nn.Linear(out_ch, num_classes)

    def forward(self, x):
        # x = self.features(x)
        x = self.backbone(x)
        return self.classifier(x)


def build_model(args):
    if args.finetune_CLIP:
        
        clip3d = CLIP3DContrastive(
            clip_model_name=args.clip_model,
            embed_dim=args.embed_dim,
            pretrained_3d=args.pretrained_3d,
            device=args.device
        ).to(args.device)

        ckpt = torch.load(args.pretrained_ckpt, map_location=args.device)
        clip3d.load_state_dict(ckpt, strict=True)
        print("Loaded pretrained contrastive weights from", args.pretrained_ckpt)

        num_classes = 2
        model = ClassifierHead(clip3d, args.embed_dim, num_classes).to(args.device)
    
    else:
        num_classes = 2
        model = ResNet3DClassifier(model_name=args.resnet_model, spatial_dims=3, in_channels=1, num_classes=num_classes, pretrained=args.pretrained_3d).to(args.device)

    return model

def train(args):
    # 0) experiment dir
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.finetune_CLIP:
        exp_dir = os.path.join(args.output_dir, 'finetune', now)
    else:
        exp_dir = os.path.join(args.output_dir, 'resnet_noCLIP', now)
    os.makedirs(exp_dir, exist_ok=True)
    print("Saving to", exp_dir)

    mapping = {0: 0, 1: 0, 2: 1}  # e.g. combine classes 0&1 into 0, 2→1
    # 1) data
    dm = Gleason3DFinetuneDataModule(
        metadata_csv_path=args.metadata_csv_path,
        vol_size=tuple(args.vol_size),
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_rate=args.cache_rate,
        val_split=args.val_split,
        test_split=args.test_split,
        random_seed=args.seed,
        device=args.device,
        label_transform=lambda x: mapping[x],
    )

    if args.split == "train":
        train_loader, val_loader = dm.get_loaders()
    
    else:
        test_loader = dm.get_loaders()

    # print dataset stats
    print("Label dist (full):", Counter([d["label"] for d in dm.data_dicts]))
    print("Label dist (train):", Counter([t["label"] for t in dm.train_ds.data]))
    print("Label dist (val):",   Counter([v["label"] for v in dm.val_ds.data]))

    # 2) model
    num_classes = 2     # binary class
    model = build_model(args)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    # 3) loop
    for epoch in range(1, args.epochs+1):
        # train
        model.train()
        all_preds, all_labels = [], []
        total_loss = 0.0

        train_loop = tqdm(
            train_loader,
            desc=f"[Epoch {epoch:02d}] ▶ train",
            leave=False
        )

        for batch in train_loop:
            vols = batch["image"].to(args.device)
            vols = vols.permute(0,1,4,2,3).contiguous()  # ensure (B,1,D,H,W)
            labels = batch["label"].to(args.device)

            optimizer.zero_grad()
            logits = model(vols)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * vols.size(0)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        train_loss = total_loss / len(train_loader.dataset)
        train_acc  = accuracy_score(all_labels, all_preds)

        # val
        model.eval()
        all_labels = []
        all_preds  = []   # for accuracy/confusion
        all_probs  = []   # for AUC
        val_loss   = 0.0

        val_loop = tqdm(
            val_loader,
            desc=f"[Epoch {epoch:02d}] ▶ val",
            leave=False
        )

        with torch.no_grad():
            for batch in val_loop:
                vols = batch["image"].to(args.device)
                vols = vols.permute(0,1,4,2,3).contiguous()
                labels = batch["label"].to(args.device)

                logits = model(vols)
                loss = criterion(logits, labels)
                val_loss += loss.item() * vols.size(0)

                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

                probs = F.softmax(logits, dim=1)       # (B,2)
                pos_probs = probs[:,1]                 # (B,) probability of class “1”
                all_probs.extend(pos_probs.cpu().numpy())

        val_loss = val_loss / len(val_loader.dataset)
        val_acc  = accuracy_score(all_labels, all_preds)
        val_precision = precision_score(all_labels, all_preds)
        val_recall = recall_score(all_labels, all_preds)

        try:
            val_auc = roc_auc_score(all_labels, all_probs)
            
        except:
            val_auc = float("nan")

        cm = confusion_matrix(all_labels, all_preds)

        print(f"Epoch {epoch}/{args.epochs}")
        print(f"  Train loss: {train_loss:.4f} | Acc: {train_acc:.3f}")
        print(f"  Val   loss: {val_loss:.4f} | Acc: {val_acc:.3f} | AUC: {val_auc:.3f} | Precision: {val_precision:.3f} | Recall: {val_recall:.3f}")
        print("  Confusion:\n", cm)

        # checkpoint best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            p = os.path.join(exp_dir, f"best_finetune_epoch{epoch}_acc{val_acc:.3f}.pt")
            torch.save(model.state_dict(), p)
            print("  saved best head to", p)

    # save last
    last_p = os.path.join(exp_dir, "last_finetune.pt")
    torch.save(model.state_dict(), last_p)
    print("Finished. Last head saved to", last_p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_csv_path",   type=str, required=True,
                        help="path to updated_metadata_all.csv")
    parser.add_argument("--finetune_CLIP", action='store_true')
    parser.add_argument("--pretrained_ckpt", type=str, default="/home/cpsc477_lc2382/project/ucla/Prostate-CLIP/exp/20250503_025050/best_epoch30_valloss2.0481.pt")
    
    parser.add_argument("--resnet_model",type=str, required=True,
                        help="which resnet model to use")
    
    parser.add_argument("--clip_model",     type=str, default="ViT-B/32")
    parser.add_argument("--embed_dim",      type=int, default=512)
    parser.add_argument("--pretrained_3d",  type=bool,default=True)
    parser.add_argument("--vol_size",       nargs=3, type=int, default=[256,256,16])
    parser.add_argument("--batch_size",     type=int, default=8)
    parser.add_argument("--num_workers",    type=int, default=4)
    parser.add_argument("--cache_rate",     type=float, default=1.0)
    parser.add_argument("--val_split",      type=float, default=0.2)
    parser.add_argument("--test_split",     type=float, default=0.2)
    parser.add_argument("--epochs",         type=int, default=100)
    parser.add_argument("--lr",             type=float,default=1e-4)
    parser.add_argument("--weight_decay",   type=float,default=1e-2)
    parser.add_argument("--output_dir",     type=str, default="/home/cpsc477_lc2382/project/ucla/Prostate-CLIP/exp")
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--device",         type=str, default="cuda")
    parser.add_argument("--split",          type=str, default='train')
    # parser.add_argument("--num_classes",    type=int, default=2)
    args = parser.parse_args()


    if args.split == 'train':
        train(args)
    else:
        test(args)
