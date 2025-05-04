import os
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
import clip
import nibabel as nib
from PIL import Image
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix
)
import monai
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRangePercentilesd,
    CenterSpatialCropd,
    Lambdad,
    SpatialPadd,
    RandFlipd,
    ToTensord
)
from monai.networks.nets import resnet50
import torch.nn.functional as F
from tqdm import tqdm
from collections import Counter

class Gleason3DDataModule:
    """
    Encapsulates MONAI transforms, dataset creation, and dataloaders
    for a 3D Gleason classification task with metadata.
    """
    def __init__(
        self,
        metadata_csv_path: str,
        prompt_csv_path: str,
        split: str,
        test_split: float = 0.2,
        val_split: float = 0.2,
        vol_size: tuple = (256, 256, 16),
        batch_size: int = 4,
        num_workers: int = 4,
        cache_rate: float = 1.0,
        random_seed: int = 42,
        device: str = "cuda",
    ):
        # config
        self.metadata_csv_path = metadata_csv_path
        self.prompt_csv_path = prompt_csv_path
        self.split = split
        self.vol_size = vol_size
        self.val_split = val_split
        self.test_split = test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_rate = cache_rate
        self.seed = random_seed
        self.device = device

        self.metadata_df = pd.read_csv(self.metadata_csv_path).dropna()
        self.prompt_df = pd.read_csv(self.prompt_csv_path).dropna()
        
        self.df = pd.concat([self.metadata_df, self.prompt_df['Prompt']], axis=1)
        print("main line62", self.df.columns, self.df.shape)
        
        # build list of dicts for MONAI

        self.data_dicts = [
            {
                "image": row["T2"],
                "prompt": row["Prompt"],
                "label": int(row["Gleason Class"])
            }
            for _, row in self.df.iterrows()
        ]

        # define a single transform for both train & val (augmentations can be added here)
    
        self.train_transforms = Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=['image'], axcodes='RAS'),
            Spacingd(keys=["image"], pixdim=(0.4, 0.4, 1.0), mode="bilinear", padding_mode='reflection'),
            ScaleIntensityRangePercentilesd(
                keys=["image"],
                lower=25, upper=75,
                b_min=-0.5, b_max=0.5,
                clip=False
            ),
            CenterSpatialCropd(keys=['image'], roi_size=self.vol_size),
            SpatialPadd(keys=['image'], spatial_size=self.vol_size, mode=('reflect')),    
            RandFlipd(keys=['image'], prob=0.5, spatial_axis=0),
            ToTensord(keys=['image'])
        ])
        self.val_transforms = Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=['image'], axcodes='RAS'),
            Spacingd(keys=["image"], pixdim=(0.4, 0.4, 1.0), mode="bilinear", padding_mode='reflection'),
            ScaleIntensityRangePercentilesd(
                keys=["image"],
                lower=25, upper=75,
                b_min=-0.5, b_max=0.5,
                clip=False
            ),
            CenterSpatialCropd(keys=['image'], roi_size=self.vol_size),
            SpatialPadd(keys=['image'], spatial_size=self.vol_size, mode=('reflect')),    
            # RandFlipd(keys=['image'], prob=0.5, spatial_axis=0),
            ToTensord(keys=['image'])
        ])
        self.test_transforms = self.val_transforms

        # Perform the split right away
        self._split_data()

    def _split_data(self):
        # First split off test
        trainval, test = train_test_split(
            self.data_dicts,
            test_size=self.test_split,
            random_state=self.seed,
            shuffle=True
        )
        # Then split train and val from the remaining
        relative_val = self.val_split / (1 - self.test_split)
        train, val = train_test_split(
            trainval,
            test_size=relative_val,
            random_state=self.seed,
            shuffle=True
        )

        self.train_dicts = train
        self.val_dicts = val
        self.test_dicts = test

    def setup(self):
        """Create MONAI CacheDatasets and PyTorch dataloaders."""
        
        if self.split == 'train':
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
                self.train_ds,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=(self.device != "cpu")
            )
            self.val_loader = DataLoader(
                self.val_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=(self.device != "cpu")
            )

        else:  
            self.test_ds = CacheDataset(
                data=self.test_dicts,
                transform=self.test_transforms,
                cache_rate=self.cache_rate
            )
            self.test_loader = DataLoader(
                self.test_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=(self.device != "cpu")
            )

        # if self.split == 'train':
        #     self.train_ds = monai.data.Dataset(
        #         data=self.train_dicts,
        #         transform=self.train_transforms,
        #     )
        #     self.val_ds = monai.data.Dataset(
        #         data=self.val_dicts,
        #         transform=self.val_transforms,
        #     )

        #     self.train_loader = DataLoader(
        #         self.train_ds,
        #         batch_size=self.batch_size,
        #         shuffle=True,
        #         num_workers=self.num_workers,
        #         pin_memory=(self.device != "cpu")
        #     )
        #     self.val_loader = DataLoader(
        #         self.val_ds,
        #         batch_size=self.batch_size,
        #         shuffle=False,
        #         num_workers=self.num_workers,
        #         pin_memory=(self.device != "cpu")
        #     )

        # else:  
        #     self.test_ds = monai.data.Dataset(
        #         data=self.test_dicts,
        #         transform=self.test_transforms,
        #     )
        #     self.test_loader = DataLoader(
        #         self.test_ds,
        #         batch_size=self.batch_size,
        #         shuffle=False,
        #         num_workers=self.num_workers,
        #         pin_memory=(self.device != "cpu")
        #     )

    def get_loaders(self):
        """Returns (train_loader, val_loader). Call setup() first."""
        if not hasattr(self, "train_loader"):
            self.setup()
        if self.split == 'train':
            return self.train_loader, self.val_loader
        else:
            return self.test_loader

class CLIP3DContrastive(nn.Module):
    """
    CLIP-style contrastive model with a 3D CNN image encoder and the standard CLIP text tower.

    Args:
        clip_model_name: name of OpenAI CLIP model to load (e.g., 'ViT-B/32').
        embed_dim:      dimensionality of the shared embedding space (e.g., 512).
        pretrained_3d:  whether to load pretrained MedicalNet weights for the 3D backbone.
        device:         torch device ('cuda' or 'cpu').
    """
    def __init__(
        self,
        clip_model_name: str = 'ViT-B/32',
        embed_dim:       int = 512,
        pretrained_3d:   bool = True,
        device:          str = 'cuda'
    ):
        super().__init__()
        self.device = device

        # 1) Load CLIP for the text encoder
        self.clip_model, _ = clip.load(clip_model_name, device=device)
        self.clip_model = self.clip_model.float()

        # Pull out the text components
        self.token_embedding      = self.clip_model.token_embedding
        self.positional_embedding = self.clip_model.positional_embedding
        self.transformer          = self.clip_model.transformer
        self.ln_final             = self.clip_model.ln_final
        self.text_projection      = self.clip_model.text_projection
        self.logit_scale          = self.clip_model.logit_scale

        # 2) Build a 3D visual tower from MONAI's MedicalNet ResNet50
        backbone = resnet50(
            spatial_dims=3,
            n_input_channels=1,
            pretrained=pretrained_3d,
            feed_forward=False,      # no final classifier
            shortcut_type="B",       # required for pretrained MedicalNet
            bias_downsample=False    # required for pretrained MedicalNet
        )
        # Determine output channels of the final ResNet block
        out_ch = backbone.layer4[-1].bn3.num_features
        # Wrap into a simple projection head matching CLIP's embed_dim
        self.visual = nn.Sequential(
            backbone,
            nn.Linear(out_ch, embed_dim, bias=False)
        )

    def encode_image(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Encodes a batch of 3D volumes into the CLIP embedding space.

        Args:
            volume: Tensor of shape (B, 1, D, H, W).
        Returns:
            Tensor of shape (B, embed_dim), L2-normalized.
        """
        x = self.visual(volume)
        return x / x.norm(dim=-1, keepdim=True)

    def encode_text(self, text_ids: torch.Tensor) -> torch.Tensor:
        """
        Encodes a batch of tokenized prompts into the CLIP embedding space.

        Args:
            text_ids: Tensor of shape (B, context_length).
        Returns:
            Tensor of shape (B, embed_dim), L2-normalized.
        """
        # token embedding + positional
        target_dtype = self.text_projection.dtype

        x = self.token_embedding(text_ids).to(target_dtype)
        x = x + self.positional_embedding.to(target_dtype)
        # transformer expects (L, B, C)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(x.dtype)
        # pick the end-of-text embedding
        eos_idx = text_ids.argmax(dim=-1)
        x = x[torch.arange(x.size(0)), eos_idx] @ self.text_projection
        return x / x.norm(dim=-1, keepdim=True)

    def forward(self, volume: torch.Tensor, text_ids: torch.Tensor):
        """
        Forward pass: returns image-to-text and text-to-image logits.

        Args:
            volume:  (B,1,D,H,W) tensor of 3D volumes.
            text_ids:(B,context_length) tensor of CLIP token IDs.

        Returns:
            logits_per_image, logits_per_text: each (B, B) tensor of cosine logits.
        """

        # 1) raw visual features (B, C)
        feat = self.visual(volume)  

        # Sanity check: print embedding spread once per forward in training
        # if self.training:
        #     # overall std across batch & channels
        #     std_all = feat.std().item()
        #     print(f"[DEBUG] visual embedding std: {std_all:.6f}")

        # 2) normalize into the joint embedding space
        img_feat = feat / feat.norm(dim=-1, keepdim=True)
        txt_feat = self.encode_text(text_ids)

        # 3) scaled cosine similarities
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * img_feat @ txt_feat.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text


def contrastive_loss(logits_img: torch.Tensor, logits_txt: torch.Tensor) -> torch.Tensor:
    """
    InfoNCE loss for matched pairs in a batch.
    Computes cross-entropy on image-to-text and text-to-image logits.
    """
    batch_size = logits_img.size(0)
    labels = torch.arange(batch_size, device=logits_img.device)
    loss_i = F.cross_entropy(logits_img, labels)
    loss_t = F.cross_entropy(logits_txt, labels)
    return (loss_i + loss_t) * 0.5

def train(args):

    # Create timestamped experiment directory
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(args.output_dir, now)
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Experiment directory: {exp_dir}")
    print()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 1) Prepare data
    dm = Gleason3DDataModule(
        metadata_csv_path = args.metadata_csv,
        prompt_csv_path = args.prompt_csv,
        split = args.split,
        test_split = args.test_split,
        val_split = args.val_split,
        vol_size = tuple(args.vol_size),
        batch_size = args.batch_size,
        num_workers = args.num_workers,
        cache_rate = args.cache_rate,
        random_seed = args.seed,
        device = device
    )

    if args.split == 'train':
        train_loader, val_loader = dm.get_loaders()

        # some inspection
        full_counts  = Counter(dm.df['Gleason Class'])
        train_counts = Counter([d['label'] for d in dm.train_dicts])
        val_counts   = Counter([d['label'] for d in dm.val_dicts])
        print(f"Full dataset label distribution: {full_counts}")
        print(f"Train split label distribution: {train_counts}")
        print(f"Val split label distribution:   {val_counts}")
        print()

    else:
        test_loader = dm.get_loaders()

    # 2) Build model
    model = CLIP3DContrastive(
        clip_model_name = args.clip_model,
        embed_dim = args.embed_dim,
        pretrained_3d = args.pretrained_3d,
        device = device
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    best_val_loss = float("inf")
    
    scaler = GradScaler()

    # 3) Training loop
    for epoch in range(1, args.epochs + 1):
        # -- train --
        model.train()
       
        train_loss = 0.0
        correct_i = correct_t = total = 0
        
        loop = tqdm(train_loader, desc=f"[Epoch {epoch}] train", leave=False)

        for batch in loop:
            optimizer.zero_grad()

            volumes = batch["image"].to(device)        # (B, 1, D, H, W) tensor
            volumes = volumes.permute(0, 1, 4, 2, 3).contiguous()       # (B, 1, H, W, D) tensor

            prompts = batch["prompt"]       # list[str] or tensor after collate
            labels  = batch["label"].to(device)        # (B,) tensor of ints

            text_ids = clip.tokenize(prompts, truncate=True).to(device)

            with autocast():
                logits_i, logits_t = model(volumes, text_ids)
                loss = contrastive_loss(logits_i, logits_t)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # logits_i, logits_t = model(volumes, text_ids)
            # loss = contrastive_loss(logits_i, logits_t)
            # loss.backward()
            # optimizer.step()

            # accumulate
            b = volumes.size(0)
            train_loss += loss.item() * b
            total += b
            labels = torch.arange(b, device=device)
            correct_i += (logits_i.argmax(1) == labels).sum().item()
            correct_t += (logits_t.argmax(1) == labels).sum().item()
            
            loop.set_postfix(loss=loss.item())

        train_loss /= total
        train_acc_i = correct_i / total
        train_acc_t = correct_t / total
        train_ret = 0.5 * (train_acc_i + train_acc_t)

        # -- validation --
        model.eval()
        val_loss = 0.0
        correct_i = correct_t = total = 0

        loop = tqdm(val_loader, desc=f"[Epoch {epoch}]  val", leave=False)
        with torch.no_grad():
            for batch in loop:
                volumes = batch["image"].to(device)        # (B, 1, D, H, W) tensor
                volumes = volumes.permute(0, 1, 4, 2, 3).contiguous()       # (B, 1, H, W, D) tensor

                prompts = batch["prompt"]       # list[str] or tensor after collate
                labels  = batch["label"].to(device)        # (B,) tensor of ints
                text_ids = clip.tokenize(prompts, truncate=True).to(device)

                logits_i, logits_t = model(volumes, text_ids)
                loss = contrastive_loss(logits_i, logits_t)
                # total_loss += loss.item() * volumes.size(0)
                b = volumes.size(0)
                val_loss += loss.item() * b
                total += b
                labels = torch.arange(b, device=device)
                correct_i += (logits_i.argmax(1) == labels).sum().item()
                correct_t += (logits_t.argmax(1) == labels).sum().item()

                loop.set_postfix(val_loss=loss.item())

        val_loss /= total
        val_acc_i = correct_i / total
        val_acc_t = correct_t / total
        val_ret = 0.5 * (val_acc_i + val_acc_t)

        # — LOGGING & CHECKPOINT —
        print("-"*100)
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"  Train loss: {train_loss:.4f} | Img→Txt Acc: {train_acc_i:.3f} | Txt→Img Acc: {train_acc_t:.3f} | Ret@1: {train_ret:.3f}")
        print(f"  Val   loss: {val_loss:.4f} | Img→Txt Acc: {val_acc_i:.3f} | Txt→Img Acc: {val_acc_t:.3f} | Ret@1: {val_ret:.3f}")

        # save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ckpt = os.path.join(exp_dir, f"best_epoch{epoch}_valloss{val_loss:.4f}.pt")
            torch.save(model.state_dict(), best_ckpt)
            print()
            print(f"  ▶ New best checkpoint (val_loss={best_val_loss:.4f}) saved to:\n    {best_ckpt}")
            print()

        print("-"*100)
        print("\n\n")

    # save last epoch
    last_ckpt = os.path.join(exp_dir, f"last_epoch{epoch}_valloss{val_loss:4f}.pt")
    torch.save(model.state_dict(), last_ckpt)
    print(f"\nTraining complete. Best val_loss={best_val_loss:.4f}")
    print(f"Last epoch checkpoint saved to:\n  {last_ckpt}")


if __name__ == '__main__':
    # required
    parser = argparse.ArgumentParser(description='Train CLIP3D Contrastive Model')
    parser.add_argument('--metadata_csv', type=str, required=True, help='Path to updated_metadata_all.csv')
    parser.add_argument('--prompt_csv', type=str, required=True, help='Path to updated_prompts.csv')
    # default 
    parser.add_argument('--split', type=str, default='train', help='whether it is train/test')
    parser.add_argument('--clip_model', type=str, default='ViT-B/32', help='CLIP vision model name')
    parser.add_argument('--embed_dim', type=int, default=512, help='Embedding dimension')
    parser.add_argument('--pretrained_3d', action='store_true', help='Use pretrained MedicalNet weights')
    parser.add_argument('--vol_size', nargs=3, type=int, default=[256,256,16], help='Volume spatial size (H W D)')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--val_split', type=float, default=0.2, help='Fraction of non-test data for validation')
    parser.add_argument('--test_split', type=float, default=0.2, help='Fraction of total data for test')
    parser.add_argument('--cache_rate', type=float, default=1.0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='/home/cpsc477_lc2382/project/ucla/Prostate-CLIP/exp', help="default experiment directory")
    parser.add_argument('--output', type=str, default='clip3d_best.pt', help='Where to save the best model')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    print(f"[MAIN] args\n {args}")
    print()

    train(args)
