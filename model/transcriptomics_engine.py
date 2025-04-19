import sys
import torch
import pytorch_lightning as pl
import socket
import os
import logging
import contextlib
import hashlib      # stdlib is fine, but xxhash / blake3 are 5‑10× faster

if "mac" in socket.gethostname():
    sys.path.append(
        "/Users/sidharrthnagappan/Documents/University/Cambridge/Courses/Dissertation/dissertation/src"
    )
else:
    sys.path.append("/home/sn666/dissertation/src")

from models.hist_to_transcriptomics import HistopathologyToTranscriptomics
from models.diffusion import MultiMagnificationDiffusionModel

@contextlib.contextmanager
def suppress_logging():
    # Disable all logging messages of level CRITICAL and below.
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        # Re-enable logging.
        logging.disable(logging.NOTSET)


# TRANSCRIPTOMICS_MODEL_PATH = "/auto/archive/tcga/sn666/trained_models/hist_to_transcriptomics/h_to_t_uni_128_b_subsetgene_then_norm_relu/epoch=9-step=3950.ckpt"

# TRANSCRIPTOMICS_MODEL_PATH = "/auto/archive/tcga/sn666/trained_models/hist_to_transcriptomics/h_to_t_uni_porpoise_genes_2layer/epoch=9-step=3950.ckpt"


class MiniPatchDataset(torch.utils.data.Dataset):
    """
    Brutally simple dataset to store individual patches coming in
    and prepare for inference on the HistopathologyToTranscriptomics model.
    """

    def __init__(self, patch_features: torch.Tensor):
        self.patch_features = patch_features

    def __len__(self):
        return self.patch_features.shape[0]

    def __getitem__(self, idx):
        # return a single patch's features
        # the existence of the foundation_model_features key tells the model to
        # skip the foundation model's inference during the full forward pass
        return {"foundation_model_features": self.patch_features[idx]}


def load_model(checkpoint_path: str):
    """
    Load the HistopathologyToTranscriptomics model from a checkpoint.
    """
    if "hist_to_transcriptomics" in checkpoint_path:
        model = HistopathologyToTranscriptomics.load_from_checkpoint(checkpoint_path)
    elif "diffusion" in checkpoint_path:
        model = MultiMagnificationDiffusionModel.load_from_checkpoint(checkpoint_path)
    else:
        raise ValueError(f"Unknown model type in checkpoint path: {checkpoint_path}")
    return model


transcriptomics_model = None

"""
This global variable is used to store the transcriptomics observations of previously 
computed patches, so that we don't have to recompute them.

Why do this?
Because the same "high-resolution" patch can end up at multiple magnifications, and we don't want to recompute
"""
CACHE: dict[str, torch.Tensor] = {}     # fp16 preds on CPU to save RAM


def tensor_fingerprint(
    t: torch.Tensor,
    *,                         # force kw‑args
    ndigits: int | None = None # round to this many decimal places first
) -> str:
    """
    Return a hex string that is identical *iff* the tensor contents
    (after optional rounding) are identical.
    """
    if ndigits is not None:
        t = torch.round(t, decimals=ndigits)

    # Make sure we are hashing a contiguous CPU view with a stable dtype.
    # .contiguous() is a no‑op if the tensor is already C‑contiguous.
    buf = t.detach().to(dtype=torch.float32, device="cpu", copy=False).contiguous().numpy().tobytes()
    return hashlib.blake2b(buf, digest_size=8).hexdigest()  # 8 bytes → 16‑char hex

def get_transcriptomics_data(patch_features: torch.Tensor, transcriptomics_model_path: str) -> torch.Tensor:
    global transcriptomics_model
    
    if transcriptomics_model is None:
        # Load the model
        print("Loading transcriptomics model...")
        transcriptomics_model = load_model(transcriptomics_model_path)
    
    B, P, _ = patch_features.shape
    device  = patch_features.device
    out_dim = transcriptomics_model.num_outputs
    result  = torch.empty((B, P, out_dim), device=device)

    # -----------------------------------------------------------------------
    # 1) split cached vs. missing by hashing each slide tensor once
    # -----------------------------------------------------------------------
    missing_idx, fingerprints = [], []

    for i in range(B):
        fp = tensor_fingerprint(patch_features[i], ndigits=4)  # or None for exact
        if fp in CACHE:                      # hit
            result[i] = CACHE[fp].to(device)
        else:                                # miss
            fingerprints.append(fp)
            missing_idx.append(i)

    # -----------------------------------------------------------------------
    # 2) run the model once for all misses
    # -----------------------------------------------------------------------
    if missing_idx:
        feats  = patch_features[missing_idx]                 # (n_missing, P, 1024)
        loader = torch.utils.data.DataLoader(
            MiniPatchDataset(feats), batch_size=len(missing_idx)
        )

        trainer = pl.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            logger=False, enable_progress_bar=False, enable_model_summary=False,
        )

        with suppress_logging():
            preds = torch.cat(trainer.predict(transcriptomics_model, dataloaders=loader), dim=0)  # (n_missing, P, out_dim)

        # write to output and cache
        result[missing_idx] = preds
        for fp, pred in zip(fingerprints, preds):
            CACHE[fp] = pred.to(dtype=torch.float16, device="cpu")  # light‑weight cache

    return result

def get_num_transcriptomics_features(transcriptomics_model_path: str) -> int:
    global transcriptomics_model
    
    if transcriptomics_model is None:
        # Load the model
        print("Loading transcriptomics model...")
        transcriptomics_model = load_model(transcriptomics_model_path)
        
    # TODO: make this dynamic idk
    return transcriptomics_model.num_outputs


def load(path):
    # root_dir = '/home/sn666/rds/rds-cl-acs-qRKC0ovsKR0/sn666/healnet/data/tcga/tcga/wsi/luad_zzb20_uni'
    # assert root_dir is not None, f"set_preprocess_dir must be called before load!"
    # path = os.path.join(root_dir, slide_id + f"_{power:.3f}.pt")
    # assert os.path.isfile(path), f"Pre-process load: path '{path}' not found!"
    return torch.load(path)


if __name__ == "__main__":
    # load the patch features
    patch_features = load(
        "/home/sn666/rds/rds-cl-acs-qRKC0ovsKR0/sn666/healnet/data/tcga/tcga/wsi/luad_zzb20_uni/TCGA-4B-A93V-01Z-00-DX1.C263DC1C-298D-47ED-AAF8-128043828530_5.000.pt"
    )
    print(patch_features[0][0])
    transcriptomics_data = get_transcriptomics_data(patch_features[0][0])
    print(transcriptomics_data)
    # get the transcriptomics data
