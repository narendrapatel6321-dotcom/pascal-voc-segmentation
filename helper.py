"""
helper.py — PASCAL VOC 2012 Segmentation Helper
=================================================
Utility functions for dataset preparation, tf.data pipeline construction,
losses, metrics, visualization, and evaluation for the Pascal VOC 2012
semantic segmentation project (U-Net vs DeepLabV3+).

Sections
--------
0. Session Setup
    setup_data()                 — one-call Kaggle download + extract + split + remap
    
1. Dataset Preparation
    prepare_sbd()                — extract SBD tarball to local disk
    build_aug_split()            — build standard aug split CSVs (train/val/test)
    load_saved_splits()          — load CSVs and remap paths to local disk

2. tf.data Pipeline
    make_tf_dataset()            — build train/val/test tf.data pipelines

3. Losses
    DiceLoss                     — multiclass Dice loss, void-aware
    CombinedLoss                 — weighted CE + Dice, void-aware

4. Metrics
    PixelAccuracy                — pixel accuracy, void-aware
    MeanIoU                      — confusion-matrix-based mIoU, void-aware
    MeanDice                     — confusion-matrix-based mean Dice, void-aware

5. Visualization — Data Inspection
    plot_sample_pairs()          — image + mask pairs from dataset
    plot_augmentation_preview()  — one image augmented 6 times
    plot_class_distribution()    — pixel frequency per class bar chart

6. Training Analysis
    plot_training_curve()        — loss + mIoU curves with best epoch marker

7. Evaluation
    get_predictions()            — run inference, return (y_true, y_pred) arrays
    plot_confusion_matrix()      — normalized 21-class confusion matrix
    plot_per_class_iou()         — per-class IoU bar chart sorted by score
    plot_qualitative_results()   — image / GT mask / predicted mask triplets

8. Utilities
    label_to_rgb()               — integer mask → RGB using VOC palette
    get_best_model_path()        — path helper for best checkpoint

Dataset
-------
Train  : SBD train + SBD val, deduplicated against VOC val — ~10,582 images
Val    : 70% of VOC 2012 val (seed 21)                     —  ~1,014 images
Test   : 30% of VOC 2012 val (seed 21)                     —    ~435 images

Mask format : single-channel uint8 PNG, values = VOC class indices (0–20)
Void label  : 255 — excluded from loss, metrics, and evaluation

Storage Strategy
----------------
Data lives on the shared Colab_Experiments Drive folder with read/write
access across all accounts. Setup runs once ever — any account that runs
the setup functions writes to the shared folder and all others see it
immediately via the guard checks.

Every training session copies the raw tarballs from Drive to /content/
and extracts locally. All tf.data reads happen from local SSD — never
directly from Drive — for maximum pipeline throughput.

Usage
-----
    # Every session — one call handles everything
    from helper import setup_data, make_tf_dataset

    train_df, val_df, test_df = setup_data(CKPT_ROOT, PROJECT)
    train_ds = make_tf_dataset(train_df, split="train", img_size=512, batch_size=8)
    val_ds   = make_tf_dataset(val_df,   split="val",   img_size=512, batch_size=8)
    test_ds  = make_tf_dataset(test_df,  split="test",  img_size=512, batch_size=8)

    # Fine-grained control (advanced / one-time setup only)
    from helper import prepare_sbd, build_aug_split, load_saved_splits

    prepare_sbd(DATA_DIR)
    build_aug_split(
        voc_dir = DATA_DIR / "VOCdevkit",
        sbd_dir = DATA_DIR / "benchmark_RELEASE" / "dataset",
        out_dir = DATA_DIR
    )
    train_df, val_df, test_df = load_saved_splits(
        data_dir          = DATA_DIR,
        local_voc_dir     = Path("/content/VOCdevkit/VOC2012"),
        local_sbd_png_dir = Path("/content/sbd_masks_png")
    )
"""

from pathlib import Path
import os
import tarfile
import shutil
import zipfile
import subprocess
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import scipy.io


def setup_data(
    ckpt_root,
    project,
    local_dir      = Path("/content/voc_seg_data"),
    kaggle_dataset = "narendraiitb27/voc-sdb-raw",
) -> tuple:
    """
    One-call data setup for every Colab session.

    Handles Kaggle auth, download (guarded), SBD extraction (guarded),
    mask conversion (guarded), split building (guarded), and path remapping.
    Safe to re-run every session — all expensive steps are skipped if already done.

    First ever run        : download + extract + convert masks + build splits (~20 min)
    Every session after   : skips everything except path remapping (~5 sec)

    Parameters
    ----------
    ckpt_root      : str or Path — root of shared Drive folder (from find_checkpoint_root)
    project        : str         — project name, e.g. "pascal_voc_segmentation"
    local_dir      : Path        — local SSD directory for extracted data
                                   (default: /content/voc_seg_data)
    kaggle_dataset : str         — Kaggle dataset slug

    Returns
    -------
    tuple : (train_df, val_df, test_df)

    Example
    -------
    >>> train_df, val_df, test_df = setup_data(CKPT_ROOT, PROJECT)
    """
    ckpt_root  = Path(ckpt_root)
    local_dir  = Path(local_dir)
    data_dir   = ckpt_root / project / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    local_dir.mkdir(parents=True, exist_ok=True)

    # ── Kaggle auth ───────────────────────────────────────────
    kaggle_json = ckpt_root / "kaggle.json"
    if not kaggle_json.exists():
        raise FileNotFoundError(
            f"kaggle.json not found at {kaggle_json}\n"
            f"Upload kaggle.json to {ckpt_root} on Drive once."
        )
    os.makedirs("/root/.kaggle", exist_ok=True)
    shutil.copy2(kaggle_json, "/root/.kaggle/kaggle.json")
    os.chmod("/root/.kaggle/kaggle.json", 0o600)
    print(" Kaggle authenticated.")

    # ── Local path constants ──────────────────────────────────
    voc_extracted  = local_dir / "VOCtrainval_11-May-2012"
    sbd_extracted  = local_dir / "benchmark_RELEASE"
    sbd_png_dir    = local_dir / "sbd_masks_png" / "sbd_masks_png"
    local_voc_dir  = voc_extracted / "VOCdevkit" / "VOC2012"
    local_sbd_dir  = sbd_extracted / "dataset"

    # ── Download from Kaggle (guarded) ────────────────────────
    if not voc_extracted.exists() or not sbd_extracted.exists() or not sbd_png_dir.exists():
        print(" Downloading dataset from Kaggle...")
        result = subprocess.run(
            ["kaggle", "datasets", "download", kaggle_dataset, "-p", str(local_dir)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Kaggle download failed:\n{result.stderr}")

        zip_path = local_dir / "voc-sdb-raw.zip"
        print(" Extracting zip...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(local_dir)
        zip_path.unlink()
        print(" Download and extraction complete.")
    else:
        print(" Dataset already on local disk, skipping download.")

    # ── Extract SBD tarball (guarded inside prepare_sbd) ─────
    sbd_tar = local_dir / "benchmark.tgz"
    prepare_sbd(tar_path=sbd_tar, extract_dir=sbd_extracted)
    if sbd_tar.exists():
        sbd_tar.unlink()

    # ── Convert masks + build splits (guarded inside build_aug_split) ──
    build_aug_split(
        voc_dir = local_voc_dir.parent.parent,   # → VOCdevkit/
        sbd_dir = local_sbd_dir,
        out_dir = data_dir,
    )

    # ── Load splits and remap paths to local SSD ─────────────
    train_df, val_df, test_df = load_saved_splits(
        data_dir          = data_dir,
        local_voc_dir     = local_voc_dir,
        local_sbd_png_dir = sbd_png_dir,
        local_sbd_dir     = local_sbd_dir,
    )

    return train_df, val_df, test_df

def prepare_sbd(tar_path, extract_dir) -> None:
    tar_path    = Path(tar_path)
    extract_dir = Path(extract_dir)
    sentinel    = extract_dir / ".extracted"

    if sentinel.exists():
        print(" SBD already extracted, skipping.")
        return

    if extract_dir.exists():
        print(" Partial extraction detected — cleaning up.")
        shutil.rmtree(extract_dir)

    print(" Extracting SBD...")
    with tarfile.open(tar_path, "r:*") as tar:
        tar.extractall(extract_dir.parent)
    sentinel.touch()

    dataset_dir = extract_dir / "dataset"
    for subdir in ["img", "cls"]:
        if not (dataset_dir / subdir).exists():
            raise FileNotFoundError(
                f"SBD dataset subdirectory missing: {dataset_dir / subdir}\n"
                f"The archive may be corrupted."
            )
    for fname in ["train.txt", "val.txt"]:
        if not (dataset_dir / fname).exists():
            raise FileNotFoundError(
                f"SBD split file missing: {dataset_dir / fname}\n"
                f"The archive may be corrupted."
            )
    print(f" SBD ready at {extract_dir}")

SEED = 21
NUM_CLASSES = 21

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

def _convert_sbd_mat_to_png(sbd_msk_dir, png_dir) -> None:
    """
    Convert all SBD .mat mask files to single-channel uint8 PNGs.
    Skips already-converted files. Called internally by build_aug_split().

    Parameters
    ----------
    sbd_msk_dir : Path — path to benchmark_RELEASE/dataset/cls/
    png_dir     : Path — output directory for converted PNGs
    """
    
    png_dir.mkdir(parents=True, exist_ok=True)
    mat_files = list(sbd_msk_dir.glob("*.mat"))

    already_done = sum(1 for f in mat_files if (png_dir / f"{f.stem}.png").exists())
    to_convert   = len(mat_files) - already_done

    if to_convert == 0:
        print(f" SBD masks already converted ({len(mat_files):,} PNGs found), skipping.")
        return

    print(f" Converting {to_convert:,} SBD .mat masks to PNG "
          f"({already_done:,} already done)...")
    
    for mat_path in tqdm(mat_files, desc="Converting SBD masks", ncols=80):
        png_path = png_dir / f"{mat_path.stem}.png"
        if png_path.exists():
            continue
        mat  = scipy.io.loadmat(str(mat_path))
        mask = mat["GTcls"][0, 0][1].astype(np.uint8)
        Image.fromarray(mask).save(png_path)

    print(f" Conversion complete — {len(mat_files):,} PNGs saved to {png_dir}")


def build_aug_split(voc_dir, sbd_dir, out_dir) -> None:
    """
    Build the standard VOC aug split (as used in DeepLabV3+ paper) and save
    train.csv, val.csv, test.csv to out_dir.

    Converts all SBD .mat masks to PNG before building the split — pipeline
    is pure TF ops after this, no scipy needed during training.

    Split strategy:
        Train : SBD train + SBD val, deduplicated against VOC val — ~10,582 images
        Val   : 70% of VOC 2012 val (seed 21)                     — ~1,014 images
        Test  : 30% of VOC 2012 val (seed 21)                     —   ~435 images

    CSV columns:
        image_path : absolute path to JPEG image
        mask_path  : absolute path to PNG mask (VOC palette or converted SBD)

    Parameters
    ----------
    voc_dir : str or Path
        Path to VOCdevkit/
    sbd_dir : str or Path
        Path to benchmark_RELEASE/dataset/
    out_dir : str or Path
        Directory where train.csv, val.csv, test.csv and sbd_masks_png/ will
        be saved.

    Returns
    -------
    None

    Example
    -------
    >>> build_aug_split(
    ...     voc_dir = DATA_DIR / "VOCdevkit",
    ...     sbd_dir = DATA_DIR / "benchmark_RELEASE" / "dataset",
    ...     out_dir = DATA_DIR
    ... )
    """
    voc_dir = Path(voc_dir)
    sbd_dir = Path(sbd_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Guard — skip if already built
    needed = ["train.csv", "val.csv", "test.csv"]
    if all((out_dir / f).exists() for f in needed):
        print(" Splits already built. Use load_saved_splits() to load them.")
        return

    # ── Convert SBD masks ─────────────────────────────────────
    sbd_msk_dir = sbd_dir / "cls"
    png_dir     = out_dir / "sbd_masks_png"
    _convert_sbd_mat_to_png(sbd_msk_dir, png_dir)

    # ── Paths ─────────────────────────────────────────────────
    voc2012     = voc_dir / "VOC2012"
    seg_dir     = voc2012 / "ImageSets" / "Segmentation"
    voc_img_dir = voc2012 / "JPEGImages"
    voc_msk_dir = voc2012 / "SegmentationClass"
    sbd_img_dir = sbd_dir / "img"

    # ── Load split files as sets for O(1) dedup ───────────────
    def read_split(path):
        with open(path) as f:
            return set(line.strip() for line in f if line.strip())

    voc_val_ids   = read_split(seg_dir / "val.txt")
    sbd_train_ids = read_split(sbd_dir / "train.txt")
    sbd_val_ids   = read_split(sbd_dir / "val.txt")

    # ── SBD train pool — deduplicated against VOC val ─────────
    sbd_pool = (sbd_train_ids | sbd_val_ids) - voc_val_ids

    # ── Build rows ────────────────────────────────────────────
    train_rows = [
        {
            "image_path": str(sbd_img_dir / f"{img_id}.jpg"),
            "mask_path" : str(png_dir     / f"{img_id}.png"),
        }
        for img_id in sbd_pool
    ]

    voc_val_rows = [
        {
            "image_path": str(voc_img_dir / f"{img_id}.jpg"),
            "mask_path" : str(voc_msk_dir / f"{img_id}.png"),
        }
        for img_id in sorted(voc_val_ids)
    ]

    # ── Val / Test split from VOC val — seed 21 ───────────────
    rng     = np.random.default_rng(SEED)
    indices = rng.permutation(len(voc_val_rows))
    n_val   = int(len(voc_val_rows) * 0.7)

    val_rows  = [voc_val_rows[i] for i in indices[:n_val]]
    test_rows = [voc_val_rows[i] for i in indices[n_val:]]

    # ── Save ──────────────────────────────────────────────────
    pd.DataFrame(train_rows).to_csv(out_dir / "train.csv", index=False)
    pd.DataFrame(val_rows).to_csv(out_dir / "val.csv",     index=False)
    pd.DataFrame(test_rows).to_csv(out_dir / "test.csv",   index=False)

    print(f" Aug split built and saved to {out_dir}")
    print(f"   Train : {len(train_rows):,} images (SBD aug)")
    print(f"   Val   : {len(val_rows):,} images (VOC 2012 val 70%)")
    print(f"   Test  : {len(test_rows):,} images (VOC 2012 val 30%)")

def load_saved_splits(data_dir, local_voc_dir=None, local_sbd_png_dir=None, local_sbd_dir=None):
    """
    Load pre-built train/val/test CSV manifests from Drive.

    Parameters
    ----------
    data_dir         : str or Path — directory containing train.csv, val.csv, test.csv
    local_voc_dir    : str or Path, optional — local path to VOCdevkit/VOC2012/
    local_sbd_png_dir: str or Path, optional — local path to sbd_masks_png/
    local_sdb_dir: PLACE_HOLDER

    Returns
    -------
    tuple : (train_df, val_df, test_df)
    """
    data_dir = Path(data_dir)

    for fname in ["train.csv", "val.csv", "test.csv"]:
        if not (data_dir / fname).exists():
            raise FileNotFoundError(
                f"Missing {fname} in {data_dir}.\n"
                f"Run build_aug_split() first."
            )

    train_df = pd.read_csv(data_dir / "train.csv")
    val_df   = pd.read_csv(data_dir / "val.csv")
    test_df  = pd.read_csv(data_dir / "test.csv")

    # ── Remap paths ───────────────────────────────────────────
    if local_voc_dir is not None or local_sbd_png_dir is not None:
        local_voc_dir     = Path(local_voc_dir)     if local_voc_dir     else None
        local_sbd_png_dir = Path(local_sbd_png_dir) if local_sbd_png_dir else None

        voc_img_local = local_voc_dir / "JPEGImages"        if local_voc_dir     else None
        voc_msk_local = local_voc_dir / "SegmentationClass" if local_voc_dir     else None

        def remap_row(p):
            p_str = str(p)
            p = Path(p)

            # SBD images are .jpg in img/, masks are .png in sbd_masks_png/
            # VOC images are .jpg in JPEGImages/, masks are .png in SegmentationClass/
            # Distinguish by suffix + parent folder name stored in path
            if "sbd_masks_png" in p_str and local_sbd_png_dir:
                return str(local_sbd_png_dir / p.name)
            if "JPEGImages" in p_str and voc_img_local:
                return str(voc_img_local / p.name)
            if "SegmentationClass" in p_str and voc_msk_local:
                return str(voc_msk_local / p.name)
            if Path(p).parent.name == "img" and local_sbd_dir:
                return str(Path(local_sbd_dir) / "img" / Path(p).name)
            return p_str  # fallback to original if no remapping rule applies

        for df in [train_df, val_df, test_df]:
            df["image_path"] = df["image_path"].apply(remap_row)
            df["mask_path"]  = df["mask_path"].apply(remap_row)

        print(" Paths remapped to local disk.")

    print(f" Splits loaded from {data_dir}")
    print(f"   Train : {len(train_df):,} | Val : {len(val_df):,} | Test : {len(test_df):,}")

    return train_df, val_df, test_df

# ─────────────────────────────────────────────
# ImageNet normalization constants
# ─────────────────────────────────────────────

IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
IMAGENET_STD  = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

# ── Shared low-level ops (used by both make_tf_dataset and prewarm_cache) ──

def _load_image_mask(img_path, mask_path):
    """Read image + mask from disk, decode, cast."""
    img  = tf.io.read_file(img_path)
    img  = tf.image.decode_jpeg(img, channels=3)
    img  = tf.cast(img, tf.float32) / 255.0
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.cast(mask, tf.int32)
    return img, mask


def _resize_image_mask(img, mask, img_size):
    """Resize image (bilinear) and mask (nearest) to img_size×img_size."""
    img  = tf.image.resize(img,  [img_size, img_size], method="bilinear")
    mask = tf.image.resize(mask, [img_size, img_size], method="nearest")
    return img, mask

def _augment(img, mask, img_size):
    """
    Train augmentation — random scale crop + flip + color jitter.
    Scale range 0.75–1.25x keeps resize cost low while preserving
    meaningful scale variation.
    """
    # 1. Random scale + crop
    scale    = tf.random.uniform([], 0.75, 1.25)
    new_size = tf.cast(
        tf.round(tf.cast([img_size, img_size], tf.float32) * scale),
        tf.int32
    )
    new_size = tf.maximum(new_size, img_size)

    img  = tf.image.resize(img,  new_size, method="bilinear")
    mask = tf.image.resize(mask, new_size, method="nearest")

    # Synchronized crop
    img_mask = tf.concat([img, tf.cast(mask, tf.float32)], axis=-1)
    img_mask = tf.image.random_crop(img_mask, size=[img_size, img_size, 4])
    img  = img_mask[:, :, :3]
    mask = tf.cast(img_mask[:, :, 3:], tf.int32)

    # 2. Synchronized horizontal flip
    img_mask = tf.concat([img, tf.cast(mask, tf.float32)], axis=-1)
    img_mask = tf.image.random_flip_left_right(img_mask)
    img  = img_mask[:, :, :3]
    mask = tf.cast(img_mask[:, :, 3:], tf.int32)

    # 3. Color jitter — image only
    img = tf.image.random_brightness(img, 0.3)
    img = tf.image.random_contrast(img,   0.7, 1.3)
    img = tf.image.random_saturation(img, 0.7, 1.3)
    img = tf.image.random_hue(img,         0.1)
    img = tf.clip_by_value(img, 0.0, 1.0)

    return img, mask


def _normalize(img, mask):
    """ImageNet mean/std normalization + squeeze mask channel."""
    img  = (img - IMAGENET_MEAN) / IMAGENET_STD
    mask = tf.squeeze(mask, axis=-1)
    return img, mask

def prewarm_cache(
    train_df,
    val_df,
    cache_dir,
    img_size = 512,
) -> None:
    import time
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    for df, name in [(train_df, "train"), (val_df, "val")]:
        sentinel = cache_dir / f".{name}_cache_ready"

        # Guard — skip if already warmed this session
        if sentinel.exists():
            print(f" {name} cache already warm, skipping.")
            continue

        img_paths  = df["image_path"].values
        mask_paths = df["mask_path"].values

        ds = (
            tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
            .map(_load_image_mask,   num_parallel_calls=tf.data.AUTOTUNE)
            .map(
                lambda img, mask: _resize_image_mask(img, mask, img_size),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            .cache(str(cache_dir / name))
            .batch(32)
            .prefetch(tf.data.AUTOTUNE)
        )

        print(f" Pre-warming {name} cache ({len(df):,} images)...")
        start = time.time()
        for _ in ds:
            pass
        elapsed = (time.time() - start) / 60
        print(f"   Done in {elapsed:.1f} min → {cache_dir / name}")
        del ds

        sentinel.touch()  # mark as done

    print(" Cache ready.")

def make_tf_dataset(
    df,
    split,
    img_size   = 512,
    batch_size = 8,
    seed       = SEED,
    cache_path = None,
) -> tf.data.Dataset:
    """
    Build a tf.data pipeline for segmentation from a CSV manifest DataFrame.

    Pipeline order:
        Train : shuffle paths → load → resize → cache (optional) →
                shuffle samples → augment → normalize → batch → prefetch
        Val   : load → resize → cache (optional) → normalize → batch → prefetch
        Test  : load → resize → normalize → batch → prefetch

    Augmentation:
        Random scale (0.75–1.25x) → random crop 512×512 → horizontal flip
        → color jitter (image only). Scale capped at 1.25x vs previous 2.0x
        to keep CPU resize cost low.

    Shuffle strategy:
        Pre-cache  : shuffles file path strings only — zero RAM cost
        Post-cache : buffer=500, shuffles decoded samples — ~1.5GB RAM max

    All images are normalized with ImageNet mean/std.
    Masks are integer class indices (0–20), void label = 255.

    Parameters
    ----------
    df         : pd.DataFrame with columns image_path, mask_path
    split      : str            — 'train', 'val', or 'test'
    img_size   : int            — output spatial size (default: 512)
    batch_size : int            — batch size (default: 8)
    seed       : int            — random seed (default: 21)
    cache_path : str, optional  — path prefix for tf.data disk cache
                                  (e.g. "/content/tf_cache/train")
                                  Must match path used in prewarm_cache().
                                  If None, no caching is applied.

    Returns
    -------
    tf.data.Dataset yielding (image, mask) batches:
        image : float32 (batch, H, W, 3) — normalized
        mask  : int32   (batch, H, W)    — class indices

    Example
    -------
    >>> CACHE_DIR = Path("/content/tf_cache")
    >>> prewarm_cache(train_df, val_df, CACHE_DIR, img_size=IMG_SIZE)
    >>> train_ds = make_tf_dataset(train_df, split="train", img_size=512,
    ...                            batch_size=8, cache_path=str(CACHE_DIR/"train"))
    >>> val_ds   = make_tf_dataset(val_df,   split="val",   img_size=512,
    ...                            batch_size=8, cache_path=str(CACHE_DIR/"val"))
    >>> test_ds  = make_tf_dataset(test_df,  split="test",  img_size=512,
    ...                            batch_size=8)   # no cache needed for test
    """
    img_paths  = df["image_path"].values
    mask_paths = df["mask_path"].values

    ds = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))

    # Pre-cache shuffle — operates on path strings only, zero RAM cost
    if split == "train":
        ds = ds.shuffle(
            buffer_size           = len(img_paths),
            seed                  = seed,
            reshuffle_each_iteration = True
        )

    # Load + resize
    ds = ds.map(_load_image_mask, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(
        lambda img, mask: _resize_image_mask(img, mask, img_size),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Cache to SSD — reads from disk happen only on first epoch
    if cache_path:
        ds = ds.cache(cache_path)

    # Post-cache: re-shuffle decoded samples (small buffer — RAM safe)
    # then augment. Val/test skip both.
    if split == "train":
        ds = ds.shuffle(
            buffer_size           = 500,
            seed                  = seed,
            reshuffle_each_iteration = True
        )
        ds = ds.map(
            lambda img, mask: _augment(img, mask, img_size),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    ds = ds.map(_normalize, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=(split == "train"))
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return ds
  
# ─────────────────────────────────────────────
# 3. Losses
# ─────────────────────────────────────────────

VOID_LABEL = 255


@tf.keras.utils.register_keras_serializable(package="voc_seg")
class DiceLoss(tf.keras.losses.Loss):
    """
    Multiclass Dice loss for semantic segmentation.

    Computes 1 - mean Dice coefficient averaged over all valid classes.
    Void label (255) is excluded from the computation.

    Parameters
    ----------
    num_classes : int   — number of classes including background (default: 21)
    smooth      : float — smoothing term to avoid division by zero (default: 1e-6)

    Example
    -------
    >>> loss_fn = DiceLoss()
    >>> loss = loss_fn(y_true, y_pred)  # y_pred: logits (B, H, W, 21)
    """

    def __init__(self, num_classes=NUM_CLASSES, smooth=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.smooth      = smooth

    def call(self, y_true, y_pred):
        # y_true : (B, H, W)    int32  class indices, 255 = void
        # y_pred : (B, H, W, C) float32 logits
        y_true = tf.cast(y_true, tf.int32)
        # Valid pixel mask — exclude void
        valid = tf.cast(tf.not_equal(y_true, VOID_LABEL), tf.float32)  # (B, H, W)

        # Probabilities
        probs = tf.nn.softmax(y_pred, axis=-1)                          # (B, H, W, C)

        # One-hot encode — clamp void pixels to 0 to avoid out-of-range
        y_true_clamp = tf.where(
            tf.equal(y_true, VOID_LABEL),
            tf.zeros_like(y_true),
            y_true
        )
        y_true_oh = tf.one_hot(y_true_clamp, self.num_classes)          # (B, H, W, C)

        # Apply valid mask — zero out void pixels in both
        valid_exp  = tf.expand_dims(valid, axis=-1)                     # (B, H, W, 1)
        probs      = probs     * valid_exp
        y_true_oh  = y_true_oh * valid_exp

        # Dice per class
        axes        = [0, 1, 2]                                         # reduce B, H, W
        intersection = tf.reduce_sum(probs * y_true_oh, axis=axes)      # (C,)
        denominator  = tf.reduce_sum(probs + y_true_oh, axis=axes)      # (C,)
        dice_per_cls = (2.0 * intersection + self.smooth) / (denominator + self.smooth)

        return 1.0 - tf.reduce_mean(dice_per_cls)

    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes, "smooth": self.smooth})
        return config


@tf.keras.utils.register_keras_serializable(package="voc_seg")
class CombinedLoss(tf.keras.losses.Loss):
    """
    Weighted combination of CrossEntropy and Dice loss.

        loss = ce_weight * CE + (1 - ce_weight) * Dice

    Void label (255) is ignored in both CE and Dice.

    Parameters
    ----------
    ce_weight   : float — weight for CrossEntropy term (default: 0.5)
    num_classes : int   — number of classes (default: 21)
    smooth      : float — Dice smoothing term (default: 1e-6)

    Example
    -------
    >>> loss_fn = CombinedLoss(ce_weight=0.5)
    >>> loss = loss_fn(y_true, y_pred)  # y_pred: logits (B, H, W, 21)
    """

    def __init__(self, ce_weight=0.5, num_classes=NUM_CLASSES, smooth=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.ce_weight   = ce_weight
        self.num_classes = num_classes
        self.smooth      = smooth
        self._dice       = DiceLoss(num_classes=num_classes, smooth=smooth)
        self._ce         = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits    = True,
            ignore_class   = VOID_LABEL,
            reduction      = "sum_over_batch_size"
        )

    def call(self, y_true, y_pred):
        y_true    = tf.cast(y_true, tf.int32)
        ce_loss   = self._ce(y_true, y_pred)
        dice_loss = self._dice(y_true, y_pred)
    
        # Guard against NaN
        ce_loss   = tf.where(tf.math.is_nan(ce_loss),   tf.zeros_like(ce_loss),   ce_loss)
        dice_loss = tf.where(tf.math.is_nan(dice_loss), tf.zeros_like(dice_loss), dice_loss)
    
        return self.ce_weight * ce_loss + (1.0 - self.ce_weight) * dice_loss

    def get_config(self):
        config = super().get_config()
        config.update({
            "ce_weight"  : self.ce_weight,
            "num_classes": self.num_classes,
            "smooth"     : self.smooth
        })
        return config

# ─────────────────────────────────────────────
# 4. Metrics
# ─────────────────────────────────────────────

@tf.keras.utils.register_keras_serializable(package="voc_seg")
class PixelAccuracy(tf.keras.metrics.Metric):
    """
    Pixel accuracy for semantic segmentation.

    Computes the fraction of correctly classified pixels,
    excluding void label (255).

    Example
    -------
    >>> metric = PixelAccuracy()
    >>> metric.update_state(y_true, y_pred)  # y_pred: logits (B, H, W, 21)
    >>> metric.result()
    """

    def __init__(self, name="pixel_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.correct = self.add_weight(name="correct", initializer="zeros")
        self.total   = self.add_weight(name="total",   initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true : (B, H, W)    int32
        # y_pred : (B, H, W, C) float32 logits
        y_true = tf.cast(y_true, tf.int32)
        pred  = tf.argmax(y_pred, axis=-1, output_type=tf.int32)  # (B, H, W)
        valid = tf.not_equal(y_true, VOID_LABEL)                   # (B, H, W)

        correct = tf.logical_and(tf.equal(pred, y_true), valid)

        self.correct.assign_add(tf.cast(tf.reduce_sum(tf.cast(correct, tf.int32)), tf.float32))
        self.total.assign_add(  tf.cast(tf.reduce_sum(tf.cast(valid,   tf.int32)), tf.float32))

    def result(self):
        return tf.math.divide_no_nan(self.correct, self.total)

    def reset_state(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)

    def get_config(self):
        return super().get_config()


@tf.keras.utils.register_keras_serializable(package="voc_seg")
class MeanIoU(tf.keras.metrics.Metric):
    """
    Mean Intersection over Union (mIoU) for semantic segmentation.

    Accumulates a confusion matrix over batches, computes per-class IoU
    at result() time. Void label (255) is excluded.

    Parameters
    ----------
    num_classes : int — number of classes (default: 21)

    Example
    -------
    >>> metric = MeanIoU()
    >>> metric.update_state(y_true, y_pred)
    >>> metric.result()
    """

    def __init__(self, num_classes=NUM_CLASSES, name="mean_iou", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.confusion   = self.add_weight(
            name        = "confusion",
            shape       = (num_classes, num_classes),
            initializer = "zeros"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        pred  = tf.argmax(y_pred, axis=-1, output_type=tf.int32)  # (B, H, W)
        valid = tf.not_equal(y_true, VOID_LABEL)                   # (B, H, W)

        y_true_flat = tf.boolean_mask(y_true, valid)
        pred_flat   = tf.boolean_mask(pred,   valid)

        indices = tf.stack([y_true_flat, pred_flat], axis=1)       # (N, 2)
        updates = tf.ones(tf.shape(y_true_flat)[0], dtype=tf.float32)
        cm      = tf.tensor_scatter_nd_add(
            tf.zeros((self.num_classes, self.num_classes), dtype=tf.float32),
            indices,
            updates
        )
        self.confusion.assign_add(cm)

    def result(self):
        # IoU per class = TP / (TP + FP + FN)
        tp       = tf.linalg.diag_part(self.confusion)             # (C,)
        fp       = tf.reduce_sum(self.confusion, axis=0) - tp      # (C,)
        fn       = tf.reduce_sum(self.confusion, axis=1) - tp      # (C,)
        iou      = tf.math.divide_no_nan(tp, tp + fp + fn)         # (C,)

        # Only average over classes that appear in ground truth
        valid_cls = tf.reduce_sum(self.confusion, axis=1) > 0      # (C,)
        iou       = tf.where(valid_cls, iou, tf.zeros_like(iou))
        return tf.math.divide_no_nan(
            tf.reduce_sum(iou),
            tf.cast(tf.reduce_sum(tf.cast(valid_cls, tf.int32)), tf.float32)
        )

    def reset_state(self):
        self.confusion.assign(tf.zeros((self.num_classes, self.num_classes)))

    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes})
        return config


@tf.keras.utils.register_keras_serializable(package="voc_seg")
class MeanDice(tf.keras.metrics.Metric):
    """
    Mean Dice coefficient for semantic segmentation.

    Accumulates a confusion matrix over batches, computes per-class Dice
    at result() time. Void label (255) is excluded.

    Parameters
    ----------
    num_classes : int — number of classes (default: 21)

    Example
    -------
    >>> metric = MeanDice()
    >>> metric.update_state(y_true, y_pred)
    >>> metric.result()
    """

    def __init__(self, num_classes=NUM_CLASSES, name="mean_dice", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.confusion   = self.add_weight(
            name        = "confusion",
            shape       = (num_classes, num_classes),
            initializer = "zeros"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        pred  = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        valid = tf.not_equal(y_true, VOID_LABEL)

        y_true_flat = tf.boolean_mask(y_true, valid)
        pred_flat   = tf.boolean_mask(pred,   valid)

        indices = tf.stack([y_true_flat, pred_flat], axis=1)
        updates = tf.ones(tf.shape(y_true_flat)[0], dtype=tf.float32)
        cm      = tf.tensor_scatter_nd_add(
            tf.zeros((self.num_classes, self.num_classes), dtype=tf.float32),
            indices,
            updates
        )
        self.confusion.assign_add(cm)

    def result(self):
        # Dice per class = 2TP / (2TP + FP + FN)
        tp       = tf.linalg.diag_part(self.confusion)
        fp       = tf.reduce_sum(self.confusion, axis=0) - tp
        fn       = tf.reduce_sum(self.confusion, axis=1) - tp
        dice     = tf.math.divide_no_nan(2.0 * tp, 2.0 * tp + fp + fn)

        # Only average over classes that appear in ground truth
        valid_cls = tf.reduce_sum(self.confusion, axis=1) > 0
        dice      = tf.where(valid_cls, dice, tf.zeros_like(dice))
        return tf.math.divide_no_nan(
            tf.reduce_sum(dice),
            tf.cast(tf.reduce_sum(tf.cast(valid_cls, tf.int32)), tf.float32)
        )

    def reset_state(self):
        self.confusion.assign(tf.zeros((self.num_classes, self.num_classes)))

    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes})
        return config
    
# ─────────────────────────────────────────────
# 5. Visualization — Data Inspection
# ─────────────────────────────────────────────

def _build_voc_palette() -> np.ndarray:
    """Build the standard VOC 2012 color palette (256 x 3 uint8)."""
    palette = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j)
            g |= ((c >> 1) & 1) << (7 - j)
            b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        palette[i] = [r, g, b]
    return palette

VOC_PALETTE = _build_voc_palette()  # (256, 3) — computed once at import


def plot_sample_pairs(dataset, n_rows=4, save_path=None) -> None:
    """
    Plot a grid of image + colored mask pairs from a tf.data dataset.

    Each row shows one sample — left: original image, right: segmentation mask
    colored with the VOC palette.

    Parameters
    ----------
    dataset   : tf.data.Dataset — yields (image, mask) batches
                image : float32 (B, H, W, 3) normalized
                mask  : int32   (B, H, W)    class indices
    n_rows    : int             — number of samples to display (default: 4)
    save_path : str or Path, optional — if provided, saves figure to this path

    Returns
    -------
    None

    Example
    -------
    >>> plot_sample_pairs(val_ds, n_rows=4, save_path=PLOTS_DIR / "sample_pairs.png")
    """
    # Denormalize image — reverse ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    images, masks = [], []
    for img_batch, mask_batch in dataset:
        for img, mask in zip(img_batch.numpy(), mask_batch.numpy()):
            images.append(img)
            masks.append(mask)
            if len(images) == n_rows:
                break
        if len(images) == n_rows:
            break

    fig, axes = plt.subplots(n_rows, 2, figsize=(8, n_rows * 3))
    axes      = np.atleast_2d(axes)

    for i, (img, mask) in enumerate(zip(images, masks)):
        # Denormalize
        img_show = np.clip(img * std + mean, 0, 1)

        # Map mask to RGB — void pixels (255) shown as black
        mask_rgb          = VOC_PALETTE[np.where(mask == 255, 0, mask)]
        mask_rgb[mask == 255] = 0

        axes[i, 0].imshow(img_show)
        axes[i, 0].axis("off")
        axes[i, 0].set_title("Image", fontsize=9)

        axes[i, 1].imshow(mask_rgb)
        axes[i, 1].axis("off")
        axes[i, 1].set_title("Mask", fontsize=9)

    plt.suptitle("Sample Image — Mask Pairs", fontsize=11, y=1.01)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f" Saved → {save_path}")

    plt.show()


def plot_augmentation_preview(df, img_size=512, save_path=None) -> None:
    """
    Show one image augmented 6 times side by side — image row + mask row.

    Picks a random sample from df, applies train augmentation 6 times
    independently, and displays the results in a 2×6 grid (top: images,
    bottom: masks). Useful for verifying augmentation strength and
    mask-image synchronization.

    Parameters
    ----------
    df        : pd.DataFrame — with columns image_path, mask_path
    img_size  : int          — spatial size passed to make_tf_dataset (default: 512)
    save_path : str or Path, optional

    Returns
    -------
    None

    Example
    -------
    >>> plot_augmentation_preview(train_df, save_path=PLOTS_DIR / "aug_preview.png")
    """
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    N    = 6

    # Pick one random sample
    row       = df.sample(1, random_state=SEED).iloc[0]
    img_path  = row["image_path"]
    mask_path = row["mask_path"]

    # Build a single-row dataset repeated N times with train augmentation
    single_df = pd.DataFrame([{"image_path": img_path, "mask_path": mask_path}] * N)
    aug_ds    = make_tf_dataset(single_df, split="train", img_size=img_size, batch_size=N)

    # Take one batch — all N augmented versions of the same image
    img_batch, mask_batch = next(iter(aug_ds))
    images = img_batch.numpy()
    masks  = mask_batch.numpy()

    fig, axes = plt.subplots(2, N, figsize=(N * 3, 7))

    for i in range(N):
        img_show          = np.clip(images[i] * std + mean, 0, 1)
        mask_rgb          = VOC_PALETTE[np.where(masks[i] == 255, 0, masks[i])]
        mask_rgb[masks[i] == 255] = 0

        axes[0, i].imshow(img_show)
        axes[0, i].axis("off")
        axes[0, i].set_title(f"Aug {i + 1}", fontsize=9)

        axes[1, i].imshow(mask_rgb)
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Image", fontsize=10)
    axes[1, 0].set_ylabel("Mask",  fontsize=10)

    plt.suptitle(
        f"Augmentation Preview — {Path(img_path).stem}",
        fontsize=11, y=1.01
    )
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f" Saved → {save_path}")

    plt.show()

def plot_class_distribution(df, save_path=None) -> None:
    """
    Plot pixel frequency per VOC class across the dataset.

    Reads all masks in df, counts pixels per class, and displays a
    horizontal bar chart sorted by frequency. Void pixels (255) are excluded.
    Useful for understanding class imbalance before training.

    Note: Scans all masks on disk — may take a few minutes for large splits.

    Parameters
    ----------
    df        : pd.DataFrame — with column mask_path
    save_path : str or Path, optional

    Returns
    -------
    None

    Example
    -------
    >>> plot_class_distribution(train_df, save_path=PLOTS_DIR / "class_dist.png")
    """
    
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)

    for mask_path in tqdm(df["mask_path"], desc="Scanning masks", ncols=80):
        mask = np.array(Image.open(mask_path))
        for cls_idx in range(NUM_CLASSES):
            counts[cls_idx] += np.sum(mask == cls_idx)

    # Sort by frequency
    order      = np.argsort(counts)
    sorted_cls = [VOC_CLASSES[i] for i in order]
    sorted_cnt = counts[order] / 1e6  # convert to millions for readability

    fig, ax = plt.subplots(figsize=(9, 8))

    bars = ax.barh(
        sorted_cls, sorted_cnt,
        color="#4C72B0", alpha=0.85, edgecolor="white", linewidth=0.5
    )

    # Value labels
    for bar, val in zip(bars, sorted_cnt):
        ax.text(
            bar.get_width() + sorted_cnt.max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}M",
            va="center", fontsize=8
        )

    ax.set_xlabel("Pixel Count (millions)", fontsize=10)
    ax.set_title("VOC Class Pixel Distribution", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f" Saved → {save_path}")

    plt.show()

# ─────────────────────────────────────────────
# 6. Training Analysis
# ─────────────────────────────────────────────

def plot_training_curve(csv_path, save_path=None) -> None:
    """
    Plot loss and mIoU training curves from a CSVLogger log file.

    Displays two panels side by side:
        Left  : train loss + val loss, marks best val loss epoch
        Right : train mIoU + val mIoU, marks best val mIoU epoch

    Parameters
    ----------
    csv_path  : str or Path — path to training_log.csv produced by ResumableTrainer
    save_path : str or Path, optional

    Returns
    -------
    None

    Example
    -------
    >>> plot_training_curve(
    ...     csv_path  = CKPT_ROOT / PROJECT / "unet" / "training_log.csv",
    ...     save_path = PLOTS_DIR / "unet_training_curve.png"
    ... )
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Training log not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Normalize epoch to 1-indexed for display
    epochs = df["epoch"].values + 1

    # Identify column names — ResumableTrainer uses whatever the model reports
    loss_col     = "loss"
    val_loss_col = "val_loss"
    miou_col     = next((c for c in df.columns if "mean_iou" in c and "val" not in c), None)
    val_miou_col = next((c for c in df.columns if "val" in c and "mean_iou" in c), None)

    has_miou = miou_col is not None and val_miou_col is not None

    n_panels = 2 if has_miou else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    # ── Loss panel ────────────────────────────────────────────
    ax = axes[0]
    ax.plot(epochs, df[loss_col],     label="Train Loss", linewidth=1.5, color="#4C72B0")
    ax.plot(epochs, df[val_loss_col], label="Val Loss",   linewidth=1.5, color="#DD8452")

    best_loss_epoch = epochs[df[val_loss_col].idxmin()]
    ax.axvline(best_loss_epoch, color="#DD8452", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(
        best_loss_epoch + 0.3, ax.get_ylim()[1] * 0.95,
        f"best epoch {best_loss_epoch}",
        color="#DD8452", fontsize=8
    )

    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Loss",  fontsize=10)
    ax.set_title("Loss",   fontsize=11)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.3)

    # ── mIoU panel ────────────────────────────────────────────
    if has_miou:
        ax = axes[1]
        ax.plot(epochs, df[miou_col],     label="Train mIoU", linewidth=1.5, color="#4C72B0")
        ax.plot(epochs, df[val_miou_col], label="Val mIoU",   linewidth=1.5, color="#DD8452")

        best_miou_epoch = epochs[df[val_miou_col].idxmax()]
        ax.axvline(best_miou_epoch, color="#DD8452", linestyle="--", linewidth=1, alpha=0.7)
        ax.text(
            best_miou_epoch + 0.3, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05,
            f"best epoch {best_miou_epoch}",
            color="#DD8452", fontsize=8
        )

        ax.set_xlabel("Epoch",  fontsize=10)
        ax.set_ylabel("mIoU",   fontsize=10)
        ax.set_title("Mean IoU", fontsize=11)
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.3)

    plt.suptitle(f"Training Curve — {csv_path.parent.name}", fontsize=12, y=1.02)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f" Saved → {save_path}")

    plt.show()

# ─────────────────────────────────────────────
# 7. Evaluation
# ─────────────────────────────────────────────

def get_predictions(model, dataset) -> tuple:
    """
    Run inference on a dataset and return ground truth and predicted masks.

    Parameters
    ----------
    model   : tf.keras.Model — trained segmentation model
    dataset : tf.data.Dataset — yields (image, mask) batches

    Returns
    -------
    tuple : (y_true, y_pred)
        y_true : np.ndarray (N, H, W) int32   — ground truth class indices
        y_pred : np.ndarray (N, H, W) int32   — predicted class indices

    Example
    -------
    >>> y_true, y_pred = get_predictions(model, val_ds)
    """
    y_true_list, y_pred_list = [], []

    for img_batch, mask_batch in dataset:
        logits = model(img_batch, training=False)           # (B, H, W, C)
        preds  = tf.argmax(logits, axis=-1, output_type=tf.int32)  # (B, H, W)
        y_true_list.append(mask_batch.numpy())
        y_pred_list.append(preds.numpy())

    y_true = np.concatenate(y_true_list, axis=0)           # (N, H, W)
    y_pred = np.concatenate(y_pred_list, axis=0)           # (N, H, W)

    return y_true, y_pred

def plot_confusion_matrix(y_true, y_pred, save_path=None) -> None:
    """
    Plot a normalized confusion matrix over VOC classes.

    Void pixels (255) are excluded. Each row is normalized by the number
    of ground truth pixels for that class — shows recall per class.

    Parameters
    ----------
    y_true    : np.ndarray (N, H, W) int32 — ground truth class indices
    y_pred    : np.ndarray (N, H, W) int32 — predicted class indices
    save_path : str or Path, optional

    Returns
    -------
    None

    Example
    -------
    >>> plot_confusion_matrix(y_true, y_pred, save_path=PLOTS_DIR / "confusion_matrix.png")
    """
    valid = y_true != VOID_LABEL

    y_true_flat = y_true[valid]
    y_pred_flat = y_pred[valid]

    # Build confusion matrix
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.add.at(cm, (y_true_flat, y_pred_flat), 1)

    # Normalize row-wise — avoid division by zero for absent classes
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = np.where(row_sums > 0, cm / row_sums, 0).astype(np.float32)

    fig, ax = plt.subplots(figsize=(14, 12))

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Class name ticks
    ticks = np.arange(NUM_CLASSES)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(VOC_CLASSES, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(VOC_CLASSES, fontsize=7)

    # Cell annotations — only on diagonal and high-confusion cells to avoid clutter
    thresh = 0.3
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            val = cm_norm[i, j]
            if val >= thresh:
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=6,
                    color="white" if val > 0.6 else "black"
                )

    ax.set_xlabel("Predicted Class", fontsize=10)
    ax.set_ylabel("True Class",      fontsize=10)
    ax.set_title("Normalized Confusion Matrix", fontsize=12)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f" Saved → {save_path}")

    plt.show()

def plot_per_class_iou(y_true, y_pred, save_path=None) -> None:
    """
    Plot per-class IoU as a horizontal bar chart, sorted by IoU score.

    Classes absent from ground truth are excluded. Mean IoU across
    present classes is shown as a vertical reference line.

    Parameters
    ----------
    y_true    : np.ndarray (N, H, W) int32 — ground truth class indices
    y_pred    : np.ndarray (N, H, W) int32 — predicted class indices
    save_path : str or Path, optional

    Returns
    -------
    None

    Example
    -------
    >>> plot_per_class_iou(y_true, y_pred, save_path=PLOTS_DIR / "per_class_iou.png")
    """
    valid       = y_true != VOID_LABEL
    y_true_flat = y_true[valid]
    y_pred_flat = y_pred[valid]

    # Build confusion matrix
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    np.add.at(cm, (y_true_flat, y_pred_flat), 1)

    # Per-class IoU
    tp       = np.diag(cm)
    fp       = cm.sum(axis=0) - tp
    fn       = cm.sum(axis=1) - tp
    iou      = np.where(tp + fp + fn > 0, tp / (tp + fp + fn), np.nan)

    # Filter absent classes
    present  = cm.sum(axis=1) > 0
    iou_vals = iou[present]
    names    = [VOC_CLASSES[i] for i in range(NUM_CLASSES) if present[i]]

    # Sort by IoU
    order     = np.argsort(iou_vals)
    iou_vals  = iou_vals[order]
    names     = [names[i] for i in order]
    mean_iou  = np.nanmean(iou_vals)

    # Color — green if above mean, red if below
    colors = ["#2ecc71" if v >= mean_iou else "#e74c3c" for v in iou_vals]

    fig, ax = plt.subplots(figsize=(9, max(5, len(names) * 0.4)))

    bars = ax.barh(names, iou_vals, color=colors, alpha=0.85,
                   edgecolor="white", linewidth=0.5)

    # Value labels
    for bar, val in zip(bars, iou_vals):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center", fontsize=8
        )

    # Mean IoU reference line
    ax.axvline(mean_iou, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(
        mean_iou + 0.005, len(names) - 0.5,
        f"mIoU {mean_iou:.3f}",
        fontsize=8, color="black"
    )

    ax.set_xlim(0, 1.05)
    ax.set_xlabel("IoU", fontsize=10)
    ax.set_title("Per-Class IoU", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f" Saved → {save_path}")

    plt.show()

def plot_qualitative_results(model, dataset, n_rows=4, save_path=None) -> None:
    """
    Plot a grid of image / ground truth mask / predicted mask triplets.

    Each row shows one sample — left: original image, middle: ground truth
    mask, right: predicted mask. Useful for visually inspecting model
    behavior on specific samples.

    Parameters
    ----------
    model     : tf.keras.Model  — trained segmentation model
    dataset   : tf.data.Dataset — yields (image, mask) batches
    n_rows    : int             — number of samples to display (default: 4)
    save_path : str or Path, optional

    Returns
    -------
    None

    Example
    -------
    >>> plot_qualitative_results(model, val_ds, n_rows=4,
    ...                          save_path=PLOTS_DIR / "qualitative.png")
    """
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    images, gt_masks, pred_masks = [], [], []

    for img_batch, mask_batch in dataset:
        logits = model(img_batch, training=False)
        preds  = tf.argmax(logits, axis=-1, output_type=tf.int32).numpy()

        for img, gt, pred in zip(img_batch.numpy(), mask_batch.numpy(), preds):
            images.append(img)
            gt_masks.append(gt)
            pred_masks.append(pred)
            if len(images) == n_rows:
                break
        if len(images) == n_rows:
            break

    fig, axes = plt.subplots(n_rows, 3, figsize=(11, n_rows * 3))
    axes      = np.atleast_2d(axes)

    col_titles = ["Image", "Ground Truth", "Prediction"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=10)

    for i, (img, gt, pred) in enumerate(zip(images, gt_masks, pred_masks)):
        img_show = np.clip(img * std + mean, 0, 1)

        gt_rgb            = VOC_PALETTE[np.where(gt   == 255, 0, gt)]
        gt_rgb[gt == 255] = 0

        pred_rgb = VOC_PALETTE[pred]

        axes[i, 0].imshow(img_show)
        axes[i, 1].imshow(gt_rgb)
        axes[i, 2].imshow(pred_rgb)

        for j in range(3):
            axes[i, j].axis("off")

    plt.suptitle("Qualitative Results", fontsize=12, y=1.01)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f" Saved → {save_path}")

    plt.show()

# ─────────────────────────────────────────────
# 8. Utilities
# ─────────────────────────────────────────────

def label_to_rgb(mask) -> np.ndarray:
    """
    Convert an integer segmentation mask to an RGB image using VOC palette.

    Void pixels (255) are shown as black.

    Parameters
    ----------
    mask : np.ndarray (H, W) int32 — class indices, 255 = void

    Returns
    -------
    np.ndarray (H, W, 3) uint8 — RGB image

    Example
    -------
    >>> rgb = label_to_rgb(mask)
    >>> plt.imshow(rgb)
    """
    rgb          = VOC_PALETTE[np.where(mask == 255, 0, mask)]
    rgb[mask == 255] = 0
    return rgb


def get_best_model_path(ckpt_root, project, experiment_name) -> Path:
    """
    Return the path to the best saved model for a given experiment.

    Parameters
    ----------
    ckpt_root       : str or Path — root checkpoint directory
    project         : str         — project name (e.g. 'pascal_voc_segmentation')
    experiment_name : str         — experiment name (e.g. 'unet')

    Returns
    -------
    Path — path to {experiment_name}_best.keras

    Raises
    ------
    FileNotFoundError
        If the best model file does not exist.

    Example
    -------
    >>> path = get_best_model_path(CKPT_ROOT, PROJECT, "unet")
    """
    path = Path(ckpt_root) / project / experiment_name / f"{experiment_name}_best.keras"
    if not path.exists():
        raise FileNotFoundError(
            f"No best model found at {path}\n"
            f"  Make sure training has completed at least one epoch."
        )
    return path


