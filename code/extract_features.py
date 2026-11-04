"""
Frozen-backbone feature extraction over the Pang thyroid cytology cohort.

Loads each timm architecture with LOCAL weights (no network: huggingface.co and
download.pytorch.org are blocked in this environment), runs a forward pass with
num_classes=0 to obtain pooled embeddings, and caches them to .npz.

Each backbone is run at its own native input resolution, taken from its timm
default_cfg. Forcing a common resolution would be wrong for the window-based
models (swin_tiny is tied to 224 by its window partitioning), so native
resolution is both the correct and the faster choice. This is a documented
deviation from the source paper, which resampled everything to 384.

Resumable: an existing output .npz for a (model, variant) pair is skipped, so
the job can be run in chunks that fit inside a single tool call.
"""
import os, sys, json, csv, time, argparse
import numpy as np
import torch, timm
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(max(1, os.cpu_count() or 2))

MODELS = [
    "resnet50.a1_in1k",
    "densenet121.ra_in1k",
    "tf_efficientnetv2_s.in21k_ft_in1k",
    "convnext_tiny.fb_in22k_ft_in1k",
    "convnextv2_tiny.fcmae_ft_in22k_in1k",
    "swin_tiny_patch4_window7_224.ms_in1k",
    "swinv2_tiny_window8_256.ms_in1k",
    "vit_base_patch16_224.augreg2_in21k_ft_in1k",
    "deit3_small_patch16_224.fb_in22k_ft_in1k",
    "maxvit_tiny_tf_224.in1k",
    "coatnet_0_rw_224.sw_in1k",
    "efficientformerv2_s2.snap_dist_in1k",
]


def build(name, weights_dir):
    """Create the architecture untrained, then load the locally staged checkpoint."""
    model = timm.create_model(name, pretrained=False, num_classes=0)
    sd = torch.load(os.path.join(weights_dir, f"{name}.pth"),
                    map_location="cpu", weights_only=True)
    sd = {k: v for k, v in sd.items() if not k.startswith(("head.fc", "head.l", "fc.", "classifier."))}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    real_missing = [k for k in missing if not k.startswith(("head.", "fc.", "classifier."))]
    if real_missing:
        raise RuntimeError(f"{name}: {len(real_missing)} missing keys, first={real_missing[:3]}")
    model.eval()
    cfg = model.default_cfg
    return model, cfg


def loader(files, img_dir, size, mean, std, bs):
    mean = np.array(mean, dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array(std, dtype=np.float32).reshape(1, 3, 1, 1)
    for i in range(0, len(files), bs):
        chunk = files[i:i + bs]
        arr = np.empty((len(chunk), size, size, 3), dtype=np.uint8)
        for j, f in enumerate(chunk):
            im = Image.open(os.path.join(img_dir, f)).convert("RGB")
            if im.size != (size, size):
                im = im.resize((size, size), Image.BICUBIC)
            arr[j] = np.asarray(im)
        x = arr.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        x = (x - mean) / std
        yield torch.from_numpy(x)


def run(name, variant, args, files):
    out_path = os.path.join(args.out, f"{variant}__{name}.npz")
    if os.path.exists(out_path) and not args.force:
        print(f"  SKIP {variant}/{name} (cached)")
        return None
    t0 = time.time()
    model, cfg = build(name, args.weights)
    size = cfg["input_size"][-1]
    img_dir = os.path.join(args.data, variant)
    feats = []
    with torch.no_grad():
        for xb in loader(files, img_dir, size, cfg["mean"], cfg["std"], args.batch):
            feats.append(model(xb).float().numpy())
    F = np.concatenate(feats, 0)
    np.savez_compressed(out_path, features=F.astype(np.float32),
                        files=np.array(files), model=name, variant=variant,
                        input_size=size)
    dt = time.time() - t0
    print(f"  DONE {variant}/{name}  dim={F.shape[1]} size={size} "
          f"{dt:.0f}s ({dt/len(files)*1000:.0f} ms/img)")
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--weights", default=os.path.join(ROOT, "weights"))
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "features"))
    ap.add_argument("--manifest", default=os.path.join(ROOT, "manifest.csv"))
    ap.add_argument("--variant", default="plain")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=0, help="debug: only N images")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(args.manifest) as fh:
        files = [r["file"] for r in csv.DictReader(fh)]
    if args.limit:
        files = files[:args.limit]
    todo = args.models or MODELS
    print(f"variant={args.variant}  images={len(files)}  models={len(todo)}  threads={torch.get_num_threads()}")
    for m in todo:
        try:
            run(m, args.variant, args, files)
        except Exception as e:
            print(f"  FAIL {args.variant}/{m}: {type(e).__name__}: {str(e)[:160]}")


if __name__ == "__main__":
    main()
