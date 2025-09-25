#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Optional, Dict, Tuple
from PIL import Image, ImageDraw, ImageFont
import re


def parse_sample_index_and_step(filename: str) -> Optional[Tuple[int, int]]:
    """Extract (index, step) from filename. Returns (index, step) or None."""
    patterns = [r"_(\d+)_(\d+)\.(png|jpg)$", r"_(\d+)\.(png|jpg)$"]
    for pat in patterns:
        m = re.search(pat, filename)
        if m:
            groups = m.groups()
            if len(groups) >= 3:  # Has step and index
                return (int(groups[-2]), int(groups[-3]))  # (index, step)
            elif len(groups) >= 2:  # Just index
                return (int(groups[-2]), 0)  # (index, step=0)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, help="Folder with generated samples")
    ap.add_argument("--test", required=True, help="Folder with source test images (inputs)")
    ap.add_argument("--gt", required=True, help="Folder with ground-truth test_results")
    ap.add_argument("--out", required=True, help="Output folder for side-by-side/triptych PNGs")
    args = ap.parse_args()

    samples_dir = Path(args.samples)
    test_dir = Path(args.test)
    gt_dir = Path(args.gt)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_files = sorted(list(samples_dir.glob("*.png")) + list(samples_dir.glob("*.jpg")))
    if not sample_files:
        print(f"No sample images found in {samples_dir}")
        return

    def draw_label(img: Image.Image, text: str) -> Image.Image:
        w, h = img.size
        band_h = max(60, h // 16)  # Larger band height
        band = Image.new("RGB", (w, band_h), (20, 20, 20))
        out = Image.new("RGB", (w, h + band_h), (0, 0, 0))
        out.paste(band, (0, 0))
        out.paste(img, (0, band_h))
        draw = ImageDraw.Draw(out)
        try:
            # Try to load a larger font
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
        except Exception:
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None
        draw.text((10, (band_h - 24) // 2), text, fill=(240, 240, 240), font=font)
        return out

    # Group samples by test index and find latest iteration for each
    latest_samples: Dict[int, Path] = {}  # test_index -> sample_path
    
    for sp in sample_files:
        result = parse_sample_index_and_step(sp.name)
        if result is None:
            continue
        idx, step = result
        
        # Keep only the latest iteration for each test index
        if idx not in latest_samples:
            latest_samples[idx] = sp
        else:
            # Compare steps to keep the latest
            current_result = parse_sample_index_and_step(latest_samples[idx].name)
            if current_result and step > current_result[1]:
                latest_samples[idx] = sp
    
    print(f"Found latest iterations for {len(latest_samples)} test indices")
    
    made = 0
    for idx, sp in latest_samples.items():
        src_path = test_dir / f"test_{idx:03d}.png"
        gt_path = gt_dir / f"test_{idx:03d}.png"
        if not (src_path.exists() and gt_path.exists()):
            continue
        # load
        gen = Image.open(sp).convert("RGB")
        src = Image.open(src_path).convert("RGB")
        gt = Image.open(gt_path).convert("RGB")
        # resize to gt size if needed
        if gen.size != gt.size:
            gen = gen.resize(gt.size, Image.BICUBIC)
        if src.size != gt.size:
            src = src.resize(gt.size, Image.BICUBIC)
        # add labels
        src_l = draw_label(src, f"source (test_{idx:03d}.png)")
        gen_l = draw_label(gen, "target (generated)")
        gt_l = draw_label(gt, f"ground truth (test_{idx:03d}.png)")
        # concat triptych
        w, h = src_l.size
        canvas = Image.new("RGB", (w * 3, h), (0, 0, 0))
        canvas.paste(src_l, (0, 0))
        canvas.paste(gen_l, (w, 0))
        canvas.paste(gt_l, (2 * w, 0))
        out_path = out_dir / f"triptych_{idx:03d}.png"
        canvas.save(out_path)
        made += 1

    print(f"Wrote {made} triptychs to {out_dir}")


if __name__ == "__main__":
    main()


