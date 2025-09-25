#!/usr/bin/env python3
"""
analyze_samples.py

Compare generated samples against ground-truth test_results.
Maps samples by filename index (e.g., ..._0.png -> test_000.png).
Supports multiple iterations (e.g., ..._0000_0.png, ..._0250_0.png).
Outputs metrics CSV, summary, and optional diff visualizations.
"""

import os
import re
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics import MeanSquaredError as MSE
from torchmetrics import MeanAbsoluteError as MAE
from lpips import LPIPS

# Optional: for diff visualizations
try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def parse_sample_index_and_step(filename: str) -> Optional[Tuple[int, int]]:
    """Extract trailing index and step from sample filename.
    Examples:
        'flux_kontext_fludyn_lora_0000_0.png' -> (0, 0)
        'flux_kontext_fludyn_lora_0250_12.png' -> (12, 250)
        'sample_5.png' -> (5, None)
    Returns (index, step) or None if parsing fails.
    """
    # Try patterns: ..._XXXX_N.png/jpg or ..._N.png/jpg
    patterns = [
        r'_(\d+)_(\d+)\.(png|jpg)$',  # ..._XXXX_N.png/jpg
        r'_(\d+)\.(png|jpg)$',        # ..._N.png/jpg
    ]
    for pattern in patterns:
        m = re.search(pattern, filename)
        if m:
            groups = m.groups()
            if len(groups) >= 3:  # Has step and index
                return (int(groups[-2]), int(groups[-3]))  # (index, step)
            elif len(groups) >= 2:  # Just index
                return (int(groups[-2]), None)  # (index, None)
    return None


def parse_sample_index(filename: str) -> Optional[int]:
    """Extract trailing index from sample filename (backward compatibility)."""
    result = parse_sample_index_and_step(filename)
    return result[0] if result else None


def find_ground_truth_path(sample_index: int, gt_dir: Path) -> Optional[Path]:
    """Map sample index to ground-truth filename.
    Example: 0 -> test_000.png, 12 -> test_012.png
    """
    gt_name = f"test_{sample_index:03d}.png"
    gt_path = gt_dir / gt_name
    return gt_path if gt_path.exists() else None


def load_image_as_rgb(path: Union[str, Path]) -> np.ndarray:
    """Load image as RGB, normalized to [0,1]."""
    img = Image.open(path).convert('RGB')
    return np.array(img, dtype=np.float32) / 255.0


def resize_to_match(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize image to target size using bilinear interpolation."""
    if img.shape[:2] == (target_h, target_w):
        return img
    # Use PIL for consistent resizing
    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    pil_resized = pil_img.resize((target_w, target_h), Image.BILINEAR)
    return np.array(pil_resized, dtype=np.float32) / 255.0


def compute_metrics(img1: np.ndarray, img2: np.ndarray, device: str = 'cpu') -> Dict[str, float]:
    """Compute MSE, PSNR, SSIM, LPIPS, MAE between two RGB images.
    Images should be [H,W,3] in [0,1].
    """
    # Convert to torch tensors [1,3,H,W]
    t1 = torch.from_numpy(img1).permute(2,0,1).unsqueeze(0).to(device)
    t2 = torch.from_numpy(img2).permute(2,0,1).unsqueeze(0).to(device)

    # MSE
    mse = F.mse_loss(t1, t2).item()

    # PSNR
    psnr = PSNR(data_range=1.0).to(device)(t1, t2).item()

    # SSIM
    ssim = SSIM().to(device)(t1, t2).item()

    # MAE
    mae = F.l1_loss(t1, t2).item()

    # LPIPS (Alex)
    lpips_model = LPIPS(net='alex').to(device)
    lpips_val = lpips_model(t1, t2).item()

    return {
        'mse': mse,
        'psnr': psnr,
        'ssim': ssim,
        'mae': mae,
        'lpips': lpips_val,
    }


def save_diff_visualization(
    sample: np.ndarray,
    gt: np.ndarray,
    metrics: Dict[str, float],
    out_dir: Path,
    sample_name: str,
    gt_name: str,
) -> None:
    """Save side-by-side triptych and heatmap if matplotlib is available."""
    if not HAS_MATPLOTLIB:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Absolute difference heatmap
    diff = np.abs(sample - gt)
    diff_heat = np.mean(diff, axis=2)  # [H,W]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'{sample_name} vs {gt_name}\nSSIM={metrics["ssim"]:.4f}, LPIPS={metrics["lpips"]:.4f}')

    axes[0].imshow(sample)
    axes[0].set_title('Generated')
    axes[0].axis('off')

    axes[1].imshow(gt)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')

    im = axes[2].imshow(diff_heat, cmap='jet', vmin=0, vmax=1)
    axes[2].set_title('|Diff|')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_dir / f'diff_{sample_name}_{gt_name}.png', dpi=150, bbox_inches='tight')
    plt.close()


def analyze_samples(
    samples_dir: Path,
    gt_dir: Path,
    out_dir: Path,
    device: str = 'cpu',
    save_diffs: bool = False,
) -> None:
    """Main analysis function."""
    samples_dir = Path(samples_dir)
    gt_dir = Path(gt_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find all sample files (PNG and JPG)
    sample_files = list(samples_dir.glob('*.png')) + list(samples_dir.glob('*.jpg'))
    if not sample_files:
        print(f"No PNG or JPG files found in {samples_dir}")
        return

    print(f"Found {len(sample_files)} sample files")

    # Build mapping: iteration -> {sample_index -> (sample_path, gt_path)}
    iteration_mapping: Dict[int, Dict[int, Tuple[Path, Path]]] = {}
    skipped: List[str] = []
    unmatched: List[str] = []

    for sample_path in sample_files:
        result = parse_sample_index_and_step(sample_path.name)
        if result is None:
            unmatched.append(sample_path.name)
            continue
        
        sample_index, step = result
        if step is None:
            step = 0  # Default to step 0 if no step found
        
        gt_path = find_ground_truth_path(sample_index, gt_dir)
        if gt_path is None:
            skipped.append(f"{sample_path.name} -> test_{sample_index:03d}.png (missing)")
            continue

        if step not in iteration_mapping:
            iteration_mapping[step] = {}
        iteration_mapping[step][sample_index] = (sample_path, gt_path)

    total_pairs = sum(len(mapping) for mapping in iteration_mapping.values())
    print(f"Mapped {total_pairs} pairs across {len(iteration_mapping)} iterations")
    if skipped:
        print(f"Skipped {len(skipped)} (missing GT)")
    if unmatched:
        print(f"Unmatched {len(unmatched)} (bad filename)")

    # Process each iteration separately
    all_rows: List[Dict] = []
    resized_log: List[str] = []
    
    for step in sorted(iteration_mapping.keys()):
        mapping = iteration_mapping[step]
        print(f"\nProcessing iteration {step} ({len(mapping)} pairs)...")
        
        iteration_rows: List[Dict] = []
        
        for sample_index, (sample_path, gt_path) in sorted(mapping.items()):
            try:
                # Load images
                sample_img = load_image_as_rgb(sample_path)
                gt_img = load_image_as_rgb(gt_path)

                # Resize if needed
                h, w = gt_img.shape[:2]
                if sample_img.shape[:2] != (h, w):
                    resized_log.append(f"{sample_path.name}: {sample_img.shape[:2]} -> {(h,w)}")
                    sample_img = resize_to_match(sample_img, h, w)

                # Compute metrics
                metrics = compute_metrics(sample_img, gt_img, device)

                # Record
                row = {
                    'iteration': step,
                    'id': sample_index,
                    'sample_path': str(sample_path),
                    'gt_path': str(gt_path),
                    'width': w,
                    'height': h,
                    **metrics,
                }
                iteration_rows.append(row)
                all_rows.append(row)

                # Save diff visualization if requested
                if save_diffs:
                    save_diff_visualization(
                        sample_img, gt_img, metrics,
                        out_dir / f'diffs_iter_{step}',
                        sample_path.stem, gt_path.stem,
                    )

            except Exception as e:
                print(f"Error processing {sample_path.name}: {e}")
                continue
        
        # Write iteration-specific CSV
        if iteration_rows:
            iter_csv_path = out_dir / f'metrics_iter_{step}.csv'
            with open(iter_csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=iteration_rows[0].keys())
                writer.writeheader()
                writer.writerows(iteration_rows)
            print(f"  Wrote {len(iteration_rows)} pairs to {iter_csv_path}")

    if not all_rows:
        print("No valid pairs processed")
        return

    # Write combined CSV
    csv_path = out_dir / 'metrics_all_iterations.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    # Compute aggregates for each iteration and overall
    metrics_keys = ['mse', 'psnr', 'ssim', 'mae', 'lpips']
    
    # Overall summary
    overall_summary = {}
    for key in metrics_keys:
        values = [r[key] for r in all_rows]
        overall_summary[key] = {
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
        }
    
    # Per-iteration summaries
    iteration_summaries = {}
    for step in sorted(iteration_mapping.keys()):
        step_rows = [r for r in all_rows if r['iteration'] == step]
        iteration_summaries[step] = {}
        for key in metrics_keys:
            values = [r[key] for r in step_rows]
            iteration_summaries[step][key] = {
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
            }

    # Write summary
    summary_path = out_dir / 'summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f"Analysis of {len(all_rows)} sample pairs across {len(iteration_mapping)} iterations\n")
        f.write(f"Samples: {samples_dir}\n")
        f.write(f"Ground truth: {gt_dir}\n\n")

        # Overall summary
        f.write("=== OVERALL SUMMARY ===\n")
        for key in metrics_keys:
            s = overall_summary[key]
            f.write(f"{key.upper()}:\n")
            f.write(f"  mean:   {s['mean']:.6f}\n")
            f.write(f"  median: {s['median']:.6f}\n")
            f.write(f"  std:    {s['std']:.6f}\n")
            f.write(f"  min:    {s['min']:.6f}\n")
            f.write(f"  max:    {s['max']:.6f}\n\n")

        # Per-iteration summaries
        for step in sorted(iteration_mapping.keys()):
            f.write(f"=== ITERATION {step} SUMMARY ===\n")
            for key in metrics_keys:
                s = iteration_summaries[step][key]
                f.write(f"{key.upper()}:\n")
                f.write(f"  mean:   {s['mean']:.6f}\n")
                f.write(f"  median: {s['median']:.6f}\n")
                f.write(f"  std:    {s['std']:.6f}\n")
                f.write(f"  min:    {s['min']:.6f}\n")
                f.write(f"  max:    {s['max']:.6f}\n\n")

        # Top-5 best/worst by SSIM and LPIPS (overall)
        rows_sorted_ssim = sorted(all_rows, key=lambda r: r['ssim'], reverse=True)
        rows_sorted_lpips = sorted(all_rows, key=lambda r: r['lpips'])

        f.write("=== TOP-5 BEST SSIM (OVERALL) ===\n")
        for i, r in enumerate(rows_sorted_ssim[:5]):
            f.write(f"  {i+1}. iter={r['iteration']:3d} id={r['id']:3d} SSIM={r['ssim']:.4f} LPIPS={r['lpips']:.4f}\n")

        f.write("\n=== TOP-5 WORST SSIM (OVERALL) ===\n")
        for i, r in enumerate(rows_sorted_ssim[-5:]):
            f.write(f"  {i+1}. iter={r['iteration']:3d} id={r['id']:3d} SSIM={r['ssim']:.4f} LPIPS={r['lpips']:.4f}\n")

        f.write("\n=== TOP-5 BEST LPIPS (OVERALL) ===\n")
        for i, r in enumerate(rows_sorted_lpips[:5]):
            f.write(f"  {i+1}. iter={r['iteration']:3d} id={r['id']:3d} LPIPS={r['lpips']:.4f} SSIM={r['ssim']:.4f}\n")

        f.write("\n=== TOP-5 WORST LPIPS (OVERALL) ===\n")
        for i, r in enumerate(rows_sorted_lpips[-5:]):
            f.write(f"  {i+1}. iter={r['iteration']:3d} id={r['id']:3d} LPIPS={r['lpips']:.4f} SSIM={r['ssim']:.4f}\n")

    # Write logs
    if skipped:
        with open(out_dir / 'skipped.txt', 'w') as f:
            f.write('\n'.join(skipped))
    if unmatched:
        with open(out_dir / 'unmatched.txt', 'w') as f:
            f.write('\n'.join(unmatched))
    if resized_log:
        with open(out_dir / 'resized.txt', 'w') as f:
            f.write('\n'.join(resized_log))

    print(f"Results written to {out_dir}")
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze generated samples against ground truth')
    parser.add_argument('--samples', required=True, help='Path to samples directory')
    parser.add_argument('--gt', required=True, help='Path to ground truth directory')
    parser.add_argument('--out', required=True, help='Output directory for analysis results')
    parser.add_argument('--device', default='cpu', help='Device for metrics computation (cpu/cuda)')
    parser.add_argument('--save-diffs', action='store_true', help='Save diff visualizations')

    args = parser.parse_args()

    analyze_samples(
        samples_dir=args.samples,
        gt_dir=args.gt,
        out_dir=args.out,
        device=args.device,
        save_diffs=args.save_diffs,
    )


if __name__ == '__main__':
    main()

