#!/usr/bin/env python3
"""
build_kontext_ds.py

Creates an Ostris AI-Toolkit Kontext dataset from a raw folder of CFD contour images.
Generates paired edit examples for fluid dynamics visualization transformations.
"""

import os
import re
import csv
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set
from PIL import Image

# Variable mapping: suffix -> token
VARIABLE_MAP = {
    'u_x': 'uxstar',
    'u_y': 'uystar', 
    'pressure': 'pressure',
    'nut': 'nut'
}

def get_variable_from_filename(filename: str) -> Optional[str]:
    """Extract variable from filename suffix."""
    stem = Path(filename).stem.lower()
    
    # Try exact matches first
    for suffix, var in VARIABLE_MAP.items():
        if stem.endswith('_' + suffix):
            return suffix
    
    # Fallback: check if any variable name appears at the end
    for suffix in VARIABLE_MAP.keys():
        if suffix in stem:
            return suffix
    
    return None

def get_case_id_from_filename(filename: str) -> str:
    """Extract case_id by removing variable suffix from filename stem."""
    stem = Path(filename).stem
    variable = get_variable_from_filename(filename)
    
    if variable:
        # Remove the variable suffix (e.g., "_u_x")
        suffix_to_remove = '_' + variable
        if stem.endswith(suffix_to_remove):
            return stem[:-len(suffix_to_remove)]
    
    return stem

def parse_metadata(stem: str) -> Dict:
    """Parse metadata from filename stem."""
    metadata = {
        'airfoil_id': None,
        'velocity': None,
        'aoa_deg': None,
        'variable': None
    }
    
    # Extract variable
    metadata['variable'] = get_variable_from_filename(stem)
    
    # Pattern to match airFoil2D_SST_velocity_aoa_...
    pattern = r'airFoil2D_SST_([\d.-]+)_([\d.-]+)_(.+)'
    match = re.search(pattern, stem)
    
    if match:
        try:
            metadata['velocity'] = float(match.group(1))
            metadata['aoa_deg'] = float(match.group(2))
            
            # Extract airfoil parameters to create airfoil_id (NACA-like)
            remaining = match.group(3)
            # Robustly strip trailing tokens like 'internal', variable names, etc.
            tokens = remaining.split('_')
            numeric_tokens: List[float] = []
            for tok in tokens:
                try:
                    numeric_tokens.append(float(tok))
                except Exception:
                    # skip non-numeric tokens like 'internal'
                    continue

            # Use the last 3 or 4 numeric tokens as the airfoil param set
            airfoil_params: Optional[List[float]] = None
            if len(numeric_tokens) >= 4:
                # Prefer 4 if available (5-digit family)
                airfoil_params = numeric_tokens[-4:]
            elif len(numeric_tokens) >= 3:
                airfoil_params = numeric_tokens[-3:]

            def format_naca(params: List[float]) -> str:
                # 4-digit: (camber, pos, thickness) -> NACA CC PPTT
                # 5-digit: (design_cl, pos, thickness, reflex) -> NACA D PP TT[_RR]
                if len(params) == 3:
                    camber, pos, thick = params
                    return f"NACA{int(camber):02d}{int(pos):02d}{int(thick):02d}"
                if len(params) == 4:
                    dcl, pos, thick, reflex = params
                    base = f"NACA{int(dcl):01d}{int(pos):02d}{int(thick):02d}"
                    if reflex and int(reflex) > 0:
                        base += f"_{int(reflex):02d}"
                    return base
                return "unknown"

            metadata['airfoil_id'] = format_naca(airfoil_params) if airfoil_params else None
            
        except ValueError:
            pass
    
    return metadata

def compute_bins(values: List[float], num_bins: int = 10) -> Dict[float, int]:
    """Compute bin indices for a list of values."""
    if not values:
        return {}
    
    min_val, max_val = min(values), max(values)
    if min_val == max_val:
        return {min_val: 0}
    
    bin_width = (max_val - min_val) / num_bins
    bins = {}
    
    for val in values:
        bin_idx = min(int((val - min_val) / bin_width), num_bins - 1)
        bins[val] = bin_idx
    
    return bins

def discover_images(raw_root: Path) -> Dict[Tuple[str, str], Dict]:
    """Discover and index all images by (case_id, variable)."""
    print(f"Discovering images in {raw_root}")
    
    image_index = {}
    all_velocities = []
    all_aoas = []
    
    # First pass: collect all images and metadata
    for img_path in raw_root.rglob('*.png'):
        if not img_path.is_file():
            continue
            
        variable = get_variable_from_filename(img_path.name)
        if not variable:
            continue
            
        case_id = get_case_id_from_filename(img_path.name)
        metadata = parse_metadata(img_path.stem)
        
        if metadata['velocity'] is not None:
            all_velocities.append(metadata['velocity'])
        if metadata['aoa_deg'] is not None:
            all_aoas.append(metadata['aoa_deg'])
        
        key = (case_id, variable)
        image_index[key] = {
            'path': img_path,
            'metadata': metadata,
            'case_id': case_id,
            'variable': variable
        }
    
    # Second pass: compute bins
    velocity_bins = compute_bins(all_velocities)
    aoa_bins = compute_bins(all_aoas)
    
    # Third pass: add bin information
    for key, entry in image_index.items():
        metadata = entry['metadata']
        entry['velocity_bin'] = velocity_bins.get(metadata['velocity'])
        entry['aoa_bin'] = aoa_bins.get(metadata['aoa_deg'])
    
    print(f"Found {len(image_index)} images across {len(set(k[0] for k in image_index))} cases")
    print(f"Variables found: {set(k[1] for k in image_index)}")
    
    return image_index

# Removed generate_variable_switch_pairs function as variable switching pairs
# are not useful for surrogate modeling (all variables computed together in CFD)

def generate_aoa_change_pairs(image_index: Dict) -> List[Dict]:
    """Generate pairs for AoA changes within same airfoil/variable."""
    pairs = []
    groups = defaultdict(list)
    
    # Group by (airfoil_id, variable)
    for entry in image_index.values():
        metadata = entry['metadata']
        airfoil_id = metadata.get('airfoil_id')
        variable = entry['variable']
        aoa = metadata.get('aoa_deg')
        
        if airfoil_id and aoa is not None:
            groups[(airfoil_id, variable)].append(entry)
    
    for (airfoil_id, variable), entries in groups.items():
        if len(entries) < 2:
            continue
            
        # Sort by AoA
        entries.sort(key=lambda x: x['metadata']['aoa_deg'])
        
        # Pair adjacent AoAs in both directions
        for i in range(len(entries) - 1):
            for direction in [0, 1]:  # Both directions
                entry_a = entries[i + direction]
                entry_b = entries[i + 1 - direction]
                
                metadata_a = entry_a['metadata']
                metadata_b = entry_b['metadata']
                
                # Check velocity bin compatibility if both have velocities
                if (metadata_a.get('velocity') is not None and 
                    metadata_b.get('velocity') is not None and
                    entry_a.get('velocity_bin') != entry_b.get('velocity_bin')):
                    continue
                
                # Build caption
                caption_parts = [
                    f"fludyn edit: set AoA from {metadata_a['aoa_deg']:.2f} to {metadata_b['aoa_deg']:.2f} deg"
                ]
                keep_parts = [f"airfoil={airfoil_id}", f"variable={variable}"]
                
                if metadata_a.get('velocity') is not None:
                    v_center = (metadata_a['velocity'] + metadata_b['velocity']) / 2
                    keep_parts.append(f"V≈{v_center:.3f}")
                
                caption_parts.append("keep " + ", ".join(keep_parts))
                caption = "; ".join(caption_parts)
                
                pairs.append({
                    'before_path': entry_a['path'],
                    'after_path': entry_b['path'],
                    'caption': caption,
                    'edit_type': 'aoa_change',
                    'airfoil_A': airfoil_id,
                    'airfoil_B': airfoil_id,
                    'aoa_A': metadata_a['aoa_deg'],
                    'aoa_B': metadata_b['aoa_deg'],
                    'vel_A': metadata_a.get('velocity'),
                    'vel_B': metadata_b.get('velocity'),
                    'var_A': variable,
                    'var_B': variable,
                    'aoa_bin': entry_a.get('aoa_bin'),
                    'vel_bin': entry_a.get('velocity_bin')
                })
    
    return pairs

def generate_velocity_change_pairs(image_index: Dict) -> List[Dict]:
    """Generate pairs for velocity changes within same airfoil/variable/aoa_bin."""
    pairs = []
    groups = defaultdict(list)
    
    # Group by (airfoil_id, variable, aoa_bin)
    for entry in image_index.values():
        metadata = entry['metadata']
        airfoil_id = metadata.get('airfoil_id')
        variable = entry['variable']
        aoa_bin = entry.get('aoa_bin')
        velocity = metadata.get('velocity')
        
        if airfoil_id and velocity is not None and aoa_bin is not None:
            groups[(airfoil_id, variable, aoa_bin)].append(entry)
    
    for (airfoil_id, variable, aoa_bin), entries in groups.items():
        if len(entries) < 2:
            continue
            
        # Sort by velocity
        entries.sort(key=lambda x: x['metadata']['velocity'])
        
        # Pair adjacent velocities in both directions
        for i in range(len(entries) - 1):
            for direction in [0, 1]:  # Both directions
                entry_a = entries[i + direction]
                entry_b = entries[i + 1 - direction]
                
                metadata_a = entry_a['metadata']
                metadata_b = entry_b['metadata']
                
                # Build caption
                caption_parts = [
                    f"fludyn edit: set V from {metadata_a['velocity']:.3f} to {metadata_b['velocity']:.3f}"
                ]
                keep_parts = [f"airfoil={airfoil_id}", f"variable={variable}"]
                
                if metadata_a.get('aoa_deg') is not None:
                    aoa_center = (metadata_a['aoa_deg'] + metadata_b['aoa_deg']) / 2
                    keep_parts.append(f"AoA={aoa_center:.2f} deg")
                
                caption_parts.append("keep " + ", ".join(keep_parts))
                caption = "; ".join(caption_parts)
                
                pairs.append({
                    'before_path': entry_a['path'],
                    'after_path': entry_b['path'],
                    'caption': caption,
                    'edit_type': 'velocity_change',
                    'airfoil_A': airfoil_id,
                    'airfoil_B': airfoil_id,
                    'aoa_A': metadata_a['aoa_deg'],
                    'aoa_B': metadata_b['aoa_deg'],
                    'vel_A': metadata_a['velocity'],
                    'vel_B': metadata_b['velocity'],
                    'var_A': variable,
                    'var_B': variable,
                    'aoa_bin': aoa_bin,
                    'vel_bin': entry_a.get('velocity_bin')
                })
    
    return pairs

def generate_airfoil_change_pairs(image_index: Dict) -> List[Dict]:
    """Generate pairs for airfoil changes within same conditions."""
    pairs = []
    groups = defaultdict(list)
    
    # Group by (aoa_bin, velocity_bin, variable)
    for entry in image_index.values():
        metadata = entry['metadata']
        airfoil_id = metadata.get('airfoil_id')
        variable = entry['variable']
        aoa_bin = entry.get('aoa_bin')
        vel_bin = entry.get('velocity_bin')
        
        if airfoil_id and aoa_bin is not None and vel_bin is not None:
            groups[(aoa_bin, vel_bin, variable)].append(entry)
    
    for (aoa_bin, vel_bin, variable), entries in groups.items():
        # Group by airfoil within this condition
        airfoil_groups = defaultdict(list)
        for entry in entries:
            airfoil_id = entry['metadata']['airfoil_id']
            airfoil_groups[airfoil_id].append(entry)
        
        airfoils = list(airfoil_groups.keys())
        if len(airfoils) < 2:
            continue
        
        # Pair different airfoils
        for i, airfoil_a in enumerate(airfoils):
            for j, airfoil_b in enumerate(airfoils):
                if i >= j:
                    continue
                    
                entries_a = airfoil_groups[airfoil_a]
                entries_b = airfoil_groups[airfoil_b]
                
                # Take one representative from each airfoil
                for entry_a in entries_a[:1]:  # Limit to avoid explosion
                    for entry_b in entries_b[:1]:
                        metadata_a = entry_a['metadata']
                        metadata_b = entry_b['metadata']
                        
                        # Build caption with approximate values
                        aoa_center = None
                        vel_center = None
                        
                        if (metadata_a.get('aoa_deg') is not None and 
                            metadata_b.get('aoa_deg') is not None):
                            aoa_center = (metadata_a['aoa_deg'] + metadata_b['aoa_deg']) / 2
                        
                        if (metadata_a.get('velocity') is not None and 
                            metadata_b.get('velocity') is not None):
                            vel_center = (metadata_a['velocity'] + metadata_b['velocity']) / 2
                        
                        caption_parts = [f"fludyn edit: change airfoil from {airfoil_a} to {airfoil_b}"]
                        keep_parts = [f"variable={variable}"]
                        
                        if aoa_center is not None:
                            keep_parts.append(f"AoA≈{aoa_center:.2f}")
                        if vel_center is not None:
                            keep_parts.append(f"V≈{vel_center:.3f}")
                        
                        caption_parts.append("keep " + ", ".join(keep_parts))
                        caption = "; ".join(caption_parts)
                        
                        # Both directions
                        for direction in [(entry_a, entry_b, airfoil_a, airfoil_b),
                                        (entry_b, entry_a, airfoil_b, airfoil_a)]:
                            e_from, e_to, af_from, af_to = direction
                            meta_from = e_from['metadata']
                            meta_to = e_to['metadata']
                            
                            dir_caption = caption.replace(f"from {airfoil_a} to {airfoil_b}",
                                                        f"from {af_from} to {af_to}")
                            
                            pairs.append({
                                'before_path': e_from['path'],
                                'after_path': e_to['path'],
                                'caption': dir_caption,
                                'edit_type': 'airfoil_change',
                                'airfoil_A': af_from,
                                'airfoil_B': af_to,
                                'aoa_A': meta_from.get('aoa_deg'),
                                'aoa_B': meta_to.get('aoa_deg'),
                                'vel_A': meta_from.get('velocity'),
                                'vel_B': meta_to.get('velocity'),
                                'var_A': variable,
                                'var_B': variable,
                                'aoa_bin': aoa_bin,
                                'vel_bin': vel_bin
                            })
    
    return pairs

def generate_identity_pairs(image_index: Dict, num_pairs: int) -> List[Dict]:
    """Generate identity pairs for stability."""
    pairs = []
    entries = list(image_index.values())
    random.shuffle(entries)
    
    for entry in entries[:num_pairs]:
        metadata = entry['metadata']
        caption = "fludyn edit: keep all conditions unchanged; preserve colors and crop"
        
        pairs.append({
            'before_path': entry['path'],
            'after_path': entry['path'],
            'caption': caption,
            'edit_type': 'identity',
            'airfoil_A': metadata.get('airfoil_id'),
            'airfoil_B': metadata.get('airfoil_id'),
            'aoa_A': metadata.get('aoa_deg'),
            'aoa_B': metadata.get('aoa_deg'),
            'vel_A': metadata.get('velocity'),
            'vel_B': metadata.get('velocity'),
            'var_A': entry['variable'],
            'var_B': entry['variable'],
            'aoa_bin': entry.get('aoa_bin'),
            'vel_bin': entry.get('velocity_bin')
        })
    
    return pairs

def check_image_compatibility(path_a: Path, path_b: Path) -> bool:
    """Check if two images have compatible sizes."""
    try:
        with Image.open(path_a) as img_a, Image.open(path_b) as img_b:
            return img_a.size == img_b.size
    except Exception:
        return False

def write_kontext_yaml(out_root: Path, test_files: List[Path], prompts: List[str]) -> None:
    """Write an Ostris Toolkit Kontext LoRA YAML config customized to OUT_ROOT and test set.

    - Captions already include the 'fludyn' trigger phrase.
    - folder_path points to after/ with per-image captions placed alongside images.
    - control_path points to before/.
    - sample.prompts length matches number of test images, each with a --ctrl_img path.
    """
    yaml_path = out_root / "kontext_lora_config.yaml"

    # Build prompt lines matching test set size
    prompt_lines: List[str] = []
    for i, test_img in enumerate(test_files):
        prompt_text = prompts[i % len(prompts)].replace("  ", " ")
        prompt_lines.append(f"          - \"{prompt_text}  --ctrl_img {str(test_img)}\"")

    content = f"""job: extension
config:
  name: "flux_kontext_fludyn_lora"
  process:
    - type: 'sd_trainer'
      training_folder: "output"
      device: cuda:0
      trigger_word: "fludyn"
      network:
        type: "lora"
        linear: 16
        linear_alpha: 16
      save:
        dtype: float16
        save_every: 250
        max_step_saves_to_keep: 4
        push_to_hub: false
      datasets:
        - folder_path: "{str(out_root / 'after')}"
          control_path: "{str(out_root / 'before')}"
          caption_ext: "txt"
          caption_dropout_rate: 0.05
          shuffle_tokens: false
          cache_latents_to_disk: true
          resolution: [ 512, 768 ]
      train:
        batch_size: 1
        steps: 3000
        gradient_accumulation_steps: 1
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        noise_scheduler: "flowmatch"
        optimizer: "adamw8bit"
        lr: 1e-4
        timestep_type: "weighted"
        dtype: bf16
      model:
        name_or_path: "black-forest-labs/FLUX.1-Kontext-dev"
        arch: "flux_kontext"
        quantize: true
      sample:
        sampler: "flowmatch"
        sample_every: 250
        width: 1024
        height: 1024
        prompts:
{os.linesep.join(prompt_lines)}
        neg: ""
        seed: 42
        walk_seed: true
        guidance_scale: 4
        sample_steps: 20
meta:
  name: "[name]"
  version: '1.0'"""

    with open(yaml_path, "w") as f:
        f.write(content)


def materialize_dataset(pairs: List[Dict], out_root: Path, test_size: int = 32) -> None:
    """Create the final dataset structure."""
    print(f"Materializing dataset to {out_root}")
    
    # Create directories
    before_dir = out_root / "before"
    after_dir = out_root / "after"
    captions_dir = out_root / "captions"  # legacy/compat
    test_dir = out_root / "test"
    
    for dir_path in [before_dir, after_dir, captions_dir, test_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Filter compatible pairs
    valid_pairs = []
    for pair in pairs:
        if check_image_compatibility(pair['before_path'], pair['after_path']):
            valid_pairs.append(pair)
        else:
            print(f"Skipping incompatible pair: {pair['before_path'].name} -> {pair['after_path'].name}")
    
    print(f"Valid pairs after compatibility check: {len(valid_pairs)}")
    
    # Copy files and create captions
    csv_data = []
    test_candidates = []
    all_pair_captions: List[str] = []
    
    for i, pair in enumerate(valid_pairs, 1):
        pair_id = f"{i:06d}"
        
        # Copy images (try hardlink first, fallback to copy)
        before_dst = before_dir / f"{pair_id}.png"
        after_dst = after_dir / f"{pair_id}.png"
        
        for src, dst in [(pair['before_path'], before_dst), (pair['after_path'], after_dst)]:
            try:
                os.link(src, dst)
            except (OSError, NotImplementedError):
                shutil.copy2(src, dst)
        
        # Write caption next to after image (Ostris expects per-image captions alongside images)
        after_caption_file = after_dir / f"{pair_id}.txt"
        with open(after_caption_file, 'w') as f:
            f.write(pair['caption'])
        # Also write to captions/ for convenience
        legacy_caption_file = captions_dir / f"{pair_id}.txt"
        with open(legacy_caption_file, 'w') as f:
            f.write(pair['caption'])
        
        # Collect for CSV
        csv_data.append({
            'id': pair_id,
            'before_path': str(pair['before_path']),
            'after_path': str(pair['after_path']),
            'caption': pair['caption'],
            'edit_type': pair['edit_type'],
            'airfoil_A': pair.get('airfoil_A'),
            'airfoil_B': pair.get('airfoil_B'),
            'aoa_A': pair.get('aoa_A'),
            'aoa_B': pair.get('aoa_B'),
            'vel_A': pair.get('vel_A'),
            'vel_B': pair.get('vel_B'),
            'var_A': pair.get('var_A'),
            'var_B': pair.get('var_B'),
            'aoa_bin': pair.get('aoa_bin'),
            'vel_bin': pair.get('vel_bin')
        })
        
        # Collect test candidates (diverse edit types) with mapping to target
        if pair['edit_type'] != 'identity':
            test_candidates.append({
                'before_path': before_dst,
                'after_path': after_dst,
                'edit_type': pair['edit_type'],
                'caption': pair['caption']
            })
        all_pair_captions.append(pair['caption'])
    
    # Create test set
    test_by_type = defaultdict(list)
    for item in test_candidates:
        test_by_type[item['edit_type']].append(item)
    
    test_files = []
    types_available = list(test_by_type.keys())
    for i in range(test_size):
        if types_available:
            edit_type = types_available[i % len(types_available)]
            if test_by_type[edit_type]:
                test_files.append(test_by_type[edit_type].pop())
    
    # Also create a test_results folder with the ground-truth target (after) images
    test_results_dir = out_root / "test_results"
    test_results_dir.mkdir(parents=True, exist_ok=True)

    for i, item in enumerate(test_files):
        src_before = item['before_path']
        src_after = item['after_path']
        test_id = f"test_{i:03d}.png"
        test_dst = test_dir / test_id
        shutil.copy2(src_before, test_dst)
        # Copy the corresponding real target image
        gt_dst = test_results_dir / test_id
        shutil.copy2(src_after, gt_dst)
    
    # Write CSV
    csv_path = out_root / "pairs.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
        writer.writeheader()
        writer.writerows(csv_data)
    
    # Write a Kontext LoRA YAML with prompts matching test images
    # Use pair captions to build prompts; match count to number of test images
    if all_pair_captions:
        prompts_for_yaml = [item['caption'] for item in test_files]
        write_kontext_yaml(out_root, [test_dir / f"test_{i:03d}.png" for i in range(len(test_files))], prompts_for_yaml)

    print(f"Created {len(valid_pairs)} pairs, {len(test_files)} test images")

def main():
    parser = argparse.ArgumentParser(description="Build Kontext dataset from CFD images")
    parser.add_argument("--raw-root", required=True, help="Root directory of raw images")
    parser.add_argument("--out-root", required=True, help="Output directory for dataset")
    parser.add_argument("--max-pairs", type=int, default=3000, help="Maximum number of pairs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    
    if not raw_root.exists():
        print(f"Error: Raw root {raw_root} does not exist")
        return 1
    
    # Discover images
    image_index = discover_images(raw_root)
    if not image_index:
        print("No images found!")
        return 1
    
    # Generate pairs
    print("Generating pairs...")
    
    all_pairs = []
    
    # Note: Variable switch pairs removed as they are not useful for surrogate modeling
    # (in CFD solvers, all variables are computed together for the same conditions)
    
    # AoA change pairs
    aoa_pairs = generate_aoa_change_pairs(image_index)
    random.shuffle(aoa_pairs)
    all_pairs.extend(aoa_pairs)
    print(f"AoA change pairs: {len(aoa_pairs)}")
    
    # Velocity change pairs
    vel_pairs = generate_velocity_change_pairs(image_index)
    random.shuffle(vel_pairs)
    all_pairs.extend(vel_pairs)
    print(f"Velocity change pairs: {len(vel_pairs)}")
    
    # Airfoil change pairs
    airfoil_pairs = generate_airfoil_change_pairs(image_index)
    random.shuffle(airfoil_pairs)
    all_pairs.extend(airfoil_pairs)
    print(f"Airfoil change pairs: {len(airfoil_pairs)}")
    
    # Identity pairs (5% of total)
    identity_count = max(1, int(0.05 * args.max_pairs))
    identity_pairs = generate_identity_pairs(image_index, identity_count)
    all_pairs.extend(identity_pairs)
    print(f"Identity pairs: {len(identity_pairs)}")
    
    # Shuffle and limit
    random.shuffle(all_pairs)
    final_pairs = all_pairs[:args.max_pairs]
    
    print(f"Total pairs generated: {len(all_pairs)}")
    print(f"Final pairs (after limit): {len(final_pairs)}")
    
    # Count by type
    type_counts = defaultdict(int)
    airfoils_covered = set()
    for pair in final_pairs:
        type_counts[pair['edit_type']] += 1
        if pair.get('airfoil_A'):
            airfoils_covered.add(pair['airfoil_A'])
        if pair.get('airfoil_B'):
            airfoils_covered.add(pair['airfoil_B'])
    
    print("\nPair distribution:")
    for edit_type, count in type_counts.items():
        print(f"  {edit_type}: {count}")
    print(f"Airfoils covered: {len(airfoils_covered)}")
    
    # Materialize dataset
    materialize_dataset(final_pairs, out_root)
    
    print(f"\nDataset created successfully at {out_root}")
    
    return 0

if __name__ == "__main__":
    exit(main())
