#!/usr/bin/env python3
"""
Script to extract min/max values from VTU files.
Reads all *_internal.vtu files in a folder and extracts min/max values for:
- u_x, u_y (from velocity field U)
- pressure (p)
- nut (turbulent viscosity)

Saves the results to a JSON file for each VTU file.
"""

import os
import json
import argparse
import glob
from pathlib import Path
import vtk
import numpy as np


def read_vtu_file(filepath):
    """Read VTU file and extract data arrays."""
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filepath)
    reader.Update()
    
    data = reader.GetOutput()
    return data


def extract_minmax_from_vtu(data):
    """Extract min/max values for the required quantities."""
    point_data = data.GetPointData()
    
    # Extract velocity field (U has 3 components: u_x, u_y, u_z)
    u_array = point_data.GetArray('U')
    if u_array is None:
        raise ValueError("Velocity field 'U' not found in VTU file")
    
    u_data = np.array([u_array.GetTuple3(i) for i in range(u_array.GetNumberOfTuples())])
    u_x = u_data[:, 0]
    u_y = u_data[:, 1]
    
    # Extract pressure
    p_array = point_data.GetArray('p')
    if p_array is None:
        raise ValueError("Pressure field 'p' not found in VTU file")
    
    p_data = np.array([p_array.GetValue(i) for i in range(p_array.GetNumberOfTuples())])
    
    # Extract nut (turbulent viscosity)
    nut_array = point_data.GetArray('nut')
    if nut_array is None:
        raise ValueError("Turbulent viscosity field 'nut' not found in VTU file")
    
    nut_data = np.array([nut_array.GetValue(i) for i in range(nut_array.GetNumberOfTuples())])
    
    # Calculate min/max values for raw quantities
    minmax_data = {
        'u_x': {'min': float(np.min(u_x)), 'max': float(np.max(u_x))},
        'u_y': {'min': float(np.min(u_y)), 'max': float(np.max(u_y))},
        'pressure': {'min': float(np.min(p_data)), 'max': float(np.max(p_data))},
        'nut': {'min': float(np.min(nut_data)), 'max': float(np.max(nut_data))}
    }
    
    return minmax_data


def extract_u_inf_from_folder_name(folder_name):
    """Extract U_inf from folder name like 'airFoil2D_SST_31.283_-4.156_0.919_6.98_14.32'."""
    # Split by underscore and get the first number after 'airFoil2D_SST_'
    parts = folder_name.split('_')
    if len(parts) >= 3 and parts[0] == 'airFoil2D' and parts[1] == 'SST':
        try:
            u_inf = float(parts[2])
            return u_inf
        except ValueError:
            print(f"Warning: Could not parse U_inf from folder name: {folder_name}")
            return None
    return None


def extract_transformed_minmax_from_vtu(data, u_inf, p_inf=None, rho=1.0, nu=1.5e-5):
    """Extract min/max values for properly transformed quantities."""
    point_data = data.GetPointData()
    
    # Extract velocity field (U has 3 components: u_x, u_y, u_z)
    u_array = point_data.GetArray('U')
    if u_array is None:
        raise ValueError("Velocity field 'U' not found in VTU file")
    
    u_data = np.array([u_array.GetTuple3(i) for i in range(u_array.GetNumberOfTuples())])
    u_x = u_data[:, 0]
    u_y = u_data[:, 1]
    
    # Transform velocities using proper U_inf normalization
    u_x_transformed = (u_x / u_inf) - 1.0  # u_x* = u_x/U_inf - 1
    u_y_transformed = u_y / u_inf          # u_y* = u_y/U_inf
    
    # Extract pressure
    p_array = point_data.GetArray('p')
    if p_array is None:
        raise ValueError("Pressure field 'p' not found in VTU file")
    
    p_data = np.array([p_array.GetValue(i) for i in range(p_array.GetNumberOfTuples())])
    
    # Transform pressure to pressure coefficient Cp
    if p_inf is None:
        # Use median pressure as p_inf (far-field approximation)
        p_inf = np.median(p_data)
    
    # Cp = (p - p_inf) / (0.5 * rho * U_inf^2)
    cp_transformed = (p_data - p_inf) / (0.5 * rho * u_inf**2)
    
    # Extract nut (turbulent viscosity)
    nut_array = point_data.GetArray('nut')
    if nut_array is None:
        raise ValueError("Turbulent viscosity field 'nut' not found in VTU file")
    
    nut_data = np.array([nut_array.GetValue(i) for i in range(nut_array.GetNumberOfTuples())])
    
    # Transform nut to log10(nut/nu)
    # Avoid division by zero and log of zero
    nut_ratio = nut_data / nu
    nut_ratio = np.where(nut_ratio > 0, nut_ratio, 1e-10)  # Replace zeros with small positive value
    nut_transformed = np.log10(nut_ratio)
    # Handle any remaining invalid values
    nut_transformed = np.where(np.isfinite(nut_transformed), nut_transformed, -10.0)
    
    # Clip transformed quantities to expanded fixed bounds for consistent LORA training
    u_x_clipped = np.clip(u_x_transformed, -1.25, 0.75)
    u_y_clipped = np.clip(u_y_transformed, -1.0, 1.0)
    cp_clipped = np.clip(cp_transformed, -4.0, 1.0)
    nut_clipped = np.clip(nut_transformed, -4.0, 2.0)
    
    # Calculate min/max values for clipped quantities
    minmax_data = {
        'u_x': {'min': float(np.min(u_x_clipped)), 'max': float(np.max(u_x_clipped))},
        'u_y': {'min': float(np.min(u_y_clipped)), 'max': float(np.max(u_y_clipped))},
        'pressure': {'min': float(np.min(cp_clipped)), 'max': float(np.max(cp_clipped))},
        'nut': {'min': float(np.min(nut_clipped)), 'max': float(np.max(nut_clipped))}
    }
    
    return minmax_data


def process_folder(folder_path, test_mode=False):
    """Process all *_internal.vtu files in the given folder with proper normalization."""
    folder_path = Path(folder_path)
    
    # Find all *_internal.vtu files recursively
    vtu_files = list(folder_path.rglob('*_internal.vtu'))
    
    if not vtu_files:
        print(f"No *_internal.vtu files found in {folder_path}")
        return
    
    print(f"Found {len(vtu_files)} *_internal.vtu files")
    
    # If in test mode, only process first 5 folders
    if test_mode:
        # Get unique folder names and limit to first 5
        folder_names = set()
        for vtu_file in vtu_files:
            folder_name = vtu_file.parent.name
            folder_names.add(folder_name)
        
        test_folders = list(folder_names)[:5]
        print(f"Test mode: Processing only first 5 folders: {test_folders}")
        
        # Filter vtu_files to only include those from test folders
        vtu_files = [f for f in vtu_files if f.parent.name in test_folders]
        print(f"Processing {len(vtu_files)} files from test folders")
    
    # Process each VTU file with proper normalization
    all_minmax_data = {}
    
    for vtu_file in vtu_files:
        print(f"Processing: {vtu_file.name}")
        
        try:
            # Extract U_inf from folder name
            folder_name = vtu_file.parent.name
            u_inf = extract_u_inf_from_folder_name(folder_name)
            
            if u_inf is None:
                print(f"  Warning: Could not extract U_inf from folder name: {folder_name}")
                continue
            
            print(f"  U_inf from folder name: {u_inf}")
            
            # Read VTU file
            data = read_vtu_file(str(vtu_file))
            
            # Extract min/max values with proper U_inf transformation
            minmax_data = extract_transformed_minmax_from_vtu(data, u_inf)
            
            # Store data with relative path as key
            relative_path = vtu_file.relative_to(folder_path)
            all_minmax_data[str(relative_path)] = minmax_data
            
            # Save individual file data
            output_file = vtu_file.with_suffix('.json')
            with open(output_file, 'w') as f:
                json.dump(minmax_data, f, indent=2)
            
            print(f"  Saved min/max data to: {output_file}")
            
        except Exception as e:
            print(f"  Error processing {vtu_file.name}: {e}")
            continue
    
    # Save combined data
    combined_output = folder_path / 'all_minmax_data.json'
    with open(combined_output, 'w') as f:
        json.dump(all_minmax_data, f, indent=2)
    
    print(f"\nSaved combined min/max data to: {combined_output}")
    
    # Calculate global min/max across all files
    global_minmax = calculate_global_minmax(all_minmax_data)
    
    global_output = folder_path / 'global_minmax_data.json'
    with open(global_output, 'w') as f:
        json.dump(global_minmax, f, indent=2)
    
    print(f"Saved global min/max data to: {global_output}")
    
    return global_minmax


def calculate_global_minmax(all_minmax_data):
    """Calculate global min/max values using fixed bounds for consistent LORA training."""
    # Use expanded fixed bounds as specified for LORA training
    global_minmax = {
        'u_x': {'min': -1.25, 'max': 0.75},    # Expanded bounds for u_x* = u_x/U_inf - 1
        'u_y': {'min': -1.0, 'max': 1.0},      # Expanded bounds for u_y* = u_y/U_inf  
        'pressure': {'min': -4.0, 'max': 1.0}, # Expanded bounds for Cp
        'nut': {'min': -4.0, 'max': 2.0}       # Expanded bounds for log10(nut/nu)
    }
    
    return global_minmax


def calculate_global_u_max(folder_path):
    """Calculate global U_max across all files."""
    folder_path = Path(folder_path)
    vtu_files = list(folder_path.rglob('*_internal.vtu'))
    
    global_u_max = 0.0
    
    for vtu_file in vtu_files:
        try:
            data = read_vtu_file(str(vtu_file))
            point_data = data.GetPointData()
            
            u_array = point_data.GetArray('U')
            if u_array is None:
                continue
                
            u_data = np.array([u_array.GetTuple3(i) for i in range(u_array.GetNumberOfTuples())])
            u_x = u_data[:, 0]
            u_y = u_data[:, 1]
            
            # Calculate velocity magnitude for this file
            u_magnitude = np.sqrt(u_x**2 + u_y**2)
            file_u_max = np.max(u_magnitude)
            
            global_u_max = max(global_u_max, file_u_max)
            
        except Exception as e:
            print(f"  Error calculating U_max for {vtu_file.name}: {e}")
            continue
    
    return global_u_max


def main():
    parser = argparse.ArgumentParser(description='Extract min/max values from VTU files with proper normalization')
    parser.add_argument('folder', help='Path to folder containing VTU files')
    parser.add_argument('--test', action='store_true', help='Test mode: process only first 5 folders')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.folder):
        print(f"Error: Folder '{args.folder}' does not exist")
        return 1
    
    try:
        global_minmax = process_folder(args.folder, test_mode=args.test)
        
        print("\nGlobal min/max values across all files:")
        for quantity, values in global_minmax.items():
            print(f"  {quantity}: min={values['min']:.6f}, max={values['max']:.6f}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
