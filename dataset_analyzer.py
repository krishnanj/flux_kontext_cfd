#!/usr/bin/env python3
"""
Complete dataset analyzer for AirfRANS dataset.
Extracts information from folder names and creates comprehensive tables.
"""

import os
import pandas as pd
import argparse
import re
from pathlib import Path
from tabulate import tabulate

def parse_folder_name(folder_name):
    """
    Parse folder name like 'airFoil2D_SST_43.597_5.932_3.551_3.1_1.0_18.252'
    to extract velocity, AoA, and NACA parameters.
    """
    pattern = r'airFoil2D_SST_([\d.-]+)_([\d.-]+)_(.+)'
    match = re.match(pattern, folder_name)
    
    if not match:
        return None
    
    try:
        velocity = float(match.group(1))
        aoa = float(match.group(2))
        
        # Parse the remaining parameters (NACA airfoil parameters)
        remaining_params = match.group(3)
        param_strings = remaining_params.split('_')
        naca_params = [float(p) for p in param_strings]
        
        # Determine if it's 4-digit or 5-digit series
        if len(naca_params) == 3:
            airfoil_type = "4-digit"
            camber, camber_pos, thickness = naca_params
            naca_number = f"{int(camber):02d}{int(camber_pos):02d}{int(thickness):02d}"
        elif len(naca_params) == 4:
            airfoil_type = "5-digit"
            design_cl, camber_pos, thickness, reflex = naca_params
            naca_number = f"{int(design_cl):01d}{int(camber_pos):02d}{int(thickness):02d}"
            if reflex > 0:
                naca_number += f"_{int(reflex):02d}"
        else:
            airfoil_type = "unknown"
            naca_number = "unknown"
        
        return {
            'folder_name': folder_name,
            'velocity': velocity,
            'aoa': aoa,
            'airfoil_type': airfoil_type,
            'naca_number': naca_number,
            'naca_params': naca_params,
            'param_count': len(naca_params)
        }
    
    except (ValueError, IndexError) as e:
        print(f"Error parsing folder '{folder_name}': {e}")
        return None

def extract_dataset_info(dataset_path):
    """Extract information from all folders in the dataset."""
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    
    folders = [f for f in dataset_path.iterdir() if f.is_dir()]
    print(f"Found {len(folders)} simulation folders")
    
    parsed_data = []
    failed_parses = []
    
    for folder in folders:
        folder_name = folder.name
        parsed = parse_folder_name(folder_name)
        
        if parsed:
            parsed_data.append(parsed)
        else:
            failed_parses.append(folder_name)
    
    if failed_parses:
        print(f"\nFailed to parse {len(failed_parses)} folders:")
        for folder in failed_parses[:10]:
            print(f"  {folder}")
        if len(failed_parses) > 10:
            print(f"  ... and {len(failed_parses) - 10} more")
    
    df = pd.DataFrame(parsed_data)
    
    if df.empty:
        print("No valid folders found!")
        return df
    
    df = df.sort_values(['velocity', 'aoa', 'naca_number']).reset_index(drop=True)
    return df

def create_summary_table(df):
    """Create a comprehensive summary table."""
    if df.empty:
        return None
    
    summary_data = [
        ["Total Simulations", len(df)],
        ["Velocity Range", f"{df['velocity'].min():.3f} - {df['velocity'].max():.3f}"],
        ["AoA Range", f"{df['aoa'].min():.3f}° - {df['aoa'].max():.3f}°"],
        ["Unique Velocities", df['velocity'].nunique()],
        ["Unique AoAs", df['aoa'].nunique()],
        ["Unique Airfoils", df['naca_number'].nunique()],
        ["4-digit Airfoils", len(df[df['airfoil_type'] == '4-digit'])],
        ["5-digit Airfoils", len(df[df['airfoil_type'] == '5-digit'])]
    ]
    
    return tabulate(summary_data, headers=["Metric", "Value"], tablefmt="grid")

def create_simple_parameter_table(df, n_rows=50):
    """Create a simple table with the three main parameters."""
    simple_df = df[['velocity', 'aoa', 'airfoil_type', 'naca_number']].copy()
    simple_df.columns = ['Velocity (U_inf)', 'AoA (degrees)', 'Airfoil Type', 'NACA Number']
    return simple_df.head(n_rows)

def create_airfoil_breakdown(df):
    """Create a breakdown by airfoil type."""
    breakdown = df.groupby('airfoil_type').agg({
        'folder_name': 'count',
        'velocity': ['min', 'max', 'mean'],
        'aoa': ['min', 'max', 'mean'],
        'naca_number': 'nunique'
    }).round(3)
    
    breakdown.columns = ['Count', 'Vel_Min', 'Vel_Max', 'Vel_Mean', 'AoA_Min', 'AoA_Max', 'AoA_Mean', 'Unique_Airfoils']
    breakdown = breakdown.reset_index()
    
    return tabulate(breakdown, headers="keys", tablefmt="grid", showindex=False)

def create_velocity_aoa_ranges(df):
    """Create velocity and AoA range analysis."""
    df['velocity_bin'] = pd.cut(df['velocity'], bins=10, precision=1)
    df['aoa_bin'] = pd.cut(df['aoa'], bins=10, precision=1)
    
    vel_counts = df['velocity_bin'].value_counts().sort_index()
    aoa_counts = df['aoa_bin'].value_counts().sort_index()
    
    vel_table = tabulate([(str(interval), count) for interval, count in vel_counts.items()], 
                        headers=["Velocity Range", "Count"], tablefmt="grid")
    
    aoa_table = tabulate([(str(interval), count) for interval, count in aoa_counts.items()], 
                        headers=["AoA Range", "Count"], tablefmt="grid")
    
    return vel_table, aoa_table

def create_naca_analysis(df):
    """Analyze NACA airfoil distribution."""
    common_airfoils = df['naca_number'].value_counts().head(20)
    airfoil_table = tabulate([(naca, count) for naca, count in common_airfoils.items()], 
                            headers=["NACA Number", "Count"], tablefmt="grid")
    return airfoil_table

def main():
    parser = argparse.ArgumentParser(description='Complete AirfRANS dataset analyzer')
    parser.add_argument('dataset_path', nargs='?', default='airfRANS_dataset',
                       help='Path to the airfRANS_dataset directory (default: airfRANS_dataset)')
    parser.add_argument('--csv', default='dataset_info.csv',
                       help='Output CSV file (default: dataset_info.csv)')
    parser.add_argument('--analysis', default='dataset_analysis.txt',
                       help='Output analysis file (default: dataset_analysis.txt)')
    parser.add_argument('--simple', default='simple_parameters.txt',
                       help='Output simple table file (default: simple_parameters.txt)')
    parser.add_argument('--samples', '-s', type=int, default=20,
                       help='Number of sample rows to show (default: 20)')
    parser.add_argument('--rows', '-r', type=int, default=50,
                       help='Number of rows for simple table (default: 50)')
    parser.add_argument('--extract-only', action='store_true',
                       help='Only extract data to CSV, skip analysis')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze existing CSV, skip extraction')
    
    args = parser.parse_args()
    
    try:
        df = None
        
        # Extract data if not analyze-only mode
        if not args.analyze_only:
            print("Extracting dataset information...")
            df = extract_dataset_info(args.dataset_path)
            
            if df.empty:
                print("No data extracted!")
                return 1
            
            # Save to CSV
            csv_path = Path(args.csv)
            df.to_csv(csv_path, index=False)
            print(f"Dataset information saved to: {csv_path}")
        
        # Load data if not extract-only mode
        if not args.extract_only:
            if df is None:
                csv_path = Path(args.csv)
                if not csv_path.exists():
                    print(f"Error: CSV file '{csv_path}' not found")
                    print("Please run without --analyze-only first to generate the CSV file")
                    return 1
                df = pd.read_csv(csv_path)
            
            # Create comprehensive analysis
            analysis_content = []
            analysis_content.append("="*80)
            analysis_content.append("AIRFRANS DATASET ANALYSIS")
            analysis_content.append("="*80)
            analysis_content.append("")
            
            analysis_content.append("DATASET SUMMARY")
            analysis_content.append("-"*40)
            analysis_content.append(create_summary_table(df))
            analysis_content.append("")
            
            analysis_content.append("SAMPLE SIMULATIONS (First 20)")
            analysis_content.append("-"*40)
            sample_df = df.head(args.samples)[['folder_name', 'velocity', 'aoa', 'airfoil_type', 'naca_number']]
            analysis_content.append(tabulate(sample_df, headers="keys", tablefmt="grid", showindex=False))
            analysis_content.append("")
            
            analysis_content.append("AIRFOIL TYPE BREAKDOWN")
            analysis_content.append("-"*40)
            analysis_content.append(create_airfoil_breakdown(df))
            analysis_content.append("")
            
            analysis_content.append("VELOCITY AND AOA DISTRIBUTIONS")
            analysis_content.append("-"*40)
            vel_table, aoa_table = create_velocity_aoa_ranges(df)
            analysis_content.append(vel_table)
            analysis_content.append("")
            analysis_content.append(aoa_table)
            analysis_content.append("")
            
            analysis_content.append("MOST COMMON NACA AIRFOILS")
            analysis_content.append("-"*40)
            analysis_content.append(create_naca_analysis(df))
            analysis_content.append("")
            
            # Save comprehensive analysis
            analysis_path = Path(args.analysis)
            with open(analysis_path, 'w') as f:
                f.write('\n'.join(analysis_content))
            print(f"Comprehensive analysis saved to: {analysis_path}")
            
            # Create simple parameter table
            simple_content = []
            simple_content.append("="*80)
            simple_content.append("AIRFRANS DATASET - MAIN PARAMETERS")
            simple_content.append("="*80)
            simple_content.append("")
            simple_content.append("The dataset contains 1000 simulations with three main varying parameters:")
            simple_content.append("1. Velocity (U_inf): Inlet velocity magnitude")
            simple_content.append("2. AoA: Angle of Attack in degrees")
            simple_content.append("3. NACA Airfoil: Airfoil geometry (4-digit or 5-digit series)")
            simple_content.append("")
            simple_content.append(f"Showing first {args.rows} simulations:")
            simple_content.append("")
            
            simple_df = create_simple_parameter_table(df, args.rows)
            simple_content.append(tabulate(simple_df, headers="keys", tablefmt="grid", showindex=False))
            simple_content.append("")
            simple_content.append("="*80)
            simple_content.append("SUMMARY STATISTICS")
            simple_content.append("="*80)
            simple_content.append(f"Total Simulations: {len(df)}")
            simple_content.append(f"Velocity Range: {df['velocity'].min():.3f} - {df['velocity'].max():.3f}")
            simple_content.append(f"AoA Range: {df['aoa'].min():.3f}° - {df['aoa'].max():.3f}°")
            simple_content.append(f"4-digit Airfoils: {len(df[df['airfoil_type'] == '4-digit'])}")
            simple_content.append(f"5-digit Airfoils: {len(df[df['airfoil_type'] == '5-digit'])}")
            simple_content.append(f"Unique NACA Numbers: {df['naca_number'].nunique()}")
            
            # Save simple table
            simple_path = Path(args.simple)
            with open(simple_path, 'w') as f:
                f.write('\n'.join(simple_content))
            print(f"Simple parameter table saved to: {simple_path}")
            
            # Print summary to console
            print("\n" + "="*50)
            print("QUICK SUMMARY")
            print("="*50)
            print(f"Total Simulations: {len(df)}")
            print(f"Velocity Range: {df['velocity'].min():.3f} - {df['velocity'].max():.3f}")
            print(f"AoA Range: {df['aoa'].min():.3f}° - {df['aoa'].max():.3f}°")
            print(f"4-digit Airfoils: {len(df[df['airfoil_type'] == '4-digit'])}")
            print(f"5-digit Airfoils: {len(df[df['airfoil_type'] == '5-digit'])}")
            print(f"Unique NACA Numbers: {df['naca_number'].nunique()}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == '__main__':
    exit(main())
