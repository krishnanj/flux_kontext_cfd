#!/usr/bin/env python3
"""
Script to plot VTU data with consistent color scales.
Reads global min/max data and plots each VTU file with consistent color scales.
"""

import os
import json
import argparse
import glob
from pathlib import Path
import vtk
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.tri as tri
# from matplotlib.colors import Normalize
# import matplotlib.cm as cm

import numpy as np

def process_airfoil_points(points_array):
    """
    Process airfoil points for plotting:
    - Sort by polar angle around centroid
    - Ensure CCW orientation
    - Rotate to start at trailing edge (max x)
    - Return closed loop (first = last)
    """
    # Convert to 2D (ignore z if present)
    points_2d = points_array[:, :2]

    # Step 1: centroid
    centroid = np.mean(points_2d, axis=0)

    # Step 2: sort by polar angle around centroid
    angles = np.arctan2(points_2d[:, 1] - centroid[1],
                        points_2d[:, 0] - centroid[0])
    sort_idx = np.argsort(angles)
    vertices = points_2d[sort_idx]

    # Step 3: ensure CCW (shoelace formula)
    def signed_area(pts):
        return 0.5 * np.sum(
            pts[:, 0] * np.roll(pts[:, 1], -1) -
            np.roll(pts[:, 0], -1) * pts[:, 1]
        )
    area = signed_area(vertices)
    if area < 0:
        vertices = vertices[::-1]

    # Step 4: rotate so trailing edge (max x) is first
    trailing_idx = np.argmax(vertices[:, 0])  # single point with max x
    vertices = np.vstack([vertices[trailing_idx:], vertices[:trailing_idx]])

    # Step 5: close the loop explicitly
    if not np.allclose(vertices[0], vertices[-1]):
        vertices = np.vstack([vertices, vertices[0]])

    print(f"Created closed CCW loop with {len(vertices)} points (first=last)")
    return vertices


def read_vtu_file(filepath):
    """Read VTU file and extract data arrays and coordinates."""
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filepath)
    reader.Update()
    
    data = reader.GetOutput()
    return data


def extract_data_from_vtu(data):
    """Extract coordinates and data arrays from VTU file."""
    points = data.GetPoints()
    point_data = data.GetPointData()
    
    # Extract coordinates
    coords = np.array([points.GetPoint(i) for i in range(points.GetNumberOfPoints())])
    x = coords[:, 0]
    y = coords[:, 1]
    
    # Extract velocity field (U has 3 components: u_x, u_y, u_z)
    u_array = point_data.GetArray('U')
    u_data = np.array([u_array.GetTuple3(i) for i in range(u_array.GetNumberOfTuples())])
    u_x = u_data[:, 0]
    u_y = u_data[:, 1]
    
    # Extract pressure
    p_array = point_data.GetArray('p')
    p_data = np.array([p_array.GetValue(i) for i in range(p_array.GetNumberOfTuples())])
    
    # Extract nut (turbulent viscosity)
    nut_array = point_data.GetArray('nut')
    nut_data = np.array([nut_array.GetValue(i) for i in range(nut_array.GetNumberOfTuples())])
    
    return x, y, u_x, u_y, p_data, nut_data


def extract_airfoil_boundary(vtu_file_path):
    """Extract airfoil boundary from the corresponding aerofoil.vtp file."""
    try:
        # Look for the corresponding aerofoil.vtp file
        aerofoil_path = str(vtu_file_path).replace('_internal.vtu', '_aerofoil.vtp')
        if not os.path.exists(aerofoil_path):
            print(f"  Warning: Aerofoil file not found: {aerofoil_path}")
            return None, None

        # Read the aerofoil VTP file
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(aerofoil_path)
        reader.Update()
        data = reader.GetOutput()
        points = data.GetPoints()

        # Extract raw airfoil coordinates
        airfoil_coords = np.array([points.GetPoint(i) for i in range(points.GetNumberOfPoints())])
        points_2d = airfoil_coords[:, :2]

        # --- Step 1: centroid
        centroid = np.mean(points_2d, axis=0)

        # --- Step 2: sort by polar angle around centroid
        angles = np.arctan2(points_2d[:, 1] - centroid[1],
                            points_2d[:, 0] - centroid[0])
        sort_idx = np.argsort(angles)
        vertices = points_2d[sort_idx]

        # --- Step 3: ensure CCW (shoelace formula)
        def signed_area(pts):
            return 0.5 * np.sum(
                pts[:, 0] * np.roll(pts[:, 1], -1) -
                np.roll(pts[:, 0], -1) * pts[:, 1]
            )
        if signed_area(vertices) < 0:
            vertices = vertices[::-1]

        # --- Step 4: rotate so trailing edge (max x) is first
        trailing_idx = np.argmax(vertices[:, 0])
        vertices = np.vstack([vertices[trailing_idx:], vertices[:trailing_idx]])

        # --- Step 5: close the loop explicitly
        if not np.allclose(vertices[0], vertices[-1]):
            vertices = np.vstack([vertices, vertices[0]])

        airfoil_x, airfoil_y = vertices[:, 0], vertices[:, 1]
        print(f"Extracted closed airfoil boundary with {len(vertices)} points")
        return airfoil_x, airfoil_y

    except Exception as e:
        print(f"  Warning: Could not extract airfoil boundary: {e}")
        return None, None


def create_triangulation(x, y):
    """Create triangulation for plotting."""
    # For unstructured grids, we need to create a triangulation
    # This is a simplified approach - for complex meshes, you might need to extract connectivity
    return tri.Triangulation(x, y)


from paraview.simple import *

def plot_field(x, y, field_data, field_name, vmin, vmax, output_path, vtu_file_path, airfoil_x=None, airfoil_y=None, u_inf=None):
    """
    Render a scalar/vector component field with ParaView using proper normalization.
    Supports u_x, u_y, Pressure (p), and nut with proper transformations.
    """
    print(f"  Rendering {field_name} -> {output_path}")

    # Use the provided VTU file path
    vtu_path = str(vtu_file_path)

    # Load VTU file
    reader = XMLUnstructuredGridReader(FileName=[vtu_path])
    reader.UpdatePipeline()

    # Apply proper transformations using Calculator filters with clipping
    if field_name in ["u_x", "u_y"]:
        # Apply velocity transformation using U_inf from folder name
        calc_transform = Calculator(Input=reader)
        if field_name == "u_x":
            calc_transform.ResultArrayName = 'U_X_transformed'
            # u_x* = u_x/U_inf - 1, clipped to [-1.25, 0.75]
            calc_transform.Function = f'max(min((U_X / {u_inf}) - 1.0, 0.75), -1.25)'
        else:  # u_y
            calc_transform.ResultArrayName = 'U_Y_transformed'
            # u_y* = u_y/U_inf, clipped to [-1.0, 1.0]
            calc_transform.Function = f'max(min(U_Y / {u_inf}, 1.0), -1.0)'
        calc_transform.UpdatePipeline()
        
        data_source = calc_transform
    elif field_name.lower().startswith("pressure"):
        # Apply pressure coefficient transformation
        # First get pressure data to compute p_inf (far-field pressure)
        import numpy as np
        point_data = reader.GetClientSideObject().GetOutput().GetPointData()
        p_array = point_data.GetArray('p')
        if p_array is None:
            raise ValueError("Pressure field 'p' not found in VTU file")
        
        p_data = np.array([p_array.GetValue(i) for i in range(p_array.GetNumberOfTuples())])
        p_inf = np.median(p_data)  # Use median as far-field pressure
        
        # Apply Cp transformation: Cp = (p - p_inf) / (0.5 * rho * U_inf^2), clipped to [-4.0, 1.0]
        rho = 1.0  # Air density
        calc_transform = Calculator(Input=reader)
        calc_transform.ResultArrayName = 'Cp_transformed'
        calc_transform.Function = f'max(min((p - {p_inf}) / (0.5 * {rho} * {u_inf} * {u_inf}), 1.0), -4.0)'
        calc_transform.UpdatePipeline()
        
        data_source = calc_transform
    elif "nut" in field_name.lower():
        # Apply log10(nut/nu) transformation, clipped to [-4.0, 2.0]
        nu = 1.5e-5  # Kinematic viscosity
        calc_transform = Calculator(Input=reader)
        calc_transform.ResultArrayName = 'Nut_transformed'
        calc_transform.Function = f'max(min(log10(max(nut / {nu}, 1e-10)), 2.0), -4.0)'  # Avoid log(0) and clip
        calc_transform.UpdatePipeline()
        
        data_source = calc_transform
    else:
        data_source = reader

    # Create render view
    view = CreateView('RenderView')
    view.ViewSize = [1280, 1024]  # Match trace size
    view.Background = [0.18, 0.10, 0.28]  # dark purple

    # Show data in view
    display = Show(data_source, view, 'UnstructuredGridRepresentation')

    # --- Map field_name to actual array and set appropriate colormap ---
    if field_name == "u_x":
        array_name = "U_X_transformed"
        ColorBy(display, ('POINTS', array_name))
        lut = GetColorTransferFunction(array_name)
        # Keep current colormap: Cool to Warm
        lut.ApplyPreset('Cool to Warm', True)
        lut.UseAboveRangeColor = 0
        lut.UseBelowRangeColor = 0
        # Use fixed bounds for u_x* = u_x/U_inf - 1
        lut.RescaleTransferFunction(vmin, vmax)
    elif field_name == "u_y":
        array_name = "U_Y_transformed"
        ColorBy(display, ('POINTS', array_name))
        lut = GetColorTransferFunction(array_name)
        # Keep current colormap: Cool to Warm
        lut.ApplyPreset('Cool to Warm', True)
        lut.UseAboveRangeColor = 0
        lut.UseBelowRangeColor = 0
        # Use fixed bounds for u_y* = u_y/U_inf
        lut.RescaleTransferFunction(vmin, vmax)
    elif field_name.lower().startswith("pressure"):
        array_name = "Cp_transformed"
        ColorBy(display, ('POINTS', array_name))
        lut = GetColorTransferFunction(array_name)
        # Keep current colormap: Cool to Warm
        lut.ApplyPreset('Cool to Warm', True)
        lut.UseAboveRangeColor = 0
        lut.UseBelowRangeColor = 0
        # Use fixed bounds for Cp
        lut.RescaleTransferFunction(vmin, vmax)
    elif "nut" in field_name.lower():
        array_name = "Nut_transformed"
        ColorBy(display, ('POINTS', array_name))
        lut = GetColorTransferFunction(array_name)
        # Keep current colormap: Viridis
        lut.ApplyPreset('Viridis (matplotlib)', True)
        lut.UseAboveRangeColor = 0
        lut.UseBelowRangeColor = 0
        # Use fixed bounds for log10(nut/nu)
        lut.RescaleTransferFunction(vmin, vmax)
    else:
        raise ValueError(f"Unsupported field_name: {field_name}")

    # Hide scalar bar
    display.SetScalarBarVisibility(view, False)

    # Hide orientation/axes
    view.OrientationAxesVisibility = 0
    view.AxesGrid.Visibility = 0

    # Reset view to fit data with proper zoom
    view.ResetCamera(False, 0.9)

    # Set 2D interaction mode and camera position for better zoom
    view.Set(
        InteractionMode='2D',
        CameraPosition=[0.6429966103317222, -0.009752059923356792, 21.88989453315735],
        CameraFocalPoint=[0.6429966103317222, -0.009752059923356792, 0.5],
        CameraParallelScale=1.140217166915732,
    )

    # --- Overlay airfoil polygon ---
    if airfoil_x is not None and airfoil_y is not None:
        from paraview import vtk
        import numpy as np

        pts = np.column_stack([airfoil_x, airfoil_y, np.zeros_like(airfoil_x)])

        points = vtk.vtkPoints()
        for pt in pts:
            points.InsertNextPoint(pt)

        poly = vtk.vtkPolyData()
        poly.SetPoints(points)

        polygon = vtk.vtkPolygon()
        polygon.GetPointIds().SetNumberOfIds(len(pts))
        for i in range(len(pts)):
            polygon.GetPointIds().SetId(i, i)

        cells = vtk.vtkCellArray()
        cells.InsertNextCell(polygon)
        poly.SetPolys(cells)

        src = TrivialProducer()
        src.GetClientSideObject().SetOutput(poly)

        outline = Show(src, view)
        outline.DiffuseColor = [0, 0, 0]       # fill black
        outline.LineWidth = 0.5  # Much thinner line
        outline.EdgeColor = [0.53, 0.81, 0.92] # light blue

    # Save screenshot with no compression
    SaveScreenshot(
        str(output_path), 
        view, 
        ImageResolution=[1280, 1024],
        CompressionLevel='0',  # No compression
        FontScaling='Scale fonts proportionally',
        OverrideColorPalette='',
        StereoMode='No change',
        TransparentBackground=0,
        SaveInBackground=0,
        EmbedParaViewState=0
    )

    # Cleanup - Delete all ParaView objects to prevent memory leaks
    try:
        Delete(view)
        del view
    except:
        pass
    
    try:
        Delete(display)
        del display
    except:
        pass
    
    try:
        Delete(lut)
        del lut
    except:
        pass
    
    try:
        Delete(data_source)
        del data_source
    except:
        pass
    
    try:
        Delete(calc_transform)
        del calc_transform
    except:
        pass
    
    try:
        Delete(reader)
        del reader
    except:
        pass
    
    try:
        Delete(src)
        del src
    except:
        pass
    
    try:
        Delete(outline)
        del outline
    except:
        pass
    
    # Force garbage collection
    import gc
    gc.collect()




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


def process_vtu_file(vtu_file, global_minmax, output_dir):
    """Process a single VTU file and create all plots with proper normalization."""
    print(f"Processing: {vtu_file.name}")
    
    try:
        # Extract U_inf from folder name
        folder_name = vtu_file.parent.name
        u_inf = extract_u_inf_from_folder_name(folder_name)
        
        if u_inf is None:
            print(f"  Warning: Could not extract U_inf from folder name: {folder_name}")
            return
        
        print(f"  U_inf from folder name: {u_inf}")
        
        # Read VTU file
        data = read_vtu_file(str(vtu_file))
        x, y, u_x, u_y, p_data, nut_data = extract_data_from_vtu(data)
        
        # Extract airfoil boundary
        airfoil_x, airfoil_y = extract_airfoil_boundary(vtu_file)
        
        # Create output directory for this file
        file_output_dir = output_dir / vtu_file.stem
        file_output_dir.mkdir(exist_ok=True)
        
        # Plot u_x
        u_x_output = file_output_dir / f"{vtu_file.stem}_u_x.png"
        plot_field(x, y, u_x, 'u_x', 
                  global_minmax['u_x']['min'], global_minmax['u_x']['max'],
                  u_x_output, vtu_file, airfoil_x, airfoil_y, u_inf)
        
        # Plot u_y
        u_y_output = file_output_dir / f"{vtu_file.stem}_u_y.png"
        plot_field(x, y, u_y, 'u_y',
                  global_minmax['u_y']['min'], global_minmax['u_y']['max'],
                  u_y_output, vtu_file, airfoil_x, airfoil_y, u_inf)
        
        # Plot pressure
        p_output = file_output_dir / f"{vtu_file.stem}_pressure.png"
        plot_field(x, y, p_data, 'Pressure',
                  global_minmax['pressure']['min'], global_minmax['pressure']['max'],
                  p_output, vtu_file, airfoil_x, airfoil_y, u_inf)
        
        # Plot nut
        nut_output = file_output_dir / f"{vtu_file.stem}_nut.png"
        plot_field(x, y, nut_data, 'Turbulent Viscosity (nut)',
                  global_minmax['nut']['min'], global_minmax['nut']['max'],
                  nut_output, vtu_file, airfoil_x, airfoil_y, u_inf)
        
        print(f"  Completed processing: {vtu_file.name}")
        
    except Exception as e:
        print(f"  Error processing {vtu_file.name}: {e}")


def clean_existing_files(folder_path, output_dir):
    """Clean up existing JSON and plot files."""
    print("Cleaning up existing files...")
    
    # Remove JSON files from dataset folders
    json_files = list(folder_path.rglob('*.json'))
    for json_file in json_files:
        try:
            json_file.unlink()
            print(f"  Removed: {json_file}")
        except Exception as e:
            print(f"  Could not remove {json_file}: {e}")
    
    # Remove existing plots directory
    if output_dir.exists():
        try:
            import shutil
            shutil.rmtree(output_dir)
            print(f"  Removed existing plots directory: {output_dir}")
        except Exception as e:
            print(f"  Could not remove plots directory: {e}")


def main():
    parser = argparse.ArgumentParser(description='Plot VTU data with consistent color scales')
    parser.add_argument('folder', help='Path to folder containing VTU files')
    parser.add_argument('--output', '-o', default='plots', 
                       help='Output directory for plots (default: plots)')
    parser.add_argument('--global-data', '-g', default='global_minmax_data.json',
                       help='Path to global min/max data file (default: global_minmax_data.json)')
    parser.add_argument('--clean', '-c', action='store_true',
                       help='Clean existing files before processing')
    parser.add_argument('--test', action='store_true',
                       help='Test mode: process only first 5 folders')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.folder):
        print(f"Error: Folder '{args.folder}' does not exist")
        return 1
    
    folder_path = Path(args.folder)
    output_dir = Path(args.output)
    
    # Clean existing files if requested
    if args.clean:
        clean_existing_files(folder_path, output_dir)
        print("Re-extracting min/max data...")
        # Re-run the extraction script with same test mode as plotting
        import subprocess
        try:
            cmd = ['python', 'extract_minmax.py', str(folder_path)]
            if args.test:
                cmd.append('--test')
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Error running extract_minmax.py: {result.stderr}")
                return 1
            print("Min/max data extraction completed")
        except Exception as e:
            print(f"Error running extract_minmax.py: {e}")
            return 1
    
    # Load global min/max data
    global_data_path = folder_path / args.global_data
    if not global_data_path.exists():
        print(f"Error: Global min/max data file '{global_data_path}' not found")
        print("Please run extract_minmax.py first to generate the data")
        return 1
    
    try:
        with open(global_data_path, 'r') as f:
            global_minmax = json.load(f)
        
        print("Loaded global min/max data:")
        for quantity, values in global_minmax.items():
            print(f"  {quantity}: min={values['min']:.6f}, max={values['max']:.6f}")
        
    except Exception as e:
        print(f"Error loading global min/max data: {e}")
        return 1
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Find all *_internal.vtu files
    vtu_files = list(folder_path.rglob('*_internal.vtu'))
    
    if not vtu_files:
        print(f"No *_internal.vtu files found in {folder_path}")
        return 1
    
    # If in test mode, only process first 5 folders
    if args.test:
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
    
    print(f"\nFound {len(vtu_files)} *_internal.vtu files")
    print(f"Output directory: {output_dir}")
    
    # Process each VTU file
    for vtu_file in vtu_files:
        process_vtu_file(vtu_file, global_minmax, output_dir)
    
    print(f"\nAll plots saved to: {output_dir}")
    return 0


if __name__ == '__main__':
    exit(main())
