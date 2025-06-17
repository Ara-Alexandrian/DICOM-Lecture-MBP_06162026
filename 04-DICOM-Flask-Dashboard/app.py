import os
import pydicom
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
import io
import base64
from PIL import Image
import json
from dotenv import load_dotenv
from scipy import ndimage
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # For 3D structures

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'default-dev-key')

# Global configuration
CONFIG = {
    'data_dir': os.environ.get('DATA_DIR', '../data/test-cases'),
    'window_presets': {
        'Soft Tissue': (40, 400),
        'Lung': (-600, 1500),
        'Bone': (500, 2000),
        'Brain': (40, 80),
        'Custom': 'custom'
    },
    'dose_colormaps': ['jet', 'hot', 'plasma', 'viridis', 'turbo', 'rainbow']
}

# Helper functions
def get_patients():
    """List all patient directories"""
    data_dir = CONFIG['data_dir']
    if not os.path.exists(data_dir):
        return []
    
    patients = [d for d in os.listdir(data_dir) 
               if os.path.isdir(os.path.join(data_dir, d))]
    return patients

def load_ct_files(patient_dir):
    """Load CT files from patient directory"""
    ct_files = [f for f in os.listdir(patient_dir) if f.startswith("CT")]
    ct_files.sort()  # Sort to get them in order
    
    ct_data = []
    for ct_file in ct_files:
        try:
            ds = pydicom.dcmread(os.path.join(patient_dir, ct_file))
            ct_data.append({
                'file': ct_file,
                'path': os.path.join(patient_dir, ct_file),
                'instance': ds,
                'z': float(ds.SliceLocation),
                'thickness': float(ds.SliceThickness) if hasattr(ds, 'SliceThickness') else 1.0
            })
        except Exception as e:
            print(f"Error loading {ct_file}: {str(e)}")
    
    # Sort by slice location
    ct_data.sort(key=lambda x: x['z'])
    
    return ct_data

def load_structure_set(patient_dir):
    """Load RT Structure Set from patient directory"""
    rs_files = [f for f in os.listdir(patient_dir) if f.startswith("RS")]
    if not rs_files:
        return None
    
    rs_path = os.path.join(patient_dir, rs_files[0])
    try:
        rs = pydicom.dcmread(rs_path)
        return {
            'file': rs_files[0],
            'path': rs_path,
            'instance': rs
        }
    except Exception as e:
        print(f"Error loading structure set: {str(e)}")
        return None

def load_rt_plan(patient_dir):
    """Load RT Plan from patient directory"""
    rp_files = [f for f in os.listdir(patient_dir) if f.startswith("RP")]
    if not rp_files:
        return None
    
    rp_path = os.path.join(patient_dir, rp_files[0])
    try:
        rp = pydicom.dcmread(rp_path)
        return {
            'file': rp_files[0],
            'path': rp_path,
            'instance': rp
        }
    except Exception as e:
        print(f"Error loading RT plan: {str(e)}")
        return None

def load_rt_dose(patient_dir):
    """Load RT Dose from patient directory"""
    rd_files = [f for f in os.listdir(patient_dir) if f.startswith("RD")]
    if not rd_files:
        return None
    
    rd_path = os.path.join(patient_dir, rd_files[0])
    try:
        rd = pydicom.dcmread(rd_path)
        return {
            'file': rd_files[0],
            'path': rd_path,
            'instance': rd
        }
    except Exception as e:
        print(f"Error loading RT dose: {str(e)}")
        return None

def get_hu_values(ct_slice):
    """Convert pixel data to Hounsfield Units (HU)"""
    pixel_array = ct_slice.pixel_array
    intercept = ct_slice.RescaleIntercept
    slope = ct_slice.RescaleSlope
    hu_values = pixel_array * slope + intercept
    return hu_values

def apply_window_level(hu_image, window_center, window_width):
    """Apply window/level to the image"""
    min_value = window_center - window_width/2
    max_value = window_center + window_width/2
    hu_image_windowed = np.clip(hu_image, min_value, max_value)
    return hu_image_windowed

def get_contour_data(rs_instance, structure_name):
    """Extract contour data for a specific structure name"""
    rs = rs_instance['instance']
    
    # First, find the ROI number for the structure name
    roi_number = None
    for roi in rs.StructureSetROISequence:
        if roi.ROIName == structure_name:
            roi_number = roi.ROINumber
            break
    
    if roi_number is None:
        return None
    
    # Next, find the contour data for this ROI number
    contour_data = []
    for roi_contour in rs.ROIContourSequence:
        if roi_contour.ReferencedROINumber == roi_number:
            # Get the color (with fallback if ROIDisplayColor is missing)
            if hasattr(roi_contour, 'ROIDisplayColor'):
                color = [c/255 for c in roi_contour.ROIDisplayColor]
            else:
                # Default to a bright color that will be visible
                color = [1.0, 0.0, 1.0]  # Magenta as fallback
            
            # Get the contour sequence
            if hasattr(roi_contour, 'ContourSequence'):
                for contour in roi_contour.ContourSequence:
                    # Skip contours without data
                    if not hasattr(contour, 'ContourData') or len(contour.ContourData) < 6:
                        continue
                        
                    contour_coords = contour.ContourData
                    # Reshape into x,y,z triplets
                    triplets = np.array(contour_coords).reshape((-1, 3))
                    
                    # Skip contours with fewer than 3 points
                    if triplets.shape[0] < 3:
                        continue
                    
                    # Calculate average z position for more accurate slice matching
                    # Some structures might have slightly varying z values within a single contour
                    avg_z = np.mean(triplets[:, 2])
                    
                    contour_data.append({
                        'triplets': triplets,
                        'z': avg_z,  # Use average Z coordinate for better slice matching
                        'z_min': np.min(triplets[:, 2]),  # Store min z for range checks
                        'z_max': np.max(triplets[:, 2]),  # Store max z for range checks
                        'color': color
                    })
    
    # Log the number of contours found
    if contour_data:
        print(f"Found {len(contour_data)} contours for structure '{structure_name}'")
    else:
        print(f"No contours found for structure '{structure_name}'")
        
    return contour_data

def list_structures(rs_instance):
    """List all structures in an RT Structure Set with extensive validation"""
    if not rs_instance:
        return []
    
    rs = rs_instance['instance']
    structures = []
    
    # Make sure we have the required sequences
    if not hasattr(rs, 'StructureSetROISequence') or not hasattr(rs, 'ROIContourSequence'):
        return []
    
    # Create a safe structure list with validation
    for roi in rs.StructureSetROISequence:
        # Skip ROIs without required attributes
        if not hasattr(roi, 'ROINumber') or not hasattr(roi, 'ROIName'):
            continue
            
        # Skip specific point-only structures by name that shouldn't be rendered
        # These often appear as reference markers, isocenter markers, etc.
        skip_keywords = ['ISO', 'MARKER', 'POINT', 'MARK', 'REF', 'CAL']
        should_skip = False
        for keyword in skip_keywords:
            if keyword in roi.ROIName.upper():
                should_skip = True
                break
        
        if should_skip:
            continue
            
        valid_contour = False
        for roi_contour in rs.ROIContourSequence:
            # Check if this contour references the current ROI
            if not hasattr(roi_contour, 'ReferencedROINumber'):
                continue
                
            if roi_contour.ReferencedROINumber == roi.ROINumber:
                # Get color with fallback
                if hasattr(roi_contour, 'ROIDisplayColor'):
                    color = [c/255 for c in roi_contour.ROIDisplayColor]
                    hex_color = "#{:02x}{:02x}{:02x}".format(
                        int(color[0]*255), int(color[1]*255), int(color[2]*255))
                else:
                    # Default bright color if no color defined
                    hex_color = "#FF00FF"  # Magenta
                
                # Carefully validate that there's actual contour data to display
                has_valid_contours = False
                if hasattr(roi_contour, 'ContourSequence') and len(roi_contour.ContourSequence) > 0:
                    # Check each contour has enough points to be a valid polygon (at least 3 points)
                    for contour in roi_contour.ContourSequence:
                        if hasattr(contour, 'ContourData') and len(contour.ContourData) >= 6:  # At least 3 points (x,y,z)
                            has_valid_contours = True
                            break
                
                if has_valid_contours:
                    valid_contour = True
                    structures.append({
                        'number': roi.ROINumber,
                        'name': roi.ROIName,
                        'color': hex_color,
                        'has_contours': True
                    })
                    break
    
    # Log how many structures were filtered out
    print(f"Found {len(structures)} valid structures with contours")
                    
    # Sort structures by name for consistency
    structures.sort(key=lambda x: x['name'])
    
    return structures

def get_dose_slice(rd_instance, z_position):
    """Get dose slice for a specific z position"""
    if not rd_instance:
        return None
    
    rd = rd_instance['instance']
    
    # Get dose dimensions
    rows = rd.Rows
    cols = rd.Columns
    num_frames = rd.NumberOfFrames if hasattr(rd, 'NumberOfFrames') else 1
    
    # Get dose grid coordinates
    dose_x = np.arange(cols) * rd.PixelSpacing[1] + rd.ImagePositionPatient[0]
    dose_y = np.arange(rows) * rd.PixelSpacing[0] + rd.ImagePositionPatient[1]
    dose_z = []
    
    # For 3D dose, get Z coordinates from grid frame offset vector
    if hasattr(rd, 'GridFrameOffsetVector'):
        for i in range(num_frames):
            dose_z.append(rd.ImagePositionPatient[2] + rd.GridFrameOffsetVector[i])
    else:
        dose_z = [rd.ImagePositionPatient[2]]
    
    # Find the closest Z slice
    closest_z_idx = np.argmin(np.abs(np.array(dose_z) - z_position))
    
    # Get the dose slice
    if num_frames > 1:
        dose_slice = rd.pixel_array[closest_z_idx] * rd.DoseGridScaling
    else:
        dose_slice = rd.pixel_array * rd.DoseGridScaling
    
    return dose_slice

def get_global_dose_max(rd_instance):
    """Get global dose max across all slices"""
    if not rd_instance:
        return 0
    
    rd = rd_instance['instance']
    num_frames = rd.NumberOfFrames if hasattr(rd, 'NumberOfFrames') else 1
    
    if num_frames > 1:
        dose_max = np.max(rd.pixel_array) * rd.DoseGridScaling
    else:
        dose_max = np.max(rd.pixel_array) * rd.DoseGridScaling
    
    return dose_max

def render_ct_slice(ct_instance, slice_idx, window_center, window_width,
                    structures=None, rs_data=None,
                    show_dose=False, dose_opacity=0.5, dose_colormap='jet',
                    rd_data=None):
    """Render CT slice with overlays and return as base64 encoded image"""
    current_slice = ct_instance[slice_idx]['instance']

    # Get HU values
    hu_image = get_hu_values(current_slice)

    # Apply window/level
    hu_image_windowed = apply_window_level(hu_image, window_center, window_width)

    # Create figure with higher DPI for better quality
    fig, ax = plt.subplots(figsize=(10, 10), dpi=120)
    ax.imshow(hu_image_windowed, cmap='gray', aspect='equal')

    # Get image dimensions and origin for coordinate conversion
    rows, cols = hu_image.shape
    origin = current_slice.ImagePositionPatient
    spacing = current_slice.PixelSpacing
    current_z = current_slice.SliceLocation

    # Debug print image size and position
    print(f"Image dimensions: {rows}x{cols}")
    print(f"Image position: {origin}")
    print(f"Pixel spacing: {spacing}")
    print(f"Slice location: {current_z}")

    # Track which structures are actually shown for reporting
    structures_shown = []

    # Add structures if requested
    if structures and rs_data:
        # Calculate vertical spacing for structure labels
        label_y_offset = 30

        # Sort structures alphabetically for consistent display
        sorted_structures = sorted(structures)

        for i, structure_name in enumerate(sorted_structures):
            contour_data = get_contour_data(rs_data, structure_name)
            if not contour_data:
                print(f"No contour data found for structure {structure_name}")
                continue

            # Use much more relaxed tolerance for z-matching - 5mm in each direction
            contours_on_slice = [c for c in contour_data if abs(c['z'] - current_z) <= 5.0]

            if not contours_on_slice:
                print(f"No contours found for structure {structure_name} on slice at {current_z}mm")
                continue

            print(f"Found {len(contours_on_slice)} contours for structure {structure_name} on slice at {current_z}mm")

            # Keep track if we successfully rendered any contours for this structure
            structure_rendered = False

            # Process each contour - use a simpler but more robust approach
            for contour in contours_on_slice:
                try:
                    # Extract contour points
                    points_3d = contour['triplets']

                    # Skip if too few points
                    if len(points_3d) < 3:
                        continue

                    # SIMPLER CONVERSION STRATEGY:
                    # 1. Extract X and Y coordinates from the 3D points
                    # 2. Scale them according to pixel spacing
                    # 3. Shift them according to the image origin

                    # Extract x and y coordinates
                    x_points = [(p[0] - origin[0]) / spacing[1] for p in points_3d]
                    y_points = [(p[1] - origin[1]) / spacing[0] for p in points_3d]

                    # Invert y coordinates - DICOM y axis is inverted compared to image coordinates
                    y_points = [rows - y for y in y_points]

                    # Print first few points for debugging
                    if len(contours_on_slice) > 0 and contour == contours_on_slice[0]:
                        print(f"Structure {structure_name} at z={contour['z']:.1f}mm (slice z={current_z:.1f}mm):")
                        for k in range(min(3, len(points_3d))):
                            print(f"  3D: ({points_3d[k][0]:.1f}, {points_3d[k][1]:.1f}, {points_3d[k][2]:.1f}) → 2D: ({x_points[k]:.1f}, {y_points[k]:.1f})")

                    # Create a closed polygon
                    ax.fill(x_points, y_points,
                           facecolor=contour['color'],
                           alpha=0.3,
                           edgecolor='black',
                           linewidth=2,
                           zorder=10)

                    # Add a solid white outline inside the black outline for better visibility
                    ax.plot(x_points + [x_points[0]], y_points + [y_points[0]],
                           color='white',
                           linewidth=1,
                           zorder=11)

                    # Mark that we've successfully rendered at least one contour
                    structure_rendered = True

                except Exception as e:
                    print(f"Error rendering structure {structure_name}: {e}")
                    continue

            # Only show the structure in the legend if we actually rendered it
            if structure_rendered:
                # Add structure name label to image with vertical spacing
                label_y = label_y_offset + (i * 25)  # Offset each label
                ax.text(10, label_y, structure_name,
                       color='white', fontsize=14, fontweight='bold',
                       bbox=dict(facecolor=contour['color'], alpha=0.7, edgecolor='black', pad=3))

                structures_shown.append(structure_name)

    # Add dose overlay if requested
    dose_display = None
    if show_dose and rd_data:
        dose_overlay = get_dose_slice(rd_data, current_slice.SliceLocation)
        global_dose_max = get_global_dose_max(rd_data)

        if dose_overlay is not None and global_dose_max > 0:
            # Use the specified colormap and opacity
            cmap = plt.get_cmap(dose_colormap)
            norm = mcolors.Normalize(vmin=0, vmax=global_dose_max)
            dose_display = ax.imshow(dose_overlay, cmap=cmap, norm=norm, alpha=dose_opacity)

            # Add colorbar for dose scale
            cbar = fig.colorbar(dose_display, ax=ax, orientation='vertical', pad=0.01)
            cbar.set_label('Dose (Gy)', rotation=270, labelpad=15)

    # Add slice position info
    slice_info = f"CT Slice at {current_slice.SliceLocation}mm"
    ax.set_title(slice_info)
    ax.axis('off')

    # Save figure to BytesIO with higher quality
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    plt.close(fig)

    # Encode the image to base64
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')

    return img_str, structures_shown

def calculate_dvh(structure_name, rs_instance, rd_instance):
    """Calculate a simple DVH for a structure"""
    if not rs_instance or not rd_instance:
        return None
    
    rs = rs_instance['instance']
    rd = rd_instance['instance']
    
    # Get structure contours
    contour_data = get_contour_data(rs_instance, structure_name)
    if not contour_data:
        return None
    
    # Create a 3D mask for the structure
    # First, get the dose grid dimensions and parameters
    dose_rows = rd.Rows
    dose_cols = rd.Columns
    dose_frames = rd.NumberOfFrames if hasattr(rd, 'NumberOfFrames') else 1
    
    # Get dose grid coordinates
    dose_x = np.arange(dose_cols) * rd.PixelSpacing[1] + rd.ImagePositionPatient[0]
    dose_y = np.arange(dose_rows) * rd.PixelSpacing[0] + rd.ImagePositionPatient[1]
    dose_z = []
    
    # For 3D dose, get Z coordinates from grid frame offset vector
    if hasattr(rd, 'GridFrameOffsetVector'):
        for i in range(dose_frames):
            dose_z.append(rd.ImagePositionPatient[2] + rd.GridFrameOffsetVector[i])
    else:
        dose_z = [rd.ImagePositionPatient[2]]
    
    # Create an empty mask
    structure_mask = np.zeros((dose_frames, dose_rows, dose_cols), dtype=bool)
    
    # Fill the mask for each contour on each slice
    for contour in contour_data:
        # Find the closest Z slice
        z_coord = contour['z']
        closest_z_idx = np.argmin(np.abs(np.array(dose_z) - z_coord))
        
        # Get x,y points
        points = contour['triplets'][:, :2]
        
        # Create a 2D mask for this contour
        mask_2d = np.zeros((dose_rows, dose_cols), dtype=bool)
        
        # Convert points to pixel coordinates
        pixel_points = np.zeros_like(points)
        pixel_points[:, 0] = np.interp(points[:, 0], dose_x, np.arange(dose_cols))
        pixel_points[:, 1] = np.interp(points[:, 1], dose_y, np.arange(dose_rows))
        
        try:
            # Create a polygon and get a mask
            from matplotlib.path import Path
            poly_path = Path(pixel_points)
            
            # Create a grid of points
            x, y = np.meshgrid(np.arange(dose_cols), np.arange(dose_rows))
            grid_points = np.vstack((x.flatten(), y.flatten())).T
            
            # Check which points are inside the polygon
            mask = poly_path.contains_points(grid_points)
            mask_2d = mask.reshape(dose_rows, dose_cols)
            
            # Add this to the 3D mask
            structure_mask[closest_z_idx] = np.logical_or(structure_mask[closest_z_idx], mask_2d)
        except Exception as e:
            print(f"Error creating mask: {e}")
            continue
    
    # Get the dose values
    dose_values = np.zeros((dose_frames, dose_rows, dose_cols))
    for i in range(dose_frames):
        if dose_frames > 1:
            dose_values[i] = rd.pixel_array[i] * rd.DoseGridScaling
        else:
            dose_values[i] = rd.pixel_array * rd.DoseGridScaling
    
    # Extract dose values within the structure
    structure_doses = dose_values[structure_mask]
    
    return structure_doses

def render_dvh(structure_name, rs_data, rd_data):
    """Render DVH for a structure and return as base64 encoded image"""
    structure_doses = calculate_dvh(structure_name, rs_data, rd_data)
    
    if structure_doses is None or len(structure_doses) == 0:
        return None, None
    
    # Create the DVH
    hist, bin_edges = np.histogram(structure_doses, bins=100, range=(0, np.max(structure_doses)))
    dvh = 1.0 - np.cumsum(hist) / float(len(structure_doses))
    
    # Plot the DVH
    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
    ax.plot(bin_edges[1:], dvh * 100, 'b-', linewidth=2)
    ax.grid(True)
    ax.set_xlabel('Dose (Gy)')
    ax.set_ylabel('Volume (%)')
    ax.set_title(f'DVH for {structure_name}')
    ax.set_xlim([0, np.max(structure_doses)])
    ax.set_ylim([0, 100])
    
    # Save figure to BytesIO
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    
    # Encode the image to base64
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    # Calculate DVH statistics
    dvh_stats = {
        "min_dose": float(np.min(structure_doses)),
        "max_dose": float(np.max(structure_doses)),
        "mean_dose": float(np.mean(structure_doses)),
        "median_dose": float(np.median(structure_doses))
    }
    
    # D95 (dose to 95% of the volume)
    sorted_doses = np.sort(structure_doses)
    d95_index = int(len(sorted_doses) * 0.05)  # 95% from the top
    d95 = sorted_doses[d95_index]
    dvh_stats["d95"] = float(d95)
    
    # V20 (volume receiving 20 Gy or more)
    v20 = np.sum(structure_doses >= 20.0) / len(structure_doses) * 100
    dvh_stats["v20"] = float(v20)
    
    return img_str, dvh_stats

def render_3d_structures(ct_data, rs_data, selected_structures=None):
    """Render 3D visualization of structures and return as base64 encoded image"""
    if not rs_data or not ct_data:
        return None
    
    # Create figure with 3D axes
    fig = plt.figure(figsize=(10, 10), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    # Get patient coordinate system from first CT slice
    first_slice = ct_data[0]['instance']
    last_slice = ct_data[-1]['instance']
    z_min = first_slice.SliceLocation
    z_max = last_slice.SliceLocation
    
    # Get x,y coordinates from first slice
    rows = first_slice.Rows
    cols = first_slice.Columns
    pixel_spacing = first_slice.PixelSpacing
    x_min = first_slice.ImagePositionPatient[0]
    y_min = first_slice.ImagePositionPatient[1]
    x_max = x_min + cols * pixel_spacing[1]
    y_max = y_min + rows * pixel_spacing[0]
    
    # Set the axes limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    
    # Set labels
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    
    # Add a title
    ax.set_title('3D Structure Visualization')
    
    # Process selected structures
    if not selected_structures:
        # If no structures are selected, get all valid structures
        all_structures = list_structures(rs_data)
        selected_structures = [s['name'] for s in all_structures]
    
    # Track which structures were actually rendered
    rendered_structures = []
    
    # Render each structure
    for structure_name in selected_structures:
        contour_data = get_contour_data(rs_data, structure_name)
        if not contour_data or len(contour_data) == 0:
            continue
        
        # Get the color for this structure
        color = contour_data[0]['color']
        
        # Collect all contour points for this structure
        all_points = []
        for contour in contour_data:
            points = contour['triplets']
            
            # Skip contours with too few points
            if len(points) < 3:
                continue
                
            # Add this contour
            all_points.append(points)
        
        # If we have contours, create a 3D visualization
        if all_points:
            for points in all_points:
                # Create a 3D polygon
                x = points[:, 0]
                y = points[:, 1]
                z = points[:, 2]
                
                # Create a Poly3DCollection
                verts = [list(zip(x, y, z))]
                poly = Poly3DCollection(verts, alpha=0.5, facecolor=color, edgecolor='black')
                ax.add_collection3d(poly)
            
            # Add this structure to the rendered list
            rendered_structures.append(structure_name)
    
    # Add a legend for the structures
    if rendered_structures:
        # Create proxy artists for the legend
        proxies = []
        for structure_name in rendered_structures:
            # Find the color for this structure
            color = None
            for contour in get_contour_data(rs_data, structure_name):
                color = contour['color']
                break
            
            if color:
                proxy = plt.Rectangle((0, 0), 1, 1, fc=color)
                proxies.append(proxy)
        
        # Add the legend
        ax.legend(proxies, rendered_structures, loc='upper right')
    
    # Save figure to BytesIO
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    
    # Encode the image to base64
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    return img_str, rendered_structures

def get_patient_info(ct_data):
    """Get patient information from CT data"""
    if not ct_data or len(ct_data) == 0:
        return {}
    
    ct_instance = ct_data[0]['instance']
    patient_info = {
        "patient_id": ct_instance.PatientID if hasattr(ct_instance, 'PatientID') else "Unknown",
        "patient_name": str(ct_instance.PatientName) if hasattr(ct_instance, 'PatientName') else "Unknown",
        "study_date": ct_instance.StudyDate if hasattr(ct_instance, 'StudyDate') else "Unknown",
        "modality": ct_instance.Modality if hasattr(ct_instance, 'Modality') else "Unknown",
        "num_slices": len(ct_data),
        "slice_thickness": f"{ct_instance.SliceThickness}mm" if hasattr(ct_instance, 'SliceThickness') else "Unknown"
    }
    
    return patient_info

def get_plan_info(rp_data):
    """Get plan information from RT Plan data"""
    if not rp_data:
        return {}
    
    rp = rp_data['instance']
    plan_info = {
        "plan_label": rp.RTPlanLabel if hasattr(rp, 'RTPlanLabel') else "Unknown",
        "plan_date": rp.RTPlanDate if hasattr(rp, 'RTPlanDate') else "Unknown",
        "beams": []
    }
    
    # Get prescription information
    if hasattr(rp, 'DoseReferenceSequence'):
        for dose_ref in rp.DoseReferenceSequence:
            if hasattr(dose_ref, 'TargetPrescriptionDose'):
                plan_info["prescription_dose"] = float(dose_ref.TargetPrescriptionDose)
                break
    
    # Get beam information
    if hasattr(rp, 'BeamSequence'):
        beams = rp.BeamSequence
        for beam in beams:
            beam_info = {
                "beam_name": beam.BeamName if hasattr(beam, 'BeamName') else "Unknown",
                "beam_type": beam.BeamType if hasattr(beam, 'BeamType') else "Unknown",
                "radiation_type": beam.RadiationType if hasattr(beam, 'RadiationType') else "Unknown"
            }
            plan_info["beams"].append(beam_info)
    
    return plan_info

# Flask routes
@app.route('/')
def index():
    """Render main page"""
    patients = get_patients()
    return render_template('index.html', patients=patients)

@app.route('/api/patients')
def api_patients():
    """API endpoint to get patient list"""
    patients = get_patients()
    return jsonify(patients)

@app.route('/api/patient/<patient_id>')
def api_patient(patient_id):
    """API endpoint to get patient data"""
    patient_dir = os.path.join(CONFIG['data_dir'], patient_id)
    
    # Load DICOM data
    ct_data = load_ct_files(patient_dir)
    rs_data = load_structure_set(patient_dir)
    rp_data = load_rt_plan(patient_dir)
    rd_data = load_rt_dose(patient_dir)
    
    # Get patient info
    patient_info = get_patient_info(ct_data)
    
    # Get structure list
    structures = list_structures(rs_data) if rs_data else []
    
    # Get plan info
    plan_info = get_plan_info(rp_data) if rp_data else {}
    
    # Get dose info
    has_dose = rd_data is not None
    
    # Get slice range
    slice_range = {
        "min": 0,
        "max": len(ct_data) - 1,
        "count": len(ct_data)
    } if ct_data else {"min": 0, "max": 0, "count": 0}
    
    return jsonify({
        "patient_id": patient_id,
        "patient_info": patient_info,
        "structures": structures,
        "plan_info": plan_info,
        "has_dose": has_dose,
        "slice_range": slice_range,
        "window_presets": CONFIG['window_presets'],
        "dose_colormaps": CONFIG['dose_colormaps']
    })

@app.route('/api/slice', methods=['POST'])
def api_slice():
    """API endpoint to get a CT slice with overlays"""
    data = request.json
    
    # Extract parameters
    patient_id = data.get('patient_id')
    slice_idx = int(data.get('slice_idx', 0))
    window_center = float(data.get('window_center', 40))
    window_width = float(data.get('window_width', 400))
    selected_structures = data.get('structures', [])
    show_dose = data.get('show_dose', False)
    dose_opacity = float(data.get('dose_opacity', 0.5))
    dose_colormap = data.get('dose_colormap', 'jet')
    
    # Load DICOM data
    patient_dir = os.path.join(CONFIG['data_dir'], patient_id)
    ct_data = load_ct_files(patient_dir)
    rs_data = load_structure_set(patient_dir)
    rd_data = load_rt_dose(patient_dir)
    
    # Get CT slice
    if not ct_data or slice_idx < 0 or slice_idx >= len(ct_data):
        return jsonify({"error": "Invalid slice index"}), 400
    
    # Render CT slice
    img_str, structures_shown = render_ct_slice(
        ct_data, 
        slice_idx, 
        window_center, 
        window_width, 
        selected_structures, 
        rs_data, 
        show_dose, 
        dose_opacity, 
        dose_colormap, 
        rd_data
    )
    
    # Get slice information
    slice_info = {
        "index": slice_idx,
        "position": float(ct_data[slice_idx]['instance'].SliceLocation),
        "structures_shown": structures_shown
    }
    
    return jsonify({
        "image": img_str,
        "slice_info": slice_info
    })

@app.route('/api/dvh', methods=['POST'])
def api_dvh():
    """API endpoint to get a DVH for a structure"""
    data = request.json
    
    # Extract parameters
    patient_id = data.get('patient_id')
    structure_name = data.get('structure')
    
    # Load DICOM data
    patient_dir = os.path.join(CONFIG['data_dir'], patient_id)
    rs_data = load_structure_set(patient_dir)
    rd_data = load_rt_dose(patient_dir)
    
    # Check if we have structure and dose data
    if not rs_data or not rd_data:
        return jsonify({"error": "Missing structure or dose data"}), 400
    
    try:
        # Render DVH with error handling
        img_str, dvh_stats = render_dvh(structure_name, rs_data, rd_data)
        
        if img_str is None:
            return jsonify({"error": f"Could not calculate DVH for {structure_name}"}), 400
        
        return jsonify({
            "image": img_str,
            "statistics": dvh_stats
        })
    except Exception as e:
        # Graceful error handling with specific message
        error_msg = str(e)
        print(f"DVH calculation error for {structure_name}: {error_msg}")
        return jsonify({
            "error": f"Error calculating DVH for {structure_name}. The structure may be missing required data or have an invalid format."
        }), 400

@app.route('/api/3d', methods=['POST'])
def api_3d():
    """API endpoint to get a 3D visualization of structures"""
    data = request.json
    
    # Extract parameters
    patient_id = data.get('patient_id')
    selected_structures = data.get('structures', [])
    
    # Load DICOM data
    patient_dir = os.path.join(CONFIG['data_dir'], patient_id)
    ct_data = load_ct_files(patient_dir)
    rs_data = load_structure_set(patient_dir)
    
    # Check if we have structure data
    if not rs_data or not ct_data:
        return jsonify({"error": "Missing CT or structure data"}), 400
    
    try:
        # Render 3D visualization
        img_str, structures_shown = render_3d_structures(ct_data, rs_data, selected_structures)
        
        if img_str is None:
            return jsonify({"error": "Could not render 3D structures"}), 400
        
        return jsonify({
            "image": img_str,
            "structures_shown": structures_shown
        })
    except Exception as e:
        # Graceful error handling
        error_msg = str(e)
        print(f"3D rendering error: {error_msg}")
        return jsonify({
            "error": f"Error rendering 3D structures: {error_msg}"
        }), 400

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    if os.environ.get('FLASK_ENV') == 'production':
        # Production
        from waitress import serve
        serve(app, host='0.0.0.0', port=5000)
    else:
        # Development
        app.run(debug=True, host='0.0.0.0')