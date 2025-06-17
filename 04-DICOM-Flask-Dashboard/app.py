import os
import pydicom
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
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
            # Get the color
            color = [c/255 for c in roi_contour.ROIDisplayColor]
            
            # Get the contour sequence
            if hasattr(roi_contour, 'ContourSequence'):
                for contour in roi_contour.ContourSequence:
                    contour_coords = contour.ContourData
                    # Reshape into x,y,z triplets
                    triplets = np.array(contour_coords).reshape((-1, 3))
                    contour_data.append({
                        'triplets': triplets,
                        'z': triplets[0, 2],  # Z coordinate (slice location)
                        'color': color
                    })
    
    return contour_data

def list_structures(rs_instance):
    """List all structures in an RT Structure Set"""
    if not rs_instance:
        return []
    
    rs = rs_instance['instance']
    structures = []
    
    if hasattr(rs, 'StructureSetROISequence'):
        for roi in rs.StructureSetROISequence:
            if hasattr(rs, 'ROIContourSequence'):
                for roi_contour in rs.ROIContourSequence:
                    if roi_contour.ReferencedROINumber == roi.ROINumber:
                        if hasattr(roi_contour, 'ROIDisplayColor'):
                            color = [c/255 for c in roi_contour.ROIDisplayColor]
                            hex_color = "#{:02x}{:02x}{:02x}".format(
                                int(color[0]*255), int(color[1]*255), int(color[2]*255))
                        else:
                            hex_color = "#FFFFFF"
                        structures.append({
                            'number': roi.ROINumber,
                            'name': roi.ROIName,
                            'color': hex_color
                        })
                        break
    
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

    # Track which structures are actually shown for reporting
    structures_shown = []

    # Add structures if requested
    if structures and rs_data:
        current_z = current_slice.SliceLocation

        # Calculate vertical spacing for structure labels
        label_y_offset = 30

        # Sort structures alphabetically for consistent display
        sorted_structures = sorted(structures)

        for i, structure_name in enumerate(sorted_structures):
            contour_data = get_contour_data(rs_data, structure_name)
            if not contour_data:
                continue

            # Find contours on this slice (with increased tolerance for small structures)
            # Use a larger tolerance (2.0mm) to ensure small structures are visible
            contours_on_slice = [c for c in contour_data if np.isclose(c['z'], current_z, atol=2.0)]

            # Add contours to the plot
            patches = []
            for contour in contours_on_slice:
                # Extract x,y coordinates
                points = contour['triplets'][:, :2]
                polygon = Polygon(points, closed=True)
                patches.append(polygon)

            # Create a collection and add it to the plot
            if patches:
                color = contour_data[0]['color']

                # Enhanced visibility:
                # - Higher alpha value (0.6) for better visibility of the structure fill
                # - Thicker linewidth (2.5) with black edge outline for better contrast
                # - Different edge color for better visibility against both bright and dark backgrounds
                edge_color = 'black'  # Black edges show up well against both bright and dark areas
                collection = PatchCollection(
                    patches,
                    alpha=0.6,  # More opaque for better visibility
                    color=color,
                    linewidths=2.5,  # Thicker lines
                    edgecolor=edge_color
                )
                ax.add_collection(collection)

                # Add structure name label to image with vertical spacing
                # Improved label visibility with contrasting outline
                label_y = label_y_offset + (i * 25)  # Offset each label
                ax.text(10, label_y, structure_name,
                       color='white', fontsize=14, fontweight='bold',
                       bbox=dict(facecolor=color, alpha=0.7, edgecolor='black', pad=3))

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
    
    # Render DVH
    img_str, dvh_stats = render_dvh(structure_name, rs_data, rd_data)
    
    if img_str is None:
        return jsonify({"error": f"Could not calculate DVH for {structure_name}"}), 400
    
    return jsonify({
        "image": img_str,
        "statistics": dvh_stats
    })

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