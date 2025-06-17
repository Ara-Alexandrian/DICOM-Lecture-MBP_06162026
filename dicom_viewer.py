# RT DICOM Viewer - Streamlit Application
# This application provides a web-based viewer for radiotherapy DICOM files
# It supports viewing of CT images, RT Structure Sets, RT Plans, and RT Dose distributions
# The viewer includes features like window/level adjustment, structure overlays, and DVH calculation

import streamlit as st         # Web application framework
import pydicom                 # For reading DICOM files
import os                      # File system operations
import numpy as np             # Numerical processing
import matplotlib.pyplot as plt                # Plotting library
from matplotlib.patches import Polygon         # For structure contour rendering
from matplotlib.collections import PatchCollection  # For grouping contour polygons
import pandas as pd            # For data manipulation and display
from matplotlib.path import Path              # For polygon masking in DVH calculation
import matplotlib.colors as mcolors           # For color mapping of dose overlays
from matplotlib.figure import Figure          # For plot generation
from io import BytesIO                        # For image handling

# Set page configuration
st.set_page_config(
    page_title="RT DICOM Viewer",
    page_icon="🏥",
    layout="wide",
)

# Add title and description
st.title("RT DICOM Viewer")
st.markdown("""
This application allows you to explore radiotherapy DICOM files including:
- CT images
- RT Structure Sets
- RT Plans
- RT Dose distributions
""")

# Function to list available patients from the data directory
def list_patients():
    """
    Scans the data directory for patient folders and returns a list of patient IDs

    Returns:
        list: A list of patient IDs found in the data directory
    """
    data_dir = "data/test-cases"  # Path to the patient data directory

    # Check if the data directory exists
    if not os.path.exists(data_dir):
        st.error(f"Data directory not found: {data_dir}")
        return []

    # List all directories in the data directory (each directory is a patient)
    patients = [d for d in os.listdir(data_dir)
               if os.path.isdir(os.path.join(data_dir, d))]
    return patients

# Patient selection
patients = list_patients()
if not patients:
    st.error("No patient data found!")
    st.stop()

selected_patient = st.sidebar.selectbox(
    "Select Patient", 
    patients
)

# Display the selected patient data directory
patient_dir = os.path.join("data/test-cases", selected_patient)
st.sidebar.info(f"Patient Directory: {patient_dir}")

# Function to load CT images from a patient directory
@st.cache_data  # Streamlit caching to avoid reloading the same files
def load_ct_files(patient_dir):
    """
    Loads all CT image files from a patient directory

    Args:
        patient_dir (str): Path to the patient directory containing DICOM files

    Returns:
        list: A list of dictionaries containing CT slice information and DICOM objects
              Each dictionary includes: file name, path, DICOM instance, z position, and slice thickness
    """
    # Find all CT image files in the directory
    ct_files = [f for f in os.listdir(patient_dir) if f.startswith("CT")]
    ct_files.sort()  # Sort filenames to get them in rough order

    ct_data = []
    for ct_file in ct_files:
        try:
            # Read the DICOM file
            ds = pydicom.dcmread(os.path.join(patient_dir, ct_file))

            # Extract and store key information
            ct_data.append({
                'file': ct_file,                           # File name
                'path': os.path.join(patient_dir, ct_file), # Full file path
                'instance': ds,                            # DICOM dataset object
                'z': float(ds.SliceLocation),              # Z coordinate (slice position)
                'thickness': float(ds.SliceThickness)      # Slice thickness in mm
            })
        except Exception as e:
            # Handle any errors during file loading
            st.warning(f"Error loading {ct_file}: {str(e)}")

    # Sort slices by z position to ensure proper order
    ct_data.sort(key=lambda x: x['z'])

    return ct_data

# Function to load RT Structure Set from a patient directory
@st.cache_data  # Streamlit caching to avoid reloading the same file
def load_structure_set(patient_dir):
    """
    Loads the RT Structure Set file from a patient directory

    Args:
        patient_dir (str): Path to the patient directory containing DICOM files

    Returns:
        dict or None: A dictionary containing RT Structure Set information and DICOM object,
                     or None if no structure set is found
    """
    # Find all RT Structure Set files (they start with "RS")
    rs_files = [f for f in os.listdir(patient_dir) if f.startswith("RS")]

    # If no structure set found, return None
    if not rs_files:
        return None

    # Use the first structure set file (usually there's only one)
    rs_path = os.path.join(patient_dir, rs_files[0])
    try:
        # Read the DICOM file
        rs = pydicom.dcmread(rs_path)

        # Return a dictionary with the structure set information
        return {
            'file': rs_files[0],     # File name
            'path': rs_path,         # Full file path
            'instance': rs           # DICOM dataset object
        }
    except Exception as e:
        # Handle any errors during file loading
        st.warning(f"Error loading structure set: {str(e)}")
        return None

# Function to load RT Plan
@st.cache_data
def load_rt_plan(patient_dir):
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
        st.warning(f"Error loading RT plan: {str(e)}")
        return None

# Function to load RT Dose
@st.cache_data
def load_rt_dose(patient_dir):
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
        st.warning(f"Error loading RT dose: {str(e)}")
        return None

# Function to convert CT pixel data to Hounsfield Units (HU)
def get_hu_values(ct_slice):
    """
    Converts raw CT pixel data to Hounsfield Units (HU) using the DICOM rescale parameters

    Hounsfield Units are a standardized scale for reporting CT numbers, where:
    - Water is 0 HU
    - Air is approximately -1000 HU
    - Bone is typically +400 to +1000 HU

    Args:
        ct_slice (pydicom.dataset.FileDataset): DICOM CT slice object

    Returns:
        numpy.ndarray: 2D array of pixel values converted to Hounsfield Units
    """
    # Get raw pixel values from the DICOM object
    pixel_array = ct_slice.pixel_array

    # Get rescale parameters from the DICOM header
    intercept = ct_slice.RescaleIntercept  # Usually -1024 for CT
    slope = ct_slice.RescaleSlope          # Usually 1 for CT

    # Apply the rescale formula: HU = pixel_value * slope + intercept
    hu_values = pixel_array * slope + intercept
    return hu_values

# Function to apply window/level (brightness/contrast) to the CT image
def apply_window_level(hu_image, window_center, window_width):
    """
    Applies windowing to a CT image to enhance visualization of specific tissue types

    Window/level (also known as window width/window center) controls how HU values
    are mapped to the grayscale display. Different window settings are used to
    visualize different anatomical structures:
    - Soft tissue: window center ~40, window width ~400
    - Lung: window center ~-600, window width ~1500
    - Bone: window center ~500, window width ~2000

    Args:
        hu_image (numpy.ndarray): 2D array of Hounsfield Units
        window_center (float): Center of the window (level)
        window_width (float): Width of the window (contrast)

    Returns:
        numpy.ndarray: Windowed image with values clipped to the specified range
    """
    # Calculate window boundaries
    min_value = window_center - window_width/2  # Lower bound
    max_value = window_center + window_width/2  # Upper bound

    # Clip HU values to the window range
    # Values below min_value will be set to min_value
    # Values above max_value will be set to max_value
    hu_image_windowed = np.clip(hu_image, min_value, max_value)

    return hu_image_windowed

# Function to get contour data for a specific structure
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

# Function to list all structures in an RT Structure Set
def list_structures(rs_instance):
    if not rs_instance:
        return []
    
    rs = rs_instance['instance']
    structures = []
    
    if hasattr(rs, 'StructureSetROISequence'):
        for roi in rs.StructureSetROISequence:
            structures.append(roi.ROIName)
    
    return structures

# Function to display CT slice with structure overlays and dose information
def display_ct_slice(ct_slice, window_center, window_width, structures_to_show=None, rs=None,
                    dose_overlay=None, dose_opacity=0.5, dose_colormap='jet', dose_max=None):
    """
    Renders a CT slice with optional structure contours and dose overlay

    This function creates a visualization of a CT slice with customizable
    window/level settings. It can also overlay RT structure contours and
    RT dose distribution if provided.

    Args:
        ct_slice (pydicom.dataset.FileDataset): DICOM CT slice object
        window_center (float): Center of the window (level)
        window_width (float): Width of the window (contrast)
        structures_to_show (list, optional): List of structure names to overlay
        rs (dict, optional): RT Structure Set data
        dose_overlay (numpy.ndarray, optional): 2D array of dose values
        dose_opacity (float, optional): Opacity of dose overlay (0.0-1.0)
        dose_colormap (str, optional): Matplotlib colormap name for dose display
        dose_max (float, optional): Maximum dose value for colormap scaling

    Returns:
        tuple: (matplotlib figure, list of structures actually displayed)
    """
    # Convert pixel data to Hounsfield Units
    hu_image = get_hu_values(ct_slice)

    # Apply window/level settings for visualization
    hu_image_windowed = apply_window_level(hu_image, window_center, window_width)

    # Create figure and plot the CT image
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(hu_image_windowed, cmap='gray')

    # Track which structures are actually shown for reporting
    structures_shown = []

    # Add structure contours if requested
    if structures_to_show and rs:
        current_z = ct_slice.SliceLocation  # Z-position of current slice

        # Calculate vertical spacing for structure labels
        label_y_offset = 30

        # Process each requested structure
        for i, structure_name in enumerate(structures_to_show):
            contour_data = get_contour_data(rs, structure_name)
            if not contour_data:
                continue

            # Find contours on this slice (with increased tolerance for small structures)
            # The 1.0mm tolerance allows for small differences in slice positioning
            contours_on_slice = [c for c in contour_data if np.isclose(c['z'], current_z, atol=1.0)]

            # Create polygons for each contour
            patches = []
            for contour in contours_on_slice:
                # Extract x,y coordinates from the 3D points
                points = contour['triplets'][:, :2]
                polygon = Polygon(points, closed=True)
                patches.append(polygon)

            # Add contours to the plot if any were found
            if patches:
                color = contour_data[0]['color']  # Use the color from DICOM
                collection = PatchCollection(
                    patches,
                    alpha=0.5,              # Semi-transparent fill
                    color=color,            # Structure color
                    linewidths=2,           # Contour line width
                    edgecolor=color         # Contour line color
                )
                ax.add_collection(collection)

                # Add structure name label to image with vertical spacing
                label_y = label_y_offset + (i * 25)  # Offset each label
                ax.text(10, label_y, structure_name,
                       color='white', fontsize=14,
                       bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', pad=3))

                structures_shown.append(structure_name)

    # Add dose overlay if provided
    dose_display = None
    if dose_overlay is not None:
        # Use the provided dose_max (global max) for consistent colormap scaling across slices
        if dose_max is not None and dose_max > 0:
            # Create the dose overlay with specified colormap and opacity
            cmap = plt.get_cmap(dose_colormap)
            norm = mcolors.Normalize(vmin=0, vmax=dose_max)  # Scale from 0 to max dose
            dose_display = ax.imshow(dose_overlay, cmap=cmap, norm=norm, alpha=dose_opacity)

            # Add colorbar to show the dose scale
            cbar = fig.colorbar(dose_display, ax=ax, orientation='vertical', pad=0.01)
            cbar.set_label('Dose (Gy)', rotation=270, labelpad=15)

    # Add slice position info as the title
    slice_info = f"CT Slice at {ct_slice.SliceLocation}mm"
    ax.set_title(slice_info)
    ax.axis('off')  # Hide axes for cleaner visualization

    return fig, structures_shown

# Function to get dose slice for a specific z position
def get_dose_slice(rd_instance, z_position):
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

# Function to get global dose max across all slices
def get_global_dose_max(rd_instance):
    if not rd_instance:
        return 0
    
    rd = rd_instance['instance']
    num_frames = rd.NumberOfFrames if hasattr(rd, 'NumberOfFrames') else 1
    
    if num_frames > 1:
        dose_max = np.max(rd.pixel_array) * rd.DoseGridScaling
    else:
        dose_max = np.max(rd.pixel_array) * rd.DoseGridScaling
    
    return dose_max

# Function to calculate a DVH for a structure
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
        
        # Create a polygon and get a mask
        poly_path = Path(pixel_points)
        
        # Create a grid of points
        x, y = np.meshgrid(np.arange(dose_cols), np.arange(dose_rows))
        grid_points = np.vstack((x.flatten(), y.flatten())).T
        
        # Check which points are inside the polygon
        mask = poly_path.contains_points(grid_points)
        mask_2d = mask.reshape(dose_rows, dose_cols)
        
        # Add this to the 3D mask
        structure_mask[closest_z_idx] = np.logical_or(structure_mask[closest_z_idx], mask_2d)
    
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

# Load DICOM data
with st.spinner("Loading DICOM data..."):
    ct_data = load_ct_files(patient_dir)
    rs_data = load_structure_set(patient_dir)
    rp_data = load_rt_plan(patient_dir)
    rd_data = load_rt_dose(patient_dir)

# Check if we have CT data
if not ct_data:
    st.error("No CT images found!")
    st.stop()

# Create sidebar controls
st.sidebar.header("Display Options")

# Patient info
with st.sidebar.expander("Patient Info", expanded=False):
    if len(ct_data) > 0:
        ct_instance = ct_data[0]['instance']
        patient_info = {
            "Patient ID": ct_instance.PatientID if hasattr(ct_instance, 'PatientID') else "Unknown",
            "Patient Name": str(ct_instance.PatientName) if hasattr(ct_instance, 'PatientName') else "Unknown",
            "Study Date": ct_instance.StudyDate if hasattr(ct_instance, 'StudyDate') else "Unknown",
        }
        
        for key, value in patient_info.items():
            st.text(f"{key}: {value}")

# Slice selection - both slider and mouse wheel scrolling will be enabled
min_slice = 0
max_slice = len(ct_data) - 1
slice_idx = st.sidebar.slider("CT Slice", min_slice, max_slice, max_slice // 2, key="slice_slider")
current_slice = ct_data[slice_idx]['instance']

# Window/level selection
window_presets = {
    "Soft Tissue": (40, 400),
    "Lung": (-600, 1500),
    "Bone": (500, 2000),
    "Brain": (40, 80),
    "Custom": "custom"
}

selected_preset = st.sidebar.selectbox(
    "Window Preset", 
    list(window_presets.keys())
)

if selected_preset == "Custom":
    col1, col2 = st.sidebar.columns(2)
    with col1:
        window_center = st.slider("Center", -1000, 3000, 40)
    with col2:
        window_width = st.slider("Width", 1, 4000, 400)
else:
    window_center, window_width = window_presets[selected_preset]

# Structure overlay selection
if rs_data:
    structures = list_structures(rs_data)
    
    # Multiselect for structures with color preview
    selected_structures = st.sidebar.multiselect(
        "Overlay Structures",
        structures
    )

    # Add structure info
    if len(selected_structures) > 0:
        st.sidebar.text(f"Selected {len(selected_structures)} structures")
else:
    selected_structures = []

# Dose overlay options
show_dose = False
dose_opacity = 0.5
dose_colormap = "jet"
if rd_data:
    show_dose = st.sidebar.checkbox("Show Dose Overlay", False)
    if show_dose:
        dose_opacity = st.sidebar.slider("Dose Opacity", 0.1, 1.0, 0.5)
        
        # Dose colormap selection
        dose_colormaps = ["jet", "hot", "plasma", "viridis", "turbo", "rainbow"]
        dose_colormap = st.sidebar.selectbox("Dose Colormap", dose_colormaps, index=0)

# Main display area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("CT Image")
    
    # Get dose overlay if requested
    dose_overlay = None
    if show_dose and rd_data:
        dose_overlay = get_dose_slice(rd_data, current_slice.SliceLocation)
    
    # Get global dose max for consistent colorbar across slices
    global_dose_max = None
    if show_dose and rd_data:
        global_dose_max = get_global_dose_max(rd_data)
    
    # Display the CT slice with overlays
    fig, structures_shown = display_ct_slice(
        current_slice, 
        window_center, 
        window_width, 
        selected_structures, 
        rs_data, 
        dose_overlay,
        dose_opacity=dose_opacity,
        dose_colormap=dose_colormap,
        dose_max=global_dose_max
    )
    
    # Display the figure
    st.pyplot(fig)
    
    # Add a slider directly under the image for quick navigation
    st.slider(
        "Quick Navigation", 
        min_slice, 
        max_slice, 
        slice_idx, 
        key="quick_nav",
        on_change=lambda: st.session_state.update({"slice_slider": st.session_state.quick_nav})
    )
    
    # Display slice information
    slice_info = f"Slice {slice_idx+1}/{len(ct_data)} at position {current_slice.SliceLocation}mm"
    
    # Add window/level info
    if selected_preset == "Custom":
        slice_info += f" | Window/Level: {window_width}/{window_center}"
    else:
        slice_info += f" | {selected_preset} window"
        
    # Add structure info if any are shown
    if structures_shown:
        slice_info += f" | Showing {len(structures_shown)} structures"
        
    st.info(slice_info)

with col2:
    # Tabs for different data displays
    tab1, tab2, tab3 = st.tabs(["Patient Info", "Plan Details", "DVH"])
    
    with tab1:
        # Display patient information
        st.subheader("Patient Information")
        if len(ct_data) > 0:
            ct_instance = ct_data[0]['instance']
            patient_info = {
                "Patient ID": ct_instance.PatientID if hasattr(ct_instance, 'PatientID') else "Unknown",
                "Patient Name": str(ct_instance.PatientName) if hasattr(ct_instance, 'PatientName') else "Unknown",
                "Study Date": ct_instance.StudyDate if hasattr(ct_instance, 'StudyDate') else "Unknown",
                "Modality": ct_instance.Modality if hasattr(ct_instance, 'Modality') else "Unknown",
                "Number of Slices": len(ct_data),
                "Slice Thickness": f"{ct_instance.SliceThickness}mm" if hasattr(ct_instance, 'SliceThickness') else "Unknown",
            }
            
            for key, value in patient_info.items():
                st.text(f"{key}: {value}")
    
    with tab2:
        # Display plan information
        st.subheader("Plan Details")
        
        if rp_data:
            rp = rp_data['instance']
            
            # Display basic plan info
            st.text(f"Plan Label: {rp.RTPlanLabel if hasattr(rp, 'RTPlanLabel') else 'Unknown'}")
            st.text(f"Plan Date: {rp.RTPlanDate if hasattr(rp, 'RTPlanDate') else 'Unknown'}")
            
            # Display prescription information
            if hasattr(rp, 'DoseReferenceSequence'):
                st.subheader("Prescription")
                for dose_ref in rp.DoseReferenceSequence:
                    if hasattr(dose_ref, 'TargetPrescriptionDose'):
                        st.text(f"Target Dose: {dose_ref.TargetPrescriptionDose} Gy")
            
            # Display beam information
            if hasattr(rp, 'BeamSequence'):
                beams = rp.BeamSequence
                st.subheader(f"Beams ({len(beams)})")
                
                beam_data = []
                for beam in beams:
                    beam_data.append({
                        "Beam Name": beam.BeamName if hasattr(beam, 'BeamName') else "Unknown",
                        "Beam Type": beam.BeamType if hasattr(beam, 'BeamType') else "Unknown",
                        "Radiation Type": beam.RadiationType if hasattr(beam, 'RadiationType') else "Unknown"
                    })
                
                st.dataframe(pd.DataFrame(beam_data))
        else:
            st.info("No RT Plan data available")
    
    with tab3:
        # DVH Display
        st.subheader("Dose Volume Histogram")
        
        if rs_data and rd_data:
            # Select a structure for DVH
            dvh_structure = st.selectbox(
                "Select Structure for DVH",
                structures if structures else []
            )
            
            if dvh_structure:
                # Calculate DVH
                with st.spinner(f"Calculating DVH for {dvh_structure}..."):
                    structure_doses = calculate_dvh(dvh_structure, rs_data, rd_data)
                
                if structure_doses is not None and len(structure_doses) > 0:
                    # Create the DVH
                    hist, bin_edges = np.histogram(structure_doses, bins=100, range=(0, np.max(structure_doses)))
                    dvh = 1.0 - np.cumsum(hist) / float(len(structure_doses))
                    
                    # Plot the DVH
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.plot(bin_edges[1:], dvh * 100, 'b-', linewidth=2)
                    ax.grid(True)
                    ax.set_xlabel('Dose (Gy)')
                    ax.set_ylabel('Volume (%)')
                    ax.set_title(f'DVH for {dvh_structure}')
                    ax.set_xlim([0, np.max(structure_doses)])
                    ax.set_ylim([0, 100])
                    
                    st.pyplot(fig)
                    
                    # Display DVH statistics
                    st.subheader("DVH Statistics")
                    dvh_stats = {
                        "Min Dose": f"{np.min(structure_doses):.2f} Gy",
                        "Max Dose": f"{np.max(structure_doses):.2f} Gy",
                        "Mean Dose": f"{np.mean(structure_doses):.2f} Gy",
                        "Median Dose": f"{np.median(structure_doses):.2f} Gy"
                    }
                    
                    # D95 (dose to 95% of the volume)
                    sorted_doses = np.sort(structure_doses)
                    d95_index = int(len(sorted_doses) * 0.05)  # 95% from the top
                    d95 = sorted_doses[d95_index]
                    dvh_stats["D95"] = f"{d95:.2f} Gy"
                    
                    # V20 (volume receiving 20 Gy or more)
                    v20 = np.sum(structure_doses >= 20.0) / len(structure_doses) * 100
                    dvh_stats["V20Gy"] = f"{v20:.2f}%"
                    
                    for key, value in dvh_stats.items():
                        st.text(f"{key}: {value}")
                else:
                    st.warning(f"Could not calculate DVH for {dvh_structure}")
        else:
            st.info("RT Structure Set and/or RT Dose data required for DVH calculation")