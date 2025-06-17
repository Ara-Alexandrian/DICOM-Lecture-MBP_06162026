# Building a DICOM Viewer with Streamlit

In this part of the workshop, we'll create a simple but powerful DICOM viewer web application using Streamlit. This will allow users to:

1. Load DICOM CT images, structures, plans, and doses
2. Browse through CT slices
3. Overlay RT structures
4. View dose distributions
5. Generate simple DVHs

## Setup and Installation

First, let's create a new file called `dicom_viewer.py` in the root directory:

```python
import streamlit as st
import pydicom
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import pandas as pd
from io import BytesIO
```

## Application Structure

```python
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
```

## Patient Selection

```python
# Function to list available patients
def list_patients():
    data_dir = "data/test-cases"
    if not os.path.exists(data_dir):
        st.error(f"Data directory not found: {data_dir}")
        return []
    
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
```

## DICOM File Loading Functions

```python
# Function to load CT images
@st.cache_data
def load_ct_files(patient_dir):
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
                'thickness': float(ds.SliceThickness)
            })
        except Exception as e:
            st.warning(f"Error loading {ct_file}: {str(e)}")
    
    # Sort by slice location
    ct_data.sort(key=lambda x: x['z'])
    
    return ct_data

# Function to load RT Structure Set
@st.cache_data
def load_structure_set(patient_dir):
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
```

## CT Image Display Functions

```python
# Function to convert pixel data to Hounsfield Units (HU)
def get_hu_values(ct_slice):
    # Get pixel values
    pixel_array = ct_slice.pixel_array
    
    # Convert to HU using slope and intercept
    intercept = ct_slice.RescaleIntercept
    slope = ct_slice.RescaleSlope
    
    hu_values = pixel_array * slope + intercept
    return hu_values

# Function to apply window/level to the image
def apply_window_level(hu_image, window_center, window_width):
    min_value = window_center - window_width/2
    max_value = window_center + window_width/2
    hu_image_windowed = np.clip(hu_image, min_value, max_value)
    return hu_image_windowed

# Function to display CT slice
def display_ct_slice(ct_slice, window_center, window_width, structures_to_show=None, rs=None, dose_overlay=None):
    # Get HU values
    hu_image = get_hu_values(ct_slice)
    
    # Apply window/level
    hu_image_windowed = apply_window_level(hu_image, window_center, window_width)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(hu_image_windowed, cmap='gray')
    
    # Add structures if requested
    if structures_to_show and rs:
        current_z = ct_slice.SliceLocation
        
        for structure_name in structures_to_show:
            contour_data = get_contour_data(rs, structure_name)
            if not contour_data:
                continue
            
            # Find contours on this slice
            contours_on_slice = [c for c in contour_data if np.isclose(c['z'], current_z, atol=0.5)]
            
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
                collection = PatchCollection(patches, alpha=0.3, color=color, linewidths=2, edgecolor=color)
                ax.add_collection(collection)
    
    # Add dose overlay if requested
    if dose_overlay is not None:
        # We assume dose_overlay is already aligned with the CT slice
        # and has the same dimensions
        # This is a simplification; in practice, you'd need to interpolate the dose to the CT grid
        ax.imshow(dose_overlay, cmap='jet', alpha=0.5)
    
    ax.set_title(f"CT Slice at {ct_slice.SliceLocation}mm")
    ax.axis('off')
    
    return fig
```

## Structure Functions

```python
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
```

## Dose Functions

```python
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
```

## Main Application Logic

```python
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

# Slice selection
min_slice = 0
max_slice = len(ct_data) - 1
slice_idx = st.sidebar.slider("CT Slice", min_slice, max_slice, max_slice // 2)
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
    window_center = st.sidebar.slider("Window Center", -1000, 3000, 40)
    window_width = st.sidebar.slider("Window Width", 1, 4000, 400)
else:
    window_center, window_width = window_presets[selected_preset]

# Structure overlay selection
if rs_data:
    structures = list_structures(rs_data)
    
    # Multiselect for structures
    selected_structures = st.sidebar.multiselect(
        "Overlay Structures",
        structures
    )
else:
    selected_structures = []

# Dose overlay toggle
show_dose = False
if rd_data:
    show_dose = st.sidebar.checkbox("Show Dose Overlay", False)

# Main display area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("CT Image")
    
    # Get dose overlay if requested
    dose_overlay = None
    if show_dose and rd_data:
        dose_overlay = get_dose_slice(rd_data, current_slice.SliceLocation)
    
    # Display the CT slice with overlays
    fig = display_ct_slice(
        current_slice, 
        window_center, 
        window_width, 
        selected_structures, 
        rs_data, 
        dose_overlay
    )
    
    # Display the figure
    st.pyplot(fig)
    
    # Display slice information
    st.info(f"Slice {slice_idx+1}/{len(ct_data)} at position {current_slice.SliceLocation}mm")

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
```

## Running the Application

To run the Streamlit application, save the above code to a file named `dicom_viewer.py` and run:

```bash
pip install streamlit pydicom matplotlib numpy pandas scikit-image
streamlit run dicom_viewer.py
```

## Further Enhancements

Here are some ideas for enhancing the viewer:

1. **3D Visualization**: Add 3D rendering of structures using plotly or VTK
2. **Plan Comparison**: Allow loading multiple plans for comparison
3. **Registration**: Support for registering and displaying multiple image series
4. **Export**: Add options to export images, structures, and DVHs
5. **Contouring**: Add simple manual contouring capabilities
6. **Isodose Lines**: Display isodose lines on the CT images
7. **DICOM Import**: Add direct DICOM import functionality
8. **Plan Evaluation**: Add tools for automatic plan evaluation

## Conclusion

This Streamlit application provides a simple but effective way to explore RT DICOM data. The interactive nature of Streamlit makes it easy to adjust display parameters and view different aspects of the treatment plan.

For a full-featured clinical application, additional features like proper registration, advanced visualization, and comprehensive plan evaluation would be needed, but this serves as a good starting point for educational purposes.