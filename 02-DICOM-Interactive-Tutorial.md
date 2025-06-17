# Interactive DICOM Tutorial

This tutorial guides you through working with RT DICOM files in Python. Each section is designed as a Jupyter-style cell that can be copied and executed.

## Setup

```python
# Install required libraries
!pip install pydicom matplotlib numpy pandas scikit-image

# Import libraries
import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from pydicom.dataset import Dataset, FileDataset
from pydicom.sequence import Sequence
import pandas as pd
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
```

## 1. Loading DICOM Images

```python
# Define path to the CT dataset
patient_id = "01149188"  # Change this to the patient you want to analyze
path = f"data/test-cases/{patient_id}/"

# Find all CT slices
ct_files = [f for f in os.listdir(path) if f.startswith("CT")]
ct_files.sort()  # Sort to get them in order

# Load the middle slice for demonstration
middle_index = len(ct_files) // 2
middle_slice_path = os.path.join(path, ct_files[middle_index])
ct_slice = pydicom.dcmread(middle_slice_path)

# Display basic info
print(f"Patient ID: {ct_slice.PatientID}")
print(f"Modality: {ct_slice.Modality}")
print(f"Slice Location: {ct_slice.SliceLocation}")
print(f"Pixel Spacing: {ct_slice.PixelSpacing}")
print(f"Total slices: {len(ct_files)}")
```

## 2. Visualizing a CT Slice

```python
# Function to convert pixel data to Hounsfield Units (HU)
def get_hu_values(ct_slice):
    """
    Convert CT pixel data to Hounsfield Units using the rescale slope and intercept

    Hounsfield Units (HU) are a standardized scale for reporting CT numbers:
    - Air: approximately -1000 HU
    - Water: 0 HU
    - Soft tissue: 20-100 HU
    - Bone: 300-1900 HU
    """
    # Get pixel values
    pixel_array = ct_slice.pixel_array

    # Convert to HU using slope and intercept
    intercept = ct_slice.RescaleIntercept
    slope = ct_slice.RescaleSlope

    hu_values = pixel_array * slope + intercept
    return hu_values

# Get HU values and display the image
hu_image = get_hu_values(ct_slice)

# Set window/level for better visualization (lung window)
window_center = -600  # Center of the window in HU
window_width = 1500   # Width of the window in HU

# Apply window/level
# This controls the brightness/contrast by limiting displayed HU range:
# - Values below (window_center - window_width/2) appear black
# - Values above (window_center + window_width/2) appear white
# - Values in between are mapped to grayscale
min_value = window_center - window_width/2
max_value = window_center + window_width/2
hu_image_windowed = np.clip(hu_image, min_value, max_value)

# Common window/level presets:
# - Lung: -600/1500 (shows air-filled structures)
# - Soft tissue: 40/400 (optimal for organs)
# - Bone: 500/2000 (best for visualizing bones)
# - Brain: 40/80 (for brain tissue contrast)

# Display image
plt.figure(figsize=(10, 10))
plt.imshow(hu_image_windowed, cmap='gray')
plt.title(f"CT Slice at {ct_slice.SliceLocation}mm")
plt.axis('off')
plt.colorbar(label='HU')
plt.show()
```

## 3. Exploring DICOM Metadata

```python
# Function to display DICOM tags in a user-friendly way
def explore_dicom_tags(ds, max_depth=2, current_depth=0, prefix=''):
    """
    Recursively explore DICOM tags with controlled depth

    Parameters:
        ds: The DICOM dataset to explore
        max_depth: Maximum recursion depth for nested sequences (default=2)
                   Controls how deep to go into nested DICOM sequences:
                   - 0: Only show top-level tags
                   - 1: Show one level of sequence items
                   - 2+: Show deeper nested sequences
        current_depth: Current recursion depth (used internally)
        prefix: String prefix for indentation (used internally)

    Returns:
        List of formatted tag descriptions
    """
    tags_info = []

    # Skip Pixel Data because it's too large
    skip_tags = [(0x7FE0, 0x0010)]  # Pixel Data

    for elem in ds:
        if elem.tag in skip_tags:
            tags_info.append(f"{prefix}{elem.tag}: [Pixel Data]")
            continue

        # Format the element
        if elem.VR == "SQ":  # Sequence
            tags_info.append(f"{prefix}{elem.tag} {elem.name}: Sequence with {len(elem.value)} item(s)")
            if current_depth < max_depth:
                for i, item in enumerate(elem.value):
                    tags_info.append(f"{prefix}  Item {i}:")
                    tags_info.extend(explore_dicom_tags(item, max_depth, current_depth + 1, prefix + '    '))
        else:
            if elem.VM > 1:  # Multiple values
                tags_info.append(f"{prefix}{elem.tag} {elem.name}: {elem.repval}")
            else:
                tags_info.append(f"{prefix}{elem.tag} {elem.name}: {elem.repval}")

    return tags_info

# Display the first 20 tags
tags = explore_dicom_tags(ct_slice, max_depth=1)
for tag in tags[:20]:
    print(tag)

print("...")
print(f"Total tags: {len(tags)}")
```

## 4. Searching for Specific DICOM Tags

```python
# Function to search for tags by keyword
def search_dicom_tags(ds, keyword, case_sensitive=False):
    """Search for DICOM tags containing a keyword"""
    results = []
    
    if not case_sensitive:
        keyword = keyword.lower()
    
    for elem in ds:
        # Skip pixel data
        if elem.tag == (0x7FE0, 0x0010):
            continue
            
        # Check if keyword is in the tag name
        elem_name = elem.name if case_sensitive else elem.name.lower()
        if keyword in elem_name:
            results.append((elem.tag, elem.name, elem.repval))
    
    return results

# Search for tags containing "patient"
patient_tags = search_dicom_tags(ct_slice, "patient")
print("Tags related to patient:")
for tag, name, value in patient_tags:
    print(f"{tag} {name}: {value}")

# Search for tags containing "study"
study_tags = search_dicom_tags(ct_slice, "study")
print("\nTags related to study:")
for tag, name, value in study_tags:
    print(f"{tag} {name}: {value}")
```

## 5. Loading RT Structure Set

```python
# Find and load the RT Structure Set file
rs_files = [f for f in os.listdir(path) if f.startswith("RS")]
if rs_files:
    rs_path = os.path.join(path, rs_files[0])
    rs = pydicom.dcmread(rs_path)
    print(f"Loaded RT Structure Set: {rs_path}")
    print(f"Structure Set Label: {rs.StructureSetLabel}")
    print(f"Structure Set Date: {rs.StructureSetDate}")
    
    # List all structures
    roi_sequence = rs.StructureSetROISequence
    print(f"\nTotal structures: {len(roi_sequence)}")
    
    structures = []
    for i, roi in enumerate(roi_sequence):
        structures.append({
            'Number': roi.ROINumber,
            'Name': roi.ROIName,
        })
    
    # Display as a table
    df = pd.DataFrame(structures)
    print(df)
else:
    print("No RT Structure Set found")
```

## 6. Visualizing RT Structures on CT

```python
# Function to get contour data for a specific structure
def get_contour_data(rs, structure_name):
    """Extract contour data for a specific structure name"""
    # First, find the ROI number for the structure name
    roi_number = None
    for roi in rs.StructureSetROISequence:
        if roi.ROIName == structure_name:
            roi_number = roi.ROINumber
            break
    
    if roi_number is None:
        print(f"Structure {structure_name} not found")
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

# Get the Z coordinate of our current slice
current_z = ct_slice.SliceLocation

# Choose a structure to visualize (e.g., "PTV" or "BODY")
structure_to_visualize = "BODY"  # Change this to any structure name from the list above

# Get contour data
contour_data = get_contour_data(rs, structure_to_visualize)

if contour_data:
    # Display the CT slice with structure overlay
    plt.figure(figsize=(10, 10))
    plt.imshow(hu_image_windowed, cmap='gray')
    
    # Find contours on this slice
    contours_on_slice = [c for c in contour_data if np.isclose(c['z'], current_z, atol=0.1)]
    
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
        plt.gca().add_collection(collection)
    
    plt.title(f"CT Slice with {structure_to_visualize} contour")
    plt.axis('off')
    plt.show()
    
    print(f"Displaying {len(contours_on_slice)} contours for {structure_to_visualize}")
else:
    print(f"No contour data found for {structure_to_visualize}")
```

## 7. Loading RT Plan

```python
# Find and load the RT Plan file
rp_files = [f for f in os.listdir(path) if f.startswith("RP")]
if rp_files:
    rp_path = os.path.join(path, rp_files[0])
    rp = pydicom.dcmread(rp_path)
    print(f"Loaded RT Plan: {rp_path}")
    print(f"Plan Label: {rp.RTPlanLabel}")
    print(f"Plan Date: {rp.RTPlanDate}")
    
    # Display prescription information
    if hasattr(rp, 'DoseReferenceSequence'):
        print("\nPrescription Information:")
        for dose_ref in rp.DoseReferenceSequence:
            if hasattr(dose_ref, 'TargetPrescriptionDose'):
                print(f"Target Prescription Dose: {dose_ref.TargetPrescriptionDose} Gy")
    
    # Display beam information
    if hasattr(rp, 'BeamSequence'):
        beams = rp.BeamSequence
        print(f"\nTotal beams: {len(beams)}")
        
        beam_info = []
        for i, beam in enumerate(beams):
            beam_info.append({
                'Beam Number': beam.BeamNumber,
                'Beam Name': beam.BeamName,
                'Beam Type': beam.BeamType,
                'Radiation Type': beam.RadiationType
            })
        
        # Display as a table
        df = pd.DataFrame(beam_info)
        print(df)
else:
    print("No RT Plan found")
```

## 8. Loading RT Dose

```python
# Find and load the RT Dose file
rd_files = [f for f in os.listdir(path) if f.startswith("RD")]
if rd_files:
    rd_path = os.path.join(path, rd_files[0])
    rd = pydicom.dcmread(rd_path)
    print(f"Loaded RT Dose: {rd_path}")
    print(f"Dose Type: {rd.DoseType}")
    print(f"Dose Units: {rd.DoseUnits}")
    print(f"Dose Grid Scaling: {rd.DoseGridScaling}")
    
    # Get dose dimensions
    rows = rd.Rows
    cols = rd.Columns
    num_frames = rd.NumberOfFrames if hasattr(rd, 'NumberOfFrames') else 1
    
    print(f"\nDose grid size: {rows} x {cols} x {num_frames}")
    
    # Display middle slice of dose
    middle_frame = num_frames // 2
    
    # For 3D dose, need to extract the right frame
    if num_frames > 1:
        dose_array = rd.pixel_array[middle_frame] * rd.DoseGridScaling
    else:
        dose_array = rd.pixel_array * rd.DoseGridScaling
    
    plt.figure(figsize=(10, 10))
    plt.imshow(dose_array, cmap='jet')
    plt.colorbar(label='Dose (Gy)')
    plt.title(f"RT Dose - Slice {middle_frame+1}/{num_frames}")
    plt.axis('off')
    plt.show()
    
    print(f"Dose range: {np.min(dose_array):.2f} to {np.max(dose_array):.2f} Gy")
else:
    print("No RT Dose found")
```

## 9. Creating a Simple DVH (Dose Volume Histogram)

```python
# Function to calculate a DVH for a structure
def calculate_dvh(structure_name, rs, rd):
    """
    Calculate a simple DVH (Dose Volume Histogram) for a structure

    A DVH summarizes the 3D dose distribution in a structure into a 2D graph:
    - X-axis: Dose (Gy)
    - Y-axis: Volume percentage of the structure
    - The curve shows what percentage of the structure volume receives at least a certain dose

    The calculation process:
    1. Create a binary mask of the structure within the dose grid
    2. Extract dose values that fall within this mask
    3. Calculate the cumulative histogram of these dose values
    """
    # Get structure contours
    contour_data = get_contour_data(rs, structure_name)
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

# Calculate DVH for a structure (e.g., "PTV" or "BODY")
structure_for_dvh = "BODY"  # Change this to any structure name
structure_doses = calculate_dvh(structure_for_dvh, rs, rd)

if structure_doses is not None:
    # Create the DVH
    hist, bin_edges = np.histogram(structure_doses, bins=100, range=(0, np.max(structure_doses)))
    dvh = 1.0 - np.cumsum(hist) / float(len(structure_doses))

    # Plot the DVH
    plt.figure(figsize=(10, 6))
    plt.plot(bin_edges[1:], dvh * 100, 'b-', linewidth=2)
    plt.grid(True)
    plt.xlabel('Dose (Gy)')
    plt.ylabel('Volume (%)')
    plt.title(f'DVH for {structure_for_dvh}')
    plt.xlim([0, np.max(structure_doses)])
    plt.ylim([0, 100])
    plt.show()

    # Print some DVH statistics
    print(f"DVH Statistics for {structure_for_dvh}:")
    print(f"Min Dose: {np.min(structure_doses):.2f} Gy")
    print(f"Max Dose: {np.max(structure_doses):.2f} Gy")
    print(f"Mean Dose: {np.mean(structure_doses):.2f} Gy")
    print(f"Median Dose: {np.median(structure_doses):.2f} Gy")

    # D95 (dose to 95% of the volume)
    # This is the dose that 95% of the structure volume receives at least
    # Important for target coverage evaluation - higher is better for targets
    sorted_doses = np.sort(structure_doses)
    d95_index = int(len(sorted_doses) * 0.05)  # 95% from the top
    d95 = sorted_doses[d95_index]
    print(f"D95: {d95:.2f} Gy")

    # V20 (volume receiving 20 Gy or more)
    # This is the percentage of the structure volume receiving at least 20 Gy
    # Important for organ sparing - lower is better for healthy tissues
    v20 = np.sum(structure_doses >= 20.0) / len(structure_doses) * 100
    print(f"V20Gy: {v20:.2f}%")
else:
    print(f"Could not calculate DVH for {structure_for_dvh}")
```

## 10. Bonus: 3D Visualization of Structures

```python
# Note: This requires ipyvolume which might need to be installed
# !pip install ipyvolume

try:
    import ipyvolume as ipv
    
    # Choose a structure to visualize in 3D
    structure_name = "BODY"  # Change to any structure
    contour_data = get_contour_data(rs, structure_name)
    
    if contour_data:
        # Create 3D plot
        ipv.figure(width=600, height=600)
        
        # Plot each contour as a line
        for contour in contour_data:
            points = contour['triplets']
            x, y, z = points[:, 0], points[:, 1], points[:, 2]
            # Close the loop
            x = np.append(x, x[0])
            y = np.append(y, y[0])
            z = np.append(z, z[0])
            ipv.plot(x, y, z, color=contour['color'])
        
        ipv.style.box_off()
        ipv.style.axes_off()
        ipv.show()
        
        print(f"3D visualization of {structure_name} with {len(contour_data)} contours")
    else:
        print(f"No contour data found for {structure_name}")
except ImportError:
    print("3D visualization requires ipyvolume. Install with: pip install ipyvolume")
```

## Additional Resources

- [DICOM Standard](https://www.dicomstandard.org/)
- [pydicom Documentation](https://pydicom.github.io/)
- [RT DICOM Primer](https://www.aapm.org/pubs/reports/RPT_246.pdf)
- [SimpleITK DICOM Documentation](https://simpleitk.readthedocs.io/en/master/Documentation/docs/source/DicomImageIO.html)