# Radiotherapy DICOM Files: The Basics
_lecture is posted at https://github.com/Ara-Alexandrian/DICOM-Lecture-MBP_06162026.git_


## What is DICOM?

DICOM (Digital Imaging and Communications in Medicine) is the international standard for medical images and related information. It defines formats for medical images and a communication protocol for exchanging them.

## RT DICOM Files

In radiotherapy, three special DICOM files work together to define treatment:

### 1. RT Structure Set (RTSTRUCT / RS)

- **SOP Class UID**: 1.2.840.10008.5.1.4.1.1.481.3
- Defines contours of anatomical structures (organs, targets)
- Contains ROI (Region of Interest) definitions
- Each structure has:
  - Name and number
  - Color for display
  - Type (PTV, OAR, etc.)
  - 3D coordinates defining the contour points

Example structure types:
- **PTVs**: Planning Target Volumes (what to treat)
- **OARs**: Organs At Risk (what to avoid)
- **External**: Body contour

### 2. RT Plan (RTPLAN / RP)

- **SOP Class UID**: 1.2.840.10008.5.1.4.1.1.481.5
- Contains treatment delivery instructions
- Defines:
  - Beam parameters (energy, gantry angles, collimator settings)
  - MLC (Multi-Leaf Collimator) positions
  - Prescription (total dose, fractionation)
  - Reference points
  - Monitor units

Key elements:
- **Fraction Groups**: How treatment is divided
- **Beams**: Individual radiation fields
- **Control Points**: MLC and machine settings over time

### 3. RT Dose (RTDOSE / RD)

- **SOP Class UID**: 1.2.840.10008.5.1.4.1.1.481.2
- 3D grid of calculated dose values
- Shows radiation distribution in the patient
- Used for:
  - Plan evaluation
  - DVH (Dose Volume Histogram) calculation
  - Quality assurance

Dose can be:
- **Absolute**: In physical units (Gy)
- **Relative**: Percentage of prescription
- **Per fraction** or **total treatment**

## Relationship Between RT Files

- Files are linked through UIDs (Unique Identifiers)
- RTSTRUCTs reference CT/MR images
- RTPLANs reference RTSTRUCTs
- RTDOSEs reference RTPLANs

This hierarchy ensures integrity of the treatment chain.

## Common DICOM Tags in RT Files

| Tag          | Name                    | Description                                |
|--------------|-------------------------|--------------------------------------------|
| (0008,0060)  | Modality                | RTSTRUCT, RTPLAN, RTDOSE                   |
| (0008,0016)  | SOP Class UID           | Identifies file type                       |
| (0008,0018)  | SOP Instance UID        | Unique identifier for this instance        |
| (0020,000D)  | Study Instance UID      | Links to same study                        |
| (0020,000E)  | Series Instance UID     | Links to same series                       |
| (300A,0002)  | RT Plan Label           | Name of the plan                           |
| (300A,0010)  | Dose Reference Sequence | Prescription information                   |
| (3006,0020)  | Structure Set ROI Seq   | List of all structures                     |
| (3006,0039)  | ROI Contour Sequence    | Actual contour data                        |

## Coordinate Systems and Registration

### Image Position and Orientation

One of the most critical aspects of RT DICOM is understanding how different grids (CT, dose, structures) align in 3D space:

- **Image Position Patient (0020,0032)**: Defines the x,y,z coordinates of the first voxel (top-left) of the image in the patient coordinate system (in mm)
- **Image Orientation Patient (0020,0037)**: Defines the direction cosines of the first row and column
- **Pixel Spacing (0028,0030)**: Physical distance between centers of adjacent pixels (in mm)
- **Slice Thickness (0018,0050)**: Thickness of each slice (in mm)
- **Slice Location (0020,1041)**: Position of the slice along the z-axis (in mm)

For RT Dose:
- **Dose Grid Scaling (3004,000E)**: Factor to convert stored values to dose units (Gy)
- **Grid Frame Offset Vector (3004,000C)**: Offsets of each dose plane in the stack (in mm)

### DICOM Patient Coordinate System

The DICOM patient coordinate system is defined as:
- **X-axis**: Increases from patient's right to left
- **Y-axis**: Increases from patient's anterior (front) to posterior (back)
- **Z-axis**: Increases from patient's inferior (feet) to superior (head)

This coordinate system is used for all RT DICOM components (images, structures, dose) to ensure proper alignment.

### Structure Contour Coordinates

Structure contours in RTSTRUCT files are stored as explicit 3D coordinates in the patient coordinate system:

- Each contour is a sequence of (x,y,z) triplets in mm
- Contours are grouped by z-coordinate (typically one contour per CT slice)
- Each structure can have multiple contours across different slices
- The contour sequence forms a 3D shape when combined

For example, a simplified contour sequence might look like:
```
Structure: "PTV"
  Contour at z=10mm: [(x1,y1,10), (x2,y2,10), (x3,y3,10), ...]
  Contour at z=15mm: [(x1,y1,15), (x2,y2,15), (x3,y3,15), ...]
  ...
```

### Coordinate Transformations for Visualization

To correctly overlay structures on images, we must transform structure coordinates to pixel indices:

1. **Patient to Pixel Coordinate Transformation**:
   - Subtract ImagePositionPatient to get position relative to the image origin
   - Divide by PixelSpacing to convert mm to pixel units
   - For CT images:
     ```
     pixel_x = (patient_x - ImagePositionPatient[0]) / PixelSpacing[1]
     pixel_y = (patient_y - ImagePositionPatient[1]) / PixelSpacing[0]
     ```

2. **Y-Axis Inversion**:
   - DICOM's y-axis runs from anterior to posterior
   - Image's y-axis typically runs from top to bottom
   - This requires inverting the y-axis:
     ```
     pixel_y = Rows - pixel_y
     ```
     where Rows is the height of the image in pixels

3. **Z-Slice Matching**:
   - Match structure contours to the correct image slice
   - Structure contour z-coordinate should be within tolerance of SliceLocation
   - Tolerance is typically half the SliceThickness

### Overlaying Dose Distributions

Dose grid alignment is particularly challenging because:

1. **Different Resolution**: Dose grids often have lower resolution than CT images
2. **Different Extents**: Dose grids may not cover the entire CT volume
3. **Different Orientation**: In rare cases, the dose grid may have a different orientation

To correctly overlay dose on CT images:

1. **Resample the Dose Grid**:
   - Determine the intersection of CT and dose volumes
   - Use tri-linear interpolation to sample dose values at CT voxel positions

2. **Transform Dose to Pixel Coordinates**:
   - Use ImagePositionPatient and PixelSpacing from both datasets
   - For each CT pixel, find the corresponding dose value

3. **Apply Color and Transparency**:
   - Map dose values to a color scale (e.g., jet, hot, plasma)
   - Apply transparency to show underlying anatomy

### Common Alignment Issues and Solutions

#### Structure Overlay Problems

**Problem**: Contours appear misaligned with the underlying image
**Solutions**:
- Verify ImagePositionPatient and PixelSpacing are correctly used
- Ensure y-axis inversion is properly implemented
- Check for SliceLocation matching with appropriate tolerance
- Verify that patient orientation is consistent between datasets

#### Dose Overlay Problems

**Problem**: Dose colorwash doesn't align with anatomy
**Solutions**:
- Check GridFrameOffsetVector interpretation
- Verify DoseGridScaling application
- Ensure the dose grid is correctly positioned in patient space
- Use consistent interpolation methods

### Interpolation Importance

When working with different resolution grids (CT, dose, structures), interpolation becomes critical:

1. **Resolution Differences**: CT resolution is typically higher than dose grid resolution
2. **Structure-to-Dose Mapping**: Converting contour points to dose grid voxels requires interpolation
3. **Dose Summation**: Adding multiple dose distributions requires careful interpolation

Interpolation methods matter:
- **Nearest Neighbor**: Fast but can create artifacts
- **Linear**: Better quality but can blur high gradients
- **Cubic**: Higher quality but computationally expensive

Accurate DVH calculation depends heavily on proper interpolation between structure contours and dose grid.

### Code Example: Structure Overlay

A simplified Python example for transforming structure coordinates to pixel coordinates:

```python
def transform_contour_to_pixel_coords(contour_points, image_position, pixel_spacing, rows):
    """
    Transform contour points from patient space to pixel space

    Args:
        contour_points: Nx3 array of (x,y,z) contour points in patient space (mm)
        image_position: ImagePositionPatient [x,y,z] of the image (mm)
        pixel_spacing: PixelSpacing [row_spacing, col_spacing] (mm)
        rows: Number of rows in the image

    Returns:
        Nx2 array of (x,y) contour points in pixel space
    """
    # Extract x and y coordinates
    x_points = [(p[0] - image_position[0]) / pixel_spacing[1] for p in contour_points]

    # Transform y coordinates with inversion (DICOM y-axis to image y-axis)
    y_points = [rows - (p[1] - image_position[1]) / pixel_spacing[0] for p in contour_points]

    # Combine into pixel coordinates
    pixel_coords = list(zip(x_points, y_points))

    return pixel_coords
```

## Software Tools for RT DICOM

- **Python**: pydicom, SimpleITK, Plastimatch
- **Viewers**: 3D Slicer, MIM, Eclipse, RayStation
- **Open Source**: CERR, SlicerRT, DICOMautomaton, dicompyler, dvhanalytics

## Glossary of RT DICOM Abbreviations

| Abbreviation | Full Term                                 | Description                                           |
|--------------|-------------------------------------------|-------------------------------------------------------|
| DICOM        | Digital Imaging and Communications in Medicine | Standard for storing and transmitting medical images  |
| RT           | Radiation Therapy                         | Medical use of ionizing radiation for treatment       |
| RTSTRUCT     | Radiation Therapy Structure Set           | DICOM file containing contoured anatomical structures |
| RTPLAN       | Radiation Therapy Plan                    | DICOM file containing treatment delivery parameters   |
| RTDOSE       | Radiation Therapy Dose                    | DICOM file containing calculated dose distributions   |
| RTIMAGE      | Radiation Therapy Image                   | DICOM file containing treatment verification images   |
| RTRECORD     | Radiation Therapy Treatment Record        | DICOM file documenting delivered treatment            |
| RS           | Structure Set (file prefix)               | Short form used for RTSTRUCT files                    |
| RP           | Plan (file prefix)                        | Short form used for RTPLAN files                      |
| RD           | Dose (file prefix)                        | Short form used for RTDOSE files                      |
| RI           | Image (file prefix)                       | Short form used for RTIMAGE files                     |
| SOP          | Service-Object Pair                       | DICOM term for a specific service applied to an object |
| UID          | Unique Identifier                         | Globally unique string identifying DICOM objects      |
| ROI          | Region of Interest                        | Contoured anatomical structure or target              |
| PTV          | Planning Target Volume                    | Volume to be treated, including margins               |
| CTV          | Clinical Target Volume                    | Tumor volume plus subclinical spread                  |
| GTV          | Gross Tumor Volume                        | Visible/palpable tumor volume                         |
| OAR          | Organ at Risk                             | Critical normal tissue that limits radiation dose     |
| DVH          | Dose Volume Histogram                     | Graph showing dose distribution in a structure        |
| MLC          | Multi-Leaf Collimator                     | Device that shapes radiation beams                    |
| HU           | Hounsfield Unit                           | CT image intensity values                             |
| TPS          | Treatment Planning System                 | Software used to create RT treatment plans            |
| IPP          | Image Position Patient                    | DICOM tag defining image origin position              |
| IOP          | Image Orientation Patient                 | DICOM tag defining image plane orientation            |