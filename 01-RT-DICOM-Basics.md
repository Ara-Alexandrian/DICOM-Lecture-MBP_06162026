# Radiotherapy DICOM Files: The Basics

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

- **Image Position Patient (0020,0032)**: Defines the x,y,z coordinates of the first voxel (top-left) of the image
- **Image Orientation Patient (0020,0037)**: Defines the direction cosines of the first row and column
- **Pixel Spacing (0028,0030)**: Physical distance between centers of adjacent pixels
- **Slice Thickness (0018,0050)**: Thickness of each slice

For RT Dose:
- **Dose Grid Scaling (3004,000E)**: Factor to convert stored values to dose units
- **Grid Frame Offset Vector (3004,000C)**: Offsets of each dose plane in the stack

### Coordinate System Alignment

Proper alignment is crucial for:
1. Accurate dose calculation
2. Correct structure visualization
3. Meaningful DVH generation

Misalignment between grids can lead to:
- Incorrect dose coverage assessment
- Misleading DVH metrics
- Treatment errors

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

## Software Tools for RT DICOM

- **Python**: pydicom, SimpleITK, Plastimatch, dicompyler-core, dvhanalytics
- **Viewers**: 3D Slicer, MIM, Eclipse, RayStation, dicompyler
- **Open Source**: CERR, SlicerRT, DICOMautomaton