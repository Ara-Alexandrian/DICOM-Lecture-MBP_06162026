# DICOM-Lecture: Radiotherapy DICOM Workshop

A 30-minute workshop on understanding and working with radiotherapy DICOM files.

## Workshop Contents

This repository contains materials for a 30-minute radiotherapy DICOM workshop:

1. **[01-RT-DICOM-Basics.md](01-RT-DICOM-Basics.md)** - A 5-minute introduction to RT DICOM file types
2. **[02-DICOM-Interactive-Tutorial.md](02-DICOM-Interactive-Tutorial.md)** - Interactive Jupyter-style tutorial
3. **[03-DICOM-Viewer-Streamlit.md](03-DICOM-Viewer-Streamlit.md)** - Streamlit application walkthrough
4. **[dicom_viewer.py](dicom_viewer.py)** - Functional Streamlit DICOM viewer application
5. **[04-DICOM-Flask-Dashboard](04-DICOM-Flask-Dashboard/)** - Flask web application for DICOM visualization

> **Note:** The structure overlay functionality in the Flask Dashboard is currently experimental and may not render all structures correctly. If you need a fully working version for research or clinical purposes, please contact the author directly for assistance.

## Test Data

Example DICOM files are located in the `data/test-cases` directory:

- Patient 01149188
- Patient 20216124
- Patient 60897841

Each patient dataset includes CT images, RT Structure Set, RT Plan, and RT Dose files.

## Getting Started

### Requirements

Using conda with the provided rt-dcm environment:

```bash
conda activate rt-dcm
pip install -r requirements.txt
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

### Running the Jupyter Notebook

```bash
conda activate rt-dcm
jupyter notebook 02-DICOM-Interactive-Tutorial.ipynb
```

### Running the Streamlit App

```bash
conda activate rt-dcm
streamlit run dicom_viewer.py
```

### Running the Flask Dashboard

```bash
conda activate rt-dcm
cd 04-DICOM-Flask-Dashboard
python app.py
```

Then open your browser to http://localhost:5001

## Workshop Sequence

1. Start with a review of the RT DICOM basics (5 minutes)
2. Go through the interactive tutorial (15 minutes)
3. Demonstrate the Streamlit viewer (10 minutes)

## License

This project is provided for educational purposes.