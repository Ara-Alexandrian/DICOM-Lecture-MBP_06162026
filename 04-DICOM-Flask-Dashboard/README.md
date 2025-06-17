# Advanced DICOM Viewer Dashboard

A feature-rich Flask-based DICOM viewer dashboard for radiotherapy data visualization.

## Features

- **Modern, responsive interface** with dark theme and intuitive controls
- **Interactive image viewing** with:
  - Window/level adjustment with presets and custom options
  - Structure overlay with customizable transparency
  - Dose distribution overlay with multiple colormap options
  - Image zoom and pan controls
  - Animation through slices
  - Screenshot capability
- **Comprehensive data display**:
  - Patient and study information
  - RT Plan details including beam parameters
  - DVH calculation and visualization
  - Structure statistics
- **Touch-friendly** interface for tablet use

## Requirements

- Python 3.7+
- Flask
- pydicom
- NumPy
- Matplotlib
- Pandas
- Pillow

## Installation

1. Navigate to the dashboard directory:
```bash
cd 04-DICOM-Flask-Dashboard
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Create an environment file:
```bash
cp .env.example .env
```

4. Edit the `.env` file to configure your environment:
```
FLASK_APP=app.py
FLASK_ENV=development  # Change to 'production' for deployment
FLASK_DEBUG=1          # Change to 0 for production
DATA_DIR=../data/test-cases
SECRET_KEY=your-secret-key-here
```

## Usage

### Development

1. Start the Flask development server:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Select a patient from the dropdown menu to begin.

### Production

For a production environment:

1. Set your environment variables in `.env`:
```
FLASK_APP=app.py
FLASK_ENV=production
FLASK_DEBUG=0
```

2. Run with waitress (included in requirements):
```bash
python app.py
```

Alternatively, you can use gunicorn:
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Directory Structure

```
04-DICOM-Flask-Dashboard/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md              # Documentation
├── static/                # Static assets
│   ├── css/               # Stylesheets
│   │   └── style.css      # Main CSS file
│   ├── js/                # JavaScript files
│   │   └── main.js        # Main JS file
│   └── img/               # Images and icons
└── templates/             # HTML templates
    └── index.html         # Main application template
```

## Data Structure

The application expects DICOM data in the following structure:

```
data/test-cases/
├── patient1/
│   ├── CT.*               # CT image files
│   ├── RS.*               # RT Structure Set files
│   ├── RP.*               # RT Plan files
│   └── RD.*               # RT Dose files
├── patient2/
│   └── ...
└── ...
```

## Future Enhancements

- 3D visualization of structures and dose
- Support for deformable image registration
- Machine learning-based auto-segmentation
- Plan comparison tools
- Report generation
- DICOM import/export functionality

## License

This project is provided for educational purposes.