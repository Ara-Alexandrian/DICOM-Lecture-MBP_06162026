# DICOM Flask Dashboard - Workshop Transcript

This transcript guides you through demonstrating the advanced Flask-based DICOM viewer application during your workshop.

## 1. Introduction

Start by introducing the dashboard as a more polished alternative to the Streamlit viewer:

> "Now that we've explored the basics of DICOM with Python and seen a simple Streamlit interface, let's look at what's possible with a more advanced web application. This Flask-based dashboard shows how you could build a professional-grade DICOM viewer with better performance and a more intuitive user interface."

## 2. Launch the Application

```bash
cd 04-DICOM-Flask-Dashboard
python app.py
```

Explain that this starts a local web server that hosts our application. Open your browser to http://localhost:5000.

> "The application is built with Flask on the backend, with a modern frontend using Bootstrap, JavaScript, and custom CSS. This architecture provides better performance and more flexibility than Streamlit."

## 3. Interface Overview

Walk through the main components of the interface:

> "The interface has a clean, dark theme design that's easier on the eyes when reviewing medical images. On the left side, we have our control panel with collapsible sections. The main area will display our DICOM images and data visualizations."

Point out these key elements:
- Left sidebar with patient selection and controls
- Main viewing area with tabs
- Responsive design that works on different screen sizes

## 4. Loading Patient Data

Select a patient from the dropdown menu:

> "Let's select a patient from our test cases. When we choose a patient, the application loads all associated DICOM data - CT images, structures, plan, and dose information - and prepares them for visualization."

Note how the UI updates to show:
- Patient information
- Slice navigation controls
- Window/level controls
- Structure and dose overlay options

## 5. Image Navigation and Manipulation

Demonstrate the various ways to navigate through slices:

> "There are multiple ways to navigate through CT slices. You can use the slider in the sidebar, the quick navigation slider below the image, the previous/next buttons, or even mouse wheel scrolling directly on the image."

Show the animation feature:

> "You can also automatically animate through slices using the play button, which is helpful for quickly reviewing an entire scan."

Demonstrate window/level adjustments:

> "Radiologists and radiation oncologists frequently need to adjust window and level settings to better visualize different tissues. Our application provides common presets like Soft Tissue, Lung, and Bone, plus custom adjustment sliders for fine control."

## 6. Structure Visualization

Show the structure overlay controls:

> "For radiotherapy planning, visualizing structures is essential. Here we can select any of the contoured structures to overlay on the CT image. You can select multiple structures simultaneously, toggle them all on or off, and see them with semi-transparent coloring."

Select some key structures like PTV, OARs, etc.

> "Notice how the structure colors match the colors defined in the original DICOM RT Structure Set, and how each structure is clearly labeled on the image itself."

## 7. Dose Visualization

Enable the dose overlay:

> "For evaluating treatment plans, visualizing the dose distribution is crucial. By enabling the dose overlay, we can see the planned radiation dose superimposed on the anatomy."

Demonstrate the customization options:

> "We can adjust the opacity of the dose overlay to better balance visibility of both the anatomy and the dose. We can also choose from different color maps that highlight different aspects of the dose distribution."

Point out the dose colorbar:

> "Notice the colorbar on the right side of the image showing the dose scale in Gray. This uses a global maximum across all slices, ensuring consistent coloring throughout the volume."

## 8. DVH Analysis

Switch to the DVH tab:

> "Dose Volume Histograms, or DVHs, are one of the most important tools for evaluating radiotherapy plans. They show what percentage of a structure receives different dose levels."

Select a structure to generate a DVH:

> "Let's select an organ at risk, like the spinal cord, to see its DVH. The application calculates this on the fly by sampling the dose grid within the structure contours."

Point out the DVH statistics:

> "Below the graph, we can see key statistics like minimum, maximum, and mean dose. We also see D95 (the dose covering 95% of the volume) and V20 (the percentage of volume receiving at least 20Gy), which are common metrics in clinical evaluation."

## 9. Plan Information

Switch to the Plan tab:

> "The Plan tab shows details from the RT Plan file, including information about the prescription dose and the treatment beams. This helps understand how the treatment will be delivered."

## 10. Advanced Features

Demonstrate some of the advanced visualization features:

> "The application also includes more advanced visualization capabilities like zooming, panning, and taking screenshots. These features make it more practical for clinical use."

Show zooming and panning:

> "You can zoom in on regions of interest using the zoom buttons or pinch gestures on touch devices. The pan tool lets you move around when zoomed in to focus on specific areas."

## 11. Mobile Compatibility

If possible, show the application on a tablet or simulate a mobile view:

> "The application is fully responsive and works well on tablets, which are increasingly common in clinical settings. All controls are touch-friendly, and we've implemented pinch-to-zoom gestures."

## 12. Comparison with Streamlit

Highlight some key advantages over the Streamlit version:

> "Compared to our earlier Streamlit application, this dashboard offers several advantages:
> - Better performance with less page reloading
> - More intuitive controls and layout
> - Advanced image manipulation like zoom and pan
> - Cleaner aesthetic with a clinical-friendly dark theme
> - Touch compatibility for tablets
> - Real-time DVH calculation"

## 13. Technical Discussion

Briefly discuss the architecture for technically-minded participants:

> "On the backend, we're using Flask to serve a RESTful API that processes DICOM data and generates visualizations. The frontend uses modern JavaScript to create a responsive single-page application that communicates with this API. This separation of concerns allows for better scalability and performance."

## 14. Conclusion

Wrap up the demonstration:

> "This demonstrates how Python's DICOM libraries can be integrated into professional-grade applications that approach the functionality of commercial systems. While this is still simpler than full clinical software, it shows the potential of open-source tools for medical imaging."

Encourage exploration:

> "I encourage you to explore the code in the repository to see how all these features are implemented. The modular design makes it straightforward to add new features or customize the application for specific clinical needs."