// Global state
const state = {
    patientId: null,
    sliceIndex: 0,
    sliceCount: 0,
    windowCenter: 40,
    windowWidth: 400,
    windowPreset: 'Soft Tissue',
    selectedStructures: [],
    showDose: false,
    doseOpacity: 0.5,
    doseColormap: 'jet',
    imageScale: 1.0,
    panOffset: { x: 0, y: 0 },
    sliceAnimation: null,
    sliceCache: {}, // Cache for preloaded slice images
    loadedSlices: 0, // Counter for loaded slices
    preloadComplete: false, // Flag for preload completion
    lastRequestTime: 0, // Timestamp of the last slice request (for handling race conditions)
    usePreloading: true, // Flag to toggle preloading behavior
    structureToggleInProgress: false, // Flag to prevent scrolling during structure toggle
    scrollTimeout: null, // For debouncing wheel events
};

// DOM elements
const elements = {
    // Panels
    initialMessage: document.getElementById('initial-message'),
    mainViewer: document.getElementById('main-viewer'),
    patientInfoPanel: document.getElementById('patient-info-panel'),
    sliceNavPanel: document.getElementById('slice-nav-panel'),
    windowLevelPanel: document.getElementById('window-level-panel'),
    structurePanel: document.getElementById('structure-panel'),
    dosePanel: document.getElementById('dose-panel'),
    customWindowControls: document.getElementById('custom-window-controls'),
    doseControls: document.getElementById('dose-controls'),
    dvhContainer: document.getElementById('dvh-container'),
    dvhStats: document.getElementById('dvh-stats'),
    dvhAvailable: document.getElementById('dvh-available'),
    dvhUnavailable: document.getElementById('dvh-unavailable'),
    planAvailable: document.getElementById('plan-available'),
    planUnavailable: document.getElementById('plan-unavailable'),
    preloadContainer: document.getElementById('preload-container'),
    preloadProgress: document.getElementById('preload-progress'),
    preloadStatus: document.getElementById('preload-status'),

    // Selects and inputs
    patientSelect: document.getElementById('patient-select'),
    windowPreset: document.getElementById('window-preset'),
    toggleDose: document.getElementById('toggle-dose'),
    doseColormap: document.getElementById('dose-colormap'),
    toggleAllStructures: document.getElementById('toggle-all-structures'),
    togglePreload: document.getElementById('toggle-preload'),
    dvhStructure: document.getElementById('dvh-structure'),
    
    // Info displays
    patientInfoTable: document.getElementById('patient-info-table'),
    infoPatientTable: document.getElementById('info-patient-table'),
    infoImageTable: document.getElementById('info-image-table'),
    planInfoTable: document.getElementById('plan-info-table'),
    planBeamsTable: document.getElementById('plan-beams-table'),
    sliceInfo: document.getElementById('slice-info'),
    sliceMin: document.getElementById('slice-min'),
    sliceCurrent: document.getElementById('slice-current'),
    sliceMax: document.getElementById('slice-max'),
    windowCenterValue: document.getElementById('window-center-value'),
    windowWidthValue: document.getElementById('window-width-value'),
    doseOpacityValue: document.getElementById('dose-opacity-value'),
    dvhMin: document.getElementById('dvh-min'),
    dvhMax: document.getElementById('dvh-max'),
    dvhMean: document.getElementById('dvh-mean'),
    dvhMedian: document.getElementById('dvh-median'),
    dvhD95: document.getElementById('dvh-d95'),
    dvhV20: document.getElementById('dvh-v20'),
    
    // Images and containers
    dicomImage: document.getElementById('dicom-image'),
    dvhImage: document.getElementById('dvh-image'),
    imageContainer: document.getElementById('image-container'),
    structureList: document.getElementById('structure-list'),
    imageLoading: document.getElementById('image-loading'),
    onDemandIndicator: document.getElementById('on-demand-indicator'),
    dvhLoading: document.getElementById('dvh-loading'),

    // 3D elements
    renderBtn: document.getElementById('render-3d'),
    threedContainer: document.getElementById('3d-container'),
    threedImage: document.getElementById('3d-image'),
    threedLoading: document.getElementById('3d-loading'),
    threedInfo: document.getElementById('3d-info'),
    threedStructuresList: document.getElementById('3d-structures-list'),
    threedAvailable: document.getElementById('3d-available'),
    threedUnavailable: document.getElementById('3d-unavailable'),
    
    // Breadcrumbs
    patientBreadcrumb: document.getElementById('patient-breadcrumb'),
    sliceBreadcrumb: document.getElementById('slice-breadcrumb'),
    
    // Buttons
    prevSlice: document.getElementById('prev-slice'),
    nextSlice: document.getElementById('next-slice'),
    playSlices: document.getElementById('play-slices'),
    zoomIn: document.getElementById('zoom-in'),
    zoomOut: document.getElementById('zoom-out'),
    panTool: document.getElementById('pan-tool'),
    resetView: document.getElementById('reset-view'),
    screenshot: document.getElementById('screenshot')
};

// Slider objects
let sliceSlider, windowCenterSlider, windowWidthSlider, doseOpacitySlider;

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    initializeSliders();
    initializeEventListeners();
});

// Initialize range sliders using noUiSlider
function initializeSliders() {
    // Slice slider
    sliceSlider = document.getElementById('slice-slider');
    noUiSlider.create(sliceSlider, {
        start: [0],
        connect: [true, false],
        step: 1,
        range: {
            'min': [0],
            'max': [100]
        }
    });
    
    // Window center slider
    windowCenterSlider = document.getElementById('window-center-slider');
    noUiSlider.create(windowCenterSlider, {
        start: [40],
        connect: [true, false],
        step: 10,
        range: {
            'min': [-1000],
            'max': [3000]
        }
    });
    
    // Window width slider
    windowWidthSlider = document.getElementById('window-width-slider');
    noUiSlider.create(windowWidthSlider, {
        start: [400],
        connect: [true, false],
        step: 10,
        range: {
            'min': [1],
            'max': [4000]
        }
    });
    
    // Dose opacity slider
    doseOpacitySlider = document.getElementById('dose-opacity-slider');
    noUiSlider.create(doseOpacitySlider, {
        start: [0.5],
        connect: [true, false],
        step: 0.05,
        range: {
            'min': [0.1],
            'max': [1]
        }
    });
    
    // Update state when sliders change
    sliceSlider.noUiSlider.on('update', (values) => {
        const index = parseInt(values[0]);
        if (index !== state.sliceIndex) {
            state.sliceIndex = index;
            elements.sliceCurrent.textContent = index;
            updateSliceView();
        }
    });
    
    windowCenterSlider.noUiSlider.on('update', (values) => {
        state.windowCenter = parseInt(values[0]);
        elements.windowCenterValue.textContent = state.windowCenter;
        updateSliceView();
    });
    
    windowWidthSlider.noUiSlider.on('update', (values) => {
        state.windowWidth = parseInt(values[0]);
        elements.windowWidthValue.textContent = state.windowWidth;
        updateSliceView();
    });
    
    doseOpacitySlider.noUiSlider.on('update', (values) => {
        state.doseOpacity = parseFloat(values[0]);
        elements.doseOpacityValue.textContent = state.doseOpacity.toFixed(2);
        updateSliceView();
    });
}

// Set up event listeners
function initializeEventListeners() {
    // Patient selection
    elements.patientSelect.addEventListener('change', handlePatientChange);

    // Window/Level preset
    elements.windowPreset.addEventListener('change', handleWindowPresetChange);

    // Dose controls
    elements.toggleDose.addEventListener('change', handleToggleDose);
    elements.doseColormap.addEventListener('change', handleDoseColormapChange);

    // Structure toggle all
    elements.toggleAllStructures.addEventListener('change', handleToggleAllStructures);

    // Preload toggle
    elements.togglePreload.addEventListener('change', handleTogglePreload);

    // DVH structure selection
    elements.dvhStructure.addEventListener('change', handleDvhStructureChange);

    // 3D rendering button
    document.getElementById('render-3d').addEventListener('click', handle3DRender);

    // Slice navigation buttons
    elements.prevSlice.addEventListener('click', handlePrevSlice);
    elements.nextSlice.addEventListener('click', handleNextSlice);
    elements.playSlices.addEventListener('click', handlePlaySlices);

    // Image zoom and pan controls
    elements.zoomIn.addEventListener('click', handleZoomIn);
    elements.zoomOut.addEventListener('click', handleZoomOut);
    elements.panTool.addEventListener('click', handlePanTool);
    elements.resetView.addEventListener('click', handleResetView);
    elements.screenshot.addEventListener('click', handleScreenshot);

    // Mouse wheel for slice navigation
    elements.imageContainer.addEventListener('wheel', handleMouseWheel);

    // Touch events for mobile
    setupTouchEvents();
}

// Handle patient selection change
async function handlePatientChange() {
    const patientId = elements.patientSelect.value;
    if (!patientId) return;

    // Reset state
    resetState();
    state.patientId = patientId;

    // Show loading
    showLoading(true);

    try {
        // Fetch patient data
        const patientData = await fetchPatientData(patientId);

        // Update UI with patient data
        updatePatientUI(patientData);

        // Show main viewer
        elements.initialMessage.classList.add('d-none');
        elements.mainViewer.classList.remove('d-none');

        // Set breadcrumb
        elements.patientBreadcrumb.textContent = `Patient: ${patientId}`;
        elements.patientBreadcrumb.classList.remove('d-none');
        elements.sliceBreadcrumb.classList.remove('d-none');

        // Start preloading slices if enabled
        if (state.usePreloading) {
            await preloadSlices();
        } else {
            console.log('Preloading disabled. Slices will be loaded on-demand.');
            // Just load the current slice
            state.preloadComplete = false;
        }

        // Update current slice view
        updateSliceView();
    } catch (error) {
        console.error('Error loading patient data:', error);
        alert('Error loading patient data. Please try again.');
    } finally {
        showLoading(false);
    }
}

// Handle preload toggle
function handleTogglePreload() {
    state.usePreloading = elements.togglePreload.checked;

    // Clear the cache when toggling to ensure consistent behavior
    state.sliceCache = {};
    state.preloadComplete = false;

    if (state.usePreloading && state.patientId) {
        // Start preloading if enabled and we have a patient
        console.log('Preloading enabled. Starting preload...');
        preloadSlices().then(() => {
            // When preloading is complete, update the current view
            updateSliceView();
        });
    } else {
        console.log('Preloading disabled. Slices will be loaded on-demand.');
    }
}

// Reset application state
function resetState() {
    state.sliceIndex = 0;
    state.sliceCount = 0;
    state.windowCenter = 40;
    state.windowWidth = 400;
    state.windowPreset = 'Soft Tissue';
    state.selectedStructures = [];
    state.showDose = false;
    state.doseOpacity = 0.5;
    state.doseColormap = 'jet';
    state.imageScale = 1.0;
    state.panOffset = { x: 0, y: 0 };

    if (state.sliceAnimation) {
        clearInterval(state.sliceAnimation);
        state.sliceAnimation = null;
    }

    // Keep the preloading preference but clear the cache
    state.sliceCache = {};
    state.loadedSlices = 0;
    state.preloadComplete = false;

    // Reset UI elements
    elements.windowPreset.value = 'Soft Tissue';
    elements.toggleDose.checked = false;
    elements.doseColormap.value = 'jet';
    elements.toggleAllStructures.checked = false;
    elements.dvhStructure.value = '';
    elements.dvhContainer.classList.add('d-none');
    elements.dvhStats.classList.add('d-none');

    // Don't reset the preload toggle - keep user preference
    // elements.togglePreload.checked = state.usePreloading;

    windowCenterSlider.noUiSlider.set(40);
    windowWidthSlider.noUiSlider.set(400);
    doseOpacitySlider.noUiSlider.set(0.5);

    elements.customWindowControls.classList.add('d-none');
    elements.doseControls.classList.add('d-none');

    // Reset image transformations
    elements.dicomImage.style.transform = '';
}

// Fetch patient data from the server
async function fetchPatientData(patientId) {
    const response = await fetch(`/api/patient/${patientId}`);
    if (!response.ok) {
        throw new Error(`Failed to fetch patient data: ${response.statusText}`);
    }
    return response.json();
}

// Update UI with patient data
function updatePatientUI(patientData) {
    // Update slice slider
    const sliceRange = patientData.slice_range;
    state.sliceCount = sliceRange.count;
    sliceSlider.noUiSlider.updateOptions({
        range: {
            'min': sliceRange.min,
            'max': sliceRange.max
        }
    });
    sliceSlider.noUiSlider.set(Math.floor(sliceRange.max / 2));
    elements.sliceMin.textContent = sliceRange.min;
    elements.sliceMax.textContent = sliceRange.max;
    
    // Show patient information
    updatePatientInfoTable(patientData.patient_info);
    
    // Show structures if available
    if (patientData.structures && patientData.structures.length > 0) {
        populateStructureList(patientData.structures);
        elements.structurePanel.classList.remove('d-none');
    } else {
        elements.structurePanel.classList.add('d-none');
    }
    
    // Show dose controls if available
    if (patientData.has_dose) {
        elements.dosePanel.classList.remove('d-none');
    } else {
        elements.dosePanel.classList.add('d-none');
    }
    
    // Show/hide plan info
    if (Object.keys(patientData.plan_info).length > 0) {
        updatePlanInfo(patientData.plan_info);
        elements.planAvailable.classList.remove('d-none');
        elements.planUnavailable.classList.add('d-none');
    } else {
        elements.planAvailable.classList.add('d-none');
        elements.planUnavailable.classList.remove('d-none');
    }
    
    // Show/hide DVH panel
    if (patientData.has_dose && patientData.structures.length > 0) {
        populateDvhStructureSelect(patientData.structures);
        elements.dvhAvailable.classList.remove('d-none');
        elements.dvhUnavailable.classList.add('d-none');
    } else {
        elements.dvhAvailable.classList.add('d-none');
        elements.dvhUnavailable.classList.remove('d-none');
    }

    // Update 3D panel
    update3DPanel();
    
    // Show control panels
    elements.patientInfoPanel.classList.remove('d-none');
    elements.sliceNavPanel.classList.remove('d-none');
    elements.windowLevelPanel.classList.remove('d-none');
}

// Update patient information tables
function updatePatientInfoTable(patientInfo) {
    // Sidebar info table
    let sidebarHtml = '';
    for (const [key, value] of Object.entries(patientInfo)) {
        const formattedKey = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
        sidebarHtml += `<tr><td>${formattedKey}</td><td>${value}</td></tr>`;
    }
    elements.patientInfoTable.innerHTML = sidebarHtml;
    
    // Main info panel
    let mainInfoHtml = '';
    for (const [key, value] of Object.entries(patientInfo)) {
        if (key !== 'slice_thickness' && key !== 'num_slices') {
            const formattedKey = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            mainInfoHtml += `<tr><th>${formattedKey}</th><td>${value}</td></tr>`;
        }
    }
    elements.infoPatientTable.innerHTML = mainInfoHtml;
    
    // Image info
    const imageInfoHtml = `
        <tr><th>Number of Slices</th><td>${patientInfo.num_slices}</td></tr>
        <tr><th>Slice Thickness</th><td>${patientInfo.slice_thickness}</td></tr>
        <tr><th>Current Slice</th><td id="current-slice-info">-</td></tr>
        <tr><th>Window/Level</th><td id="current-window-info">${state.windowPreset} (${state.windowWidth}/${state.windowCenter})</td></tr>
    `;
    elements.infoImageTable.innerHTML = imageInfoHtml;
}

// Update plan information
function updatePlanInfo(planInfo) {
    // Basic plan info
    let planInfoHtml = '';
    if (planInfo.plan_label) {
        planInfoHtml += `<tr><th>Plan Label</th><td>${planInfo.plan_label}</td></tr>`;
    }
    if (planInfo.plan_date) {
        planInfoHtml += `<tr><th>Plan Date</th><td>${planInfo.plan_date}</td></tr>`;
    }
    if (planInfo.prescription_dose) {
        planInfoHtml += `<tr><th>Prescription Dose</th><td>${planInfo.prescription_dose} Gy</td></tr>`;
    }
    elements.planInfoTable.innerHTML = planInfoHtml;
    
    // Beam info
    let beamsHtml = '';
    if (planInfo.beams && planInfo.beams.length > 0) {
        for (const beam of planInfo.beams) {
            beamsHtml += `<tr>
                <td>${beam.beam_name}</td>
                <td>${beam.beam_type}</td>
                <td>${beam.radiation_type}</td>
            </tr>`;
        }
    }
    elements.planBeamsTable.innerHTML = beamsHtml;
}

// Populate structure list
function populateStructureList(structures) {
    let html = '';
    
    for (const structure of structures) {
        html += `
            <div class="structure-item">
                <input class="form-check-input structure-checkbox" type="checkbox" 
                       id="structure-${structure.number}" data-name="${structure.name}">
                <span class="structure-color" style="background-color: ${structure.color}"></span>
                <label class="form-check-label" for="structure-${structure.number}">
                    ${structure.name}
                </label>
            </div>
        `;
    }
    
    elements.structureList.innerHTML = html;
    
    // Add event listeners to checkboxes
    document.querySelectorAll('.structure-checkbox').forEach(checkbox => {
        checkbox.addEventListener('change', handleStructureToggle);
    });
}

// Populate DVH structure select
function populateDvhStructureSelect(structures) {
    let html = '<option value="">Select a structure</option>';
    
    for (const structure of structures) {
        html += `<option value="${structure.name}">${structure.name}</option>`;
    }
    
    elements.dvhStructure.innerHTML = html;
}

// Preload all slices to cache with optimized loading order
async function preloadSlices() {
    if (!state.patientId || state.sliceCount === 0) return;

    // Show preload UI
    elements.preloadContainer.classList.remove('d-none');
    state.preloadComplete = false;
    state.loadedSlices = 0;
    state.sliceCache = {};

    // Determine current slice index (center of volume or user selected)
    const currentSliceIdx = state.sliceIndex;

    // Generate a priority-ordered list of slices to load
    // This loads slices in "outward waves" from the current position
    // to ensure a good user experience if they start navigating before preload is complete
    const sliceLoadOrder = generatePriorityLoadOrder(currentSliceIdx, state.sliceCount);

    // Calculate batch size - process 5 slices at a time for better performance
    const batchSize = 5;
    const totalBatches = Math.ceil(sliceLoadOrder.length / batchSize);

    try {
        for (let batch = 0; batch < totalBatches; batch++) {
            const batchPromises = [];

            // Process a batch of slices
            for (let i = 0; i < batchSize; i++) {
                const batchIdx = batch * batchSize + i;
                if (batchIdx >= sliceLoadOrder.length) break;

                const sliceIdx = sliceLoadOrder[batchIdx];
                batchPromises.push(loadSlice(sliceIdx));
            }

            // Wait for all slices in the batch to load
            await Promise.all(batchPromises);

            // Update progress
            const progress = Math.min(100, Math.round((state.loadedSlices / state.sliceCount) * 100));
            elements.preloadProgress.style.width = `${progress}%`;
            elements.preloadStatus.textContent = `${progress}% Complete (${state.loadedSlices}/${state.sliceCount})`;

            // Allow the UI to update by yielding to the event loop occasionally
            if (batch % 4 === 0) {
                await new Promise(resolve => setTimeout(resolve, 0));
            }
        }

        state.preloadComplete = true;
        console.log('Preloading complete!');

        // Fade out the preload UI with a smooth transition
        elements.preloadContainer.style.transition = 'opacity 0.5s ease-out';
        elements.preloadContainer.style.opacity = '0';

        // Hide after transition completes
        setTimeout(() => {
            elements.preloadContainer.classList.add('d-none');
            elements.preloadContainer.style.opacity = '1';
        }, 500);
    } catch (error) {
        console.error('Error preloading slices:', error);
        alert('Error preloading slices. Some images may load more slowly.');
        elements.preloadContainer.classList.add('d-none');
    }
}

// Generate a priority-ordered array of slice indices
// This creates a sequence that loads outward from the current slice
// to provide the best user experience during navigation
function generatePriorityLoadOrder(currentIdx, totalCount) {
    const result = [currentIdx]; // Start with the current slice
    let distanceFromCurrent = 1;

    // Build the sequence by alternating between next and previous slices
    // until we've covered all slices in the volume
    while (result.length < totalCount) {
        // Add the next slice (if it exists)
        const nextIdx = currentIdx + distanceFromCurrent;
        if (nextIdx < totalCount) {
            result.push(nextIdx);
        }

        // Add the previous slice (if it exists)
        const prevIdx = currentIdx - distanceFromCurrent;
        if (prevIdx >= 0) {
            result.push(prevIdx);
        }

        // Increase distance for next iteration
        distanceFromCurrent++;
    }

    return result;
}

// Load a single slice and cache it
async function loadSlice(sliceIdx) {
    const cacheKey = getCacheKey(sliceIdx);

    // Skip if already cached
    if (state.sliceCache[cacheKey]) {
        return state.sliceCache[cacheKey];
    }

    try {
        const response = await fetch('/api/slice', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                patient_id: state.patientId,
                slice_idx: sliceIdx,
                window_center: state.windowCenter,
                window_width: state.windowWidth,
                structures: state.selectedStructures,
                show_dose: state.showDose,
                dose_opacity: state.doseOpacity,
                dose_colormap: state.doseColormap
            })
        });

        if (!response.ok) {
            throw new Error(`Failed to fetch slice: ${response.statusText}`);
        }

        const data = await response.json();

        // Cache the slice data
        state.sliceCache[cacheKey] = data;
        state.loadedSlices++;

        return data;
    } catch (error) {
        console.error(`Error loading slice ${sliceIdx}:`, error);
        throw error;
    }
}

// Generate a cache key for the current view settings
function getCacheKey(sliceIdx) {
    return `${state.patientId}_${sliceIdx}_${state.windowCenter}_${state.windowWidth}_${state.selectedStructures.join('_')}_${state.showDose}_${state.doseOpacity}_${state.doseColormap}`;
}

// Update the slice view from cache or fetch
async function updateSliceView() {
    if (!state.patientId) return;

    const currentSliceIdx = state.sliceIndex;
    const cacheKey = getCacheKey(currentSliceIdx);

    // We'll only show the loading indicator if the slice isn't cached
    const showLoadingIndicator = !state.sliceCache[cacheKey];

    if (showLoadingIndicator) {
        showLoading(true);
    }

    // Keep track of the request time to handle race conditions
    const requestTime = Date.now();
    state.lastRequestTime = requestTime;

    try {
        let data;

        // Check if we have this slice in cache
        if (state.sliceCache[cacheKey]) {
            data = state.sliceCache[cacheKey];
            console.log(`Using cached slice ${currentSliceIdx}`);
        } else {
            // If not in cache, load it
            console.log(`Fetching slice ${currentSliceIdx} (not in cache)`);

            // If not using preloading, add a small delay to simulate network latency
            // This makes the performance difference more noticeable for the demo
            if (!state.usePreloading) {
                // Show a message in the console to indicate the artificial delay
                console.log('Adding artificial delay to highlight on-demand loading performance');
                await new Promise(resolve => setTimeout(resolve, 300));
            }

            data = await loadSlice(currentSliceIdx);
        }

        // Only update the UI if this is still the most recent request
        // This prevents older requests from overriding newer ones
        if (requestTime !== state.lastRequestTime) {
            console.log(`Skipping update for slice ${currentSliceIdx} (newer request exists)`);
            return;
        }

        // Only update the UI if this is still the current slice
        if (currentSliceIdx === state.sliceIndex) {
            // Update image with an enhanced smooth fade effect
            elements.dicomImage.style.transition = 'opacity 0.15s ease-in-out';
            elements.dicomImage.style.opacity = 0;

            // Create a new Image object to preload the next image
            const newImage = new Image();

            // When the image is loaded, update the display
            newImage.onload = () => {
                // Apply the new image
                elements.dicomImage.src = newImage.src;

                // Use requestAnimationFrame for smoother transition
                requestAnimationFrame(() => {
                    elements.dicomImage.style.opacity = 1;

                    // Update all slice info
                    updateSliceInfo(currentSliceIdx, data);
                });
            };

            // Set the source to trigger loading
            newImage.src = `data:image/png;base64,${data.image}`;

            // If the image is already cached in browser, the onload might not fire
            // In that case, update immediately
            if (newImage.complete) {
                elements.dicomImage.src = newImage.src;
                requestAnimationFrame(() => {
                    elements.dicomImage.style.opacity = 1;
                    updateSliceInfo(currentSliceIdx, data);
                });
            }
        }
    } catch (error) {
        console.error('Error updating slice view:', error);
        // Only show alert if this is still the current slice and most recent request
        if (currentSliceIdx === state.sliceIndex && requestTime === state.lastRequestTime) {
            alert('Error updating slice view. Please try again.');
        }
    } finally {
        if (showLoadingIndicator) {
            showLoading(false);
        }
    }

    // Preload adjacent slices for smooth navigation
    // Only if preloading is enabled and preloading process is complete
    if (state.usePreloading && state.preloadComplete) {
        preloadAdjacentSlices(currentSliceIdx);
    }
}

// Helper function to update all slice info elements
function updateSliceInfo(sliceIdx, data) {
    // Update slice info
    elements.sliceInfo.textContent = `CT Slice ${sliceIdx + 1}/${state.sliceCount} at ${data.slice_info.position.toFixed(2)}mm`;
    elements.sliceBreadcrumb.textContent = `Slice ${sliceIdx + 1}/${state.sliceCount}`;

    // Update current slice info in the info panel
    const currentSliceInfo = document.getElementById('current-slice-info');
    if (currentSliceInfo) {
        currentSliceInfo.textContent = `${sliceIdx + 1}/${state.sliceCount} (${data.slice_info.position.toFixed(2)}mm)`;
    }

    // Update window info in the info panel
    const currentWindowInfo = document.getElementById('current-window-info');
    if (currentWindowInfo) {
        currentWindowInfo.textContent = `${state.windowPreset} (${state.windowWidth}/${state.windowCenter})`;
    }
}

// Preload adjacent slices for smoother navigation
async function preloadAdjacentSlices(currentIdx) {
    const preloadRange = 3; // Preload 3 slices in each direction

    // Create an array of indices to preload, prioritizing the next slices
    const indicesToPreload = [];

    // Next slices (higher priority)
    for (let i = 1; i <= preloadRange; i++) {
        if (currentIdx + i < state.sliceCount) {
            indicesToPreload.push(currentIdx + i);
        }
    }

    // Previous slices (lower priority)
    for (let i = 1; i <= preloadRange; i++) {
        if (currentIdx - i >= 0) {
            indicesToPreload.push(currentIdx - i);
        }
    }

    // Preload in background
    for (const idx of indicesToPreload) {
        const cacheKey = getCacheKey(idx);
        if (!state.sliceCache[cacheKey]) {
            loadSlice(idx).catch(err => {
                // Silently ignore errors in background preloading
                console.warn(`Background preload failed for slice ${idx}:`, err);
            });
        }
    }
}

// Handle window preset change
function handleWindowPresetChange() {
    const preset = elements.windowPreset.value;
    state.windowPreset = preset;

    if (preset === 'Custom') {
        elements.customWindowControls.classList.remove('d-none');
    } else {
        elements.customWindowControls.classList.add('d-none');

        // Set preset values
        const presets = {
            'Soft Tissue': [40, 400],
            'Lung': [-600, 1500],
            'Bone': [500, 2000],
            'Brain': [40, 80]
        };

        const [center, width] = presets[preset];
        state.windowCenter = center;
        state.windowWidth = width;

        // Update sliders
        windowCenterSlider.noUiSlider.set(center);
        windowWidthSlider.noUiSlider.set(width);

        // Clear the cache when window/level changes
        state.sliceCache = {};

        // If we've already preloaded, start preloading again in the background
        if (state.preloadComplete) {
            state.preloadComplete = false;
            preloadSlices().then(() => {
                // When preloading is done, update the current view
                updateSliceView();
            });
        } else {
            updateSliceView();
        }
    }
}

// Handle dose toggle
function handleToggleDose() {
    state.showDose = elements.toggleDose.checked;

    if (state.showDose) {
        elements.doseControls.classList.remove('d-none');
    } else {
        elements.doseControls.classList.add('d-none');
    }

    // Clear the cache when dose display changes
    state.sliceCache = {};

    // If we've already preloaded, start preloading again in the background
    if (state.preloadComplete) {
        state.preloadComplete = false;
        preloadSlices().then(() => {
            // When preloading is done, update the current view
            updateSliceView();
        });
    } else {
        updateSliceView();
    }
}

// Handle dose colormap change
function handleDoseColormapChange() {
    state.doseColormap = elements.doseColormap.value;

    // Clear the cache when dose colormap changes
    state.sliceCache = {};

    // If we've already preloaded, start preloading again in the background
    if (state.preloadComplete) {
        state.preloadComplete = false;
        preloadSlices().then(() => {
            // When preloading is done, update the current view
            updateSliceView();
        });
    } else {
        updateSliceView();
    }
}

// Handle structure toggle
function handleStructureToggle(event) {
    // Set the structureToggleInProgress flag to prevent scrolling
    state.structureToggleInProgress = true;

    const structureName = event.target.dataset.name;

    if (event.target.checked) {
        if (!state.selectedStructures.includes(structureName)) {
            state.selectedStructures.push(structureName);
        }
    } else {
        state.selectedStructures = state.selectedStructures.filter(name => name !== structureName);
    }

    // Clear the cache when structures change
    state.sliceCache = {};

    // Force preloading to stop by setting flag to false
    const wasPreloading = state.preloadComplete;
    state.preloadComplete = false;

    // Update the current slice immediately to show changes
    showLoading(true);

    // Immediately load the current slice to show the structure change
    loadSlice(state.sliceIndex).then(data => {
        // Update the display with the new slice that includes structure changes
        const currentSliceIdx = state.sliceIndex;
        elements.dicomImage.style.transition = 'opacity 0.15s ease-in-out';
        elements.dicomImage.style.opacity = 0;

        // Use a timeout to ensure the fade out is visible
        setTimeout(() => {
            elements.dicomImage.src = `data:image/png;base64,${data.image}`;
            elements.dicomImage.style.opacity = 1;
            updateSliceInfo(currentSliceIdx, data);

            // Hide loading indicator
            showLoading(false);

            // Re-enable scrolling after the current slice is updated
            setTimeout(() => {
                // Reset the structure toggle flag to allow scrolling again
                state.structureToggleInProgress = false;

                // If we were preloading before, restart preloading in the background
                if (wasPreloading && state.usePreloading) {
                    preloadSlices();
                }
            }, 200); // Short delay to prevent immediate scrolling
        }, 50);
    }).catch(error => {
        console.error('Error updating slice after structure toggle:', error);
        showLoading(false);
        // Make sure to reset the flag even on error
        state.structureToggleInProgress = false;
    });
}

// Handle toggle all structures
function handleToggleAllStructures() {
    // Set the structureToggleInProgress flag to prevent scrolling
    state.structureToggleInProgress = true;

    const checkAll = elements.toggleAllStructures.checked;

    // Update all checkboxes
    document.querySelectorAll('.structure-checkbox').forEach(checkbox => {
        checkbox.checked = checkAll;

        // Update selected structures array
        const structureName = checkbox.dataset.name;
        if (checkAll) {
            if (!state.selectedStructures.includes(structureName)) {
                state.selectedStructures.push(structureName);
            }
        } else {
            state.selectedStructures = [];
        }
    });

    // Clear the cache when structures change
    state.sliceCache = {};

    // Force preloading to stop by setting flag to false
    const wasPreloading = state.preloadComplete;
    state.preloadComplete = false;

    // Update the current slice immediately to show changes
    showLoading(true);

    // Immediately load the current slice to show the structure change
    loadSlice(state.sliceIndex).then(data => {
        // Update the display with the new slice that includes structure changes
        const currentSliceIdx = state.sliceIndex;
        elements.dicomImage.style.transition = 'opacity 0.15s ease-in-out';
        elements.dicomImage.style.opacity = 0;

        // Use a timeout to ensure the fade out is visible
        setTimeout(() => {
            elements.dicomImage.src = `data:image/png;base64,${data.image}`;
            elements.dicomImage.style.opacity = 1;
            updateSliceInfo(currentSliceIdx, data);

            // Hide loading indicator
            showLoading(false);

            // Re-enable scrolling after the current slice is updated
            setTimeout(() => {
                // Reset the structure toggle flag to allow scrolling again
                state.structureToggleInProgress = false;

                // If we were preloading before, restart preloading in the background
                if (wasPreloading && state.usePreloading) {
                    preloadSlices();
                }
            }, 200); // Short delay to prevent immediate scrolling
        }, 50);
    }).catch(error => {
        console.error('Error updating slice after toggling all structures:', error);
        showLoading(false);
        // Make sure to reset the flag even on error
        state.structureToggleInProgress = false;
    });
}

// Handle DVH structure change
async function handleDvhStructureChange() {
    const structureName = elements.dvhStructure.value;
    if (!structureName) {
        elements.dvhContainer.classList.add('d-none');
        elements.dvhStats.classList.add('d-none');
        return;
    }
    
    elements.dvhContainer.classList.remove('d-none');
    elements.dvhLoading.classList.remove('d-none');
    
    try {
        const response = await fetch('/api/dvh', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                patient_id: state.patientId,
                structure: structureName
            })
        });
        
        if (!response.ok) {
            throw new Error(`Failed to fetch DVH: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        // Update DVH image
        elements.dvhImage.src = `data:image/png;base64,${data.image}`;
        
        // Update DVH statistics
        elements.dvhMin.textContent = `${data.statistics.min_dose.toFixed(2)} Gy`;
        elements.dvhMax.textContent = `${data.statistics.max_dose.toFixed(2)} Gy`;
        elements.dvhMean.textContent = `${data.statistics.mean_dose.toFixed(2)} Gy`;
        elements.dvhMedian.textContent = `${data.statistics.median_dose.toFixed(2)} Gy`;
        elements.dvhD95.textContent = `${data.statistics.d95.toFixed(2)} Gy`;
        elements.dvhV20.textContent = `${data.statistics.v20.toFixed(2)}%`;
        
        elements.dvhStats.classList.remove('d-none');
    } catch (error) {
        console.error('Error loading DVH:', error);
        alert('Error loading DVH. Please try again.');
        elements.dvhContainer.classList.add('d-none');
        elements.dvhStats.classList.add('d-none');
    } finally {
        elements.dvhLoading.classList.add('d-none');
    }
}

// Handle previous slice button
function handlePrevSlice() {
    if (state.sliceIndex > 0) {
        state.sliceIndex--;
        sliceSlider.noUiSlider.set(state.sliceIndex);
    }
}

// Handle next slice button
function handleNextSlice() {
    if (state.sliceIndex < state.sliceCount - 1) {
        state.sliceIndex++;
        sliceSlider.noUiSlider.set(state.sliceIndex);
    }
}

// Handle play slices button
function handlePlaySlices() {
    if (state.sliceAnimation) {
        // Stop animation
        clearInterval(state.sliceAnimation);
        state.sliceAnimation = null;
        elements.playSlices.innerHTML = '<i class="fas fa-play"></i>';
    } else {
        // Start animation
        elements.playSlices.innerHTML = '<i class="fas fa-pause"></i>';
        
        state.sliceAnimation = setInterval(() => {
            if (state.sliceIndex < state.sliceCount - 1) {
                state.sliceIndex++;
            } else {
                state.sliceIndex = 0;
            }
            sliceSlider.noUiSlider.set(state.sliceIndex);
        }, 200);
    }
}

// Handle zoom in button
function handleZoomIn() {
    state.imageScale += 0.1;
    applyImageTransform();
}

// Handle zoom out button
function handleZoomOut() {
    if (state.imageScale > 0.2) {
        state.imageScale -= 0.1;
        applyImageTransform();
    }
}

// Handle pan tool button
function handlePanTool() {
    elements.panTool.classList.toggle('active');
    
    if (elements.panTool.classList.contains('active')) {
        // Enable panning
        elements.imageContainer.style.cursor = 'move';
        setupPanEvents();
    } else {
        // Disable panning
        elements.imageContainer.style.cursor = 'default';
        removePanEvents();
    }
}

// Handle reset view button
function handleResetView() {
    state.imageScale = 1.0;
    state.panOffset = { x: 0, y: 0 };
    applyImageTransform();
    elements.panTool.classList.remove('active');
    elements.imageContainer.style.cursor = 'default';
}

// Handle screenshot button
function handleScreenshot() {
    // Create a temporary link
    const link = document.createElement('a');
    link.download = `dicom-slice-${state.patientId}-${state.sliceIndex}.png`;
    link.href = elements.dicomImage.src;
    
    // Click the link to trigger download
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// Apply image transformations
function applyImageTransform() {
    elements.dicomImage.style.transform = `scale(${state.imageScale}) translate(${state.panOffset.x}px, ${state.panOffset.y}px)`;
}

// Handle mouse wheel for slice navigation
function handleMouseWheel(event) {
    // Prevent default scrolling
    event.preventDefault();

    // If structure toggle is in progress, don't allow scrolling
    if (state.structureToggleInProgress) {
        console.log('Ignoring scroll during structure toggle');
        return;
    }

    // Create a debounced version of the scroll handler
    // This prevents too many slice changes in quick succession
    if (!state.scrollTimeout) {
        // Change slice based on wheel direction
        if (event.deltaY < 0) {
            // Scroll up, go to previous slice
            handlePrevSlice();
        } else {
            // Scroll down, go to next slice
            handleNextSlice();
        }

        // Set a timeout to prevent rapid scrolling
        state.scrollTimeout = setTimeout(() => {
            state.scrollTimeout = null;
        }, 50); // 50ms debounce
    }
}

// Set up panning events
function setupPanEvents() {
    let isDragging = false;
    let startX, startY;
    let startOffsetX = state.panOffset.x;
    let startOffsetY = state.panOffset.y;
    
    elements.imageContainer.addEventListener('mousedown', onMouseDown);
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
    
    function onMouseDown(event) {
        if (elements.panTool.classList.contains('active')) {
            isDragging = true;
            startX = event.clientX;
            startY = event.clientY;
            startOffsetX = state.panOffset.x;
            startOffsetY = state.panOffset.y;
        }
    }
    
    function onMouseMove(event) {
        if (isDragging) {
            const dx = (event.clientX - startX) / state.imageScale;
            const dy = (event.clientY - startY) / state.imageScale;
            
            state.panOffset.x = startOffsetX + dx;
            state.panOffset.y = startOffsetY + dy;
            
            applyImageTransform();
        }
    }
    
    function onMouseUp() {
        isDragging = false;
    }
}

// Remove panning events
function removePanEvents() {
    elements.imageContainer.removeEventListener('mousedown', onMouseDown);
    document.removeEventListener('mousemove', onMouseMove);
    document.removeEventListener('mouseup', onMouseUp);
}

// Setup touch events for mobile
function setupTouchEvents() {
    let touchStartX, touchStartY;
    let lastTouchX, lastTouchY;
    let startOffsetX, startOffsetY;
    let touchDistance;
    let startScale;
    
    elements.imageContainer.addEventListener('touchstart', onTouchStart, { passive: false });
    elements.imageContainer.addEventListener('touchmove', onTouchMove, { passive: false });
    elements.imageContainer.addEventListener('touchend', onTouchEnd, { passive: false });
    
    function onTouchStart(event) {
        event.preventDefault();
        
        if (event.touches.length === 1) {
            // Single touch for panning
            touchStartX = event.touches[0].clientX;
            touchStartY = event.touches[0].clientY;
            lastTouchX = touchStartX;
            lastTouchY = touchStartY;
            startOffsetX = state.panOffset.x;
            startOffsetY = state.panOffset.y;
        } else if (event.touches.length === 2) {
            // Two touches for pinch zoom
            const dx = event.touches[0].clientX - event.touches[1].clientX;
            const dy = event.touches[0].clientY - event.touches[1].clientY;
            touchDistance = Math.sqrt(dx * dx + dy * dy);
            startScale = state.imageScale;
        }
    }
    
    function onTouchMove(event) {
        event.preventDefault();
        
        if (event.touches.length === 1) {
            // Pan with single touch
            const currentX = event.touches[0].clientX;
            const currentY = event.touches[0].clientY;
            
            // Calculate delta
            const dx = (currentX - lastTouchX) / state.imageScale;
            const dy = (currentY - lastTouchY) / state.imageScale;
            
            // Update offsets
            state.panOffset.x += dx;
            state.panOffset.y += dy;
            
            // Apply transform
            applyImageTransform();
            
            // Update last touch position
            lastTouchX = currentX;
            lastTouchY = currentY;
        } else if (event.touches.length === 2) {
            // Pinch zoom with two touches
            const dx = event.touches[0].clientX - event.touches[1].clientX;
            const dy = event.touches[0].clientY - event.touches[1].clientY;
            const newTouchDistance = Math.sqrt(dx * dx + dy * dy);
            
            // Calculate new scale
            state.imageScale = startScale * (newTouchDistance / touchDistance);
            
            // Limit scale
            if (state.imageScale < 0.2) state.imageScale = 0.2;
            if (state.imageScale > 3) state.imageScale = 3;
            
            // Apply transform
            applyImageTransform();
        }
    }
    
    function onTouchEnd(event) {
        event.preventDefault();
    }
}

// Handle 3D structure rendering
async function handle3DRender() {
    if (!state.patientId) return;

    // Get currently selected structures
    const selectedStructures = state.selectedStructures;

    // If no structures are selected, show an alert
    if (selectedStructures.length === 0) {
        alert('Please select at least one structure to render in 3D.');
        return;
    }

    // Show loading indicator
    elements.threedContainer.classList.remove('d-none');
    elements.threedLoading.classList.remove('d-none');
    elements.threedInfo.classList.add('d-none');

    try {
        const response = await fetch('/api/3d', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                patient_id: state.patientId,
                structures: selectedStructures
            })
        });

        if (!response.ok) {
            throw new Error(`Failed to render 3D view: ${response.statusText}`);
        }

        const data = await response.json();

        // Hide loading
        elements.threedLoading.classList.add('d-none');

        // Update image and info
        elements.threedImage.src = `data:image/png;base64,${data.image}`;

        // Update structures list
        elements.threedStructuresList.innerHTML = '';
        data.structures_shown.forEach(structure => {
            elements.threedStructuresList.innerHTML += `
                <li class="list-group-item">${structure}</li>
            `;
        });

        // Show info
        elements.threedInfo.classList.remove('d-none');
    } catch (error) {
        console.error('Error rendering 3D view:', error);
        alert('Error rendering 3D view. Please try again.');
        elements.threedLoading.classList.add('d-none');
    }
}

// Update 3D panel availability
function update3DPanel() {
    // Show/hide based on whether we have structures
    if (state.patientId && document.querySelectorAll('.structure-checkbox').length > 0) {
        elements.threedAvailable.classList.remove('d-none');
        elements.threedUnavailable.classList.add('d-none');
    } else {
        elements.threedAvailable.classList.add('d-none');
        elements.threedUnavailable.classList.remove('d-none');
    }
}

// Show/hide loading indicator
function showLoading(show) {
    if (show) {
        elements.imageLoading.classList.remove('d-none');

        // Show on-demand indicator when preloading is disabled
        if (!state.usePreloading && state.patientId) {
            elements.onDemandIndicator.classList.remove('d-none');
        } else {
            elements.onDemandIndicator.classList.add('d-none');
        }
    } else {
        elements.imageLoading.classList.add('d-none');
        elements.onDemandIndicator.classList.add('d-none');
    }
}