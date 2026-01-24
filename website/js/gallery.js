// Configuration
const GALLERY_JSON_URL = 'https://prod-photo-gallery.s3.us-east-1.amazonaws.com/metadata/gallery.json';
const PHOTOS_PER_LOAD = 20;

// State
let allPhotos = [];
let displayedPhotos = [];
let currentIndex = 0;
let isLoading = false;
let lightbox = null;
let filteredPhotos = [];
let activeSearchMode = null; // 'color', 'label', or null
let selectedLabels = new Set(); // Track selected labels for multiselect
let colorSelected = false; // Track if user has selected a color

// Elements
const photoGrid = document.getElementById('photo-grid');
const loadingSpinner = document.getElementById('loading');
const loadMoreTrigger = document.getElementById('load-more-trigger');

// Gallery Info Expander
const galleryInfoToggle = document.getElementById('gallery-info-toggle');
const galleryInfoContent = document.getElementById('gallery-info-content');

if (galleryInfoToggle && galleryInfoContent) {
    galleryInfoToggle.addEventListener('click', () => {
        const isExpanded = galleryInfoToggle.getAttribute('aria-expanded') === 'true';

        if (isExpanded) {
            galleryInfoToggle.setAttribute('aria-expanded', 'false');
            galleryInfoContent.classList.remove('expanded');
        } else {
            galleryInfoToggle.setAttribute('aria-expanded', 'true');
            galleryInfoContent.classList.add('expanded');
        }
    });
}

// Fetch gallery data
async function fetchGalleryData() {
    try {
        const response = await fetch(GALLERY_JSON_URL);
        if (!response.ok) {
            throw new Error(`Failed to fetch gallery data: ${response.status}`);
        }

        const data = await response.json();
        allPhotos = shuffleArray(data.photos || []);

        console.log(`Loaded ${allPhotos.length} photos from gallery`);

        // Populate label dropdown
        populateLabelDropdown();

        // Load initial batch
        loadMorePhotos();

    } catch (error) {
        console.error('Error fetching gallery:', error);
        loadingSpinner.innerHTML = `
            <p style="color: var(--error-color);">
                Error loading gallery. Please try again later.
            </p>
        `;
    }
}

// Extract all labels and count frequencies
function getLabelFrequencies() {
    const labelCounts = {};

    allPhotos.forEach(photo => {
        if (photo.rekognition && photo.rekognition.labels) {
            photo.rekognition.labels.forEach(label => {
                const labelName = label.name;
                labelCounts[labelName] = (labelCounts[labelName] || 0) + 1;
            });
        }
    });

    // Convert to array and sort by frequency (descending)
    return Object.entries(labelCounts)
        .map(([name, count]) => ({ name, count }))
        .sort((a, b) => b.count - a.count);
}

// Populate label dropdown with sorted labels
function populateLabelDropdown() {
    const labelOptions = document.getElementById('label-options');
    if (!labelOptions) return;

    const labels = getLabelFrequencies();

    labelOptions.innerHTML = labels.map(label => `
        <div class="multiselect-option" data-label="${label.name}">
            <input type="checkbox" id="label-${label.name}" value="${label.name}">
            <label for="label-${label.name}">
                <span class="label-name">${label.name}</span>
                <span class="label-count">(${label.count})</span>
            </label>
        </div>
    `).join('');

    // Add event listeners to checkboxes
    labelOptions.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
        checkbox.addEventListener('change', handleLabelSelection);
    });
}

// Shuffle array (Fisher-Yates algorithm)
function shuffleArray(array) {
    const shuffled = [...array];
    for (let i = shuffled.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    return shuffled;
}

// Convert hex color to RGB
function hexToRgb(hex) {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16)
    } : null;
}

// Calculate perceptually-weighted color distance between two RGB colors
// Uses weighted Euclidean distance that better matches human perception
// Returns a value from 0 (identical) to higher values (more different)
function colorDistance(rgb1, rgb2) {
    // Calculate luminance (perceived brightness) using standard weights
    // Humans perceive green as brightest, then red, then blue
    const lum1 = 0.299 * rgb1.r + 0.587 * rgb1.g + 0.114 * rgb1.b;
    const lum2 = 0.299 * rgb2.r + 0.587 * rgb2.g + 0.114 * rgb2.b;
    const lumDiff = Math.abs(lum1 - lum2);

    // Calculate chroma (color) difference using weighted RGB
    const rDiff = rgb1.r - rgb2.r;
    const gDiff = rgb1.g - rgb2.g;
    const bDiff = rgb1.b - rgb2.b;

    // Weighted Euclidean distance (redmean approximation)
    const rMean = (rgb1.r + rgb2.r) / 2;
    const r = rDiff * rDiff * (2 + rMean / 256);
    const g = gDiff * gDiff * 4;
    const b = bDiff * bDiff * (2 + (255 - rMean) / 256);
    const chromaDistance = Math.sqrt(r + g + b);

    // Combine luminance and chroma with extra weight on luminance
    // This prevents dark blue from matching black, light blue from matching white, etc.
    // Weight luminance more heavily (3x) to prevent brightness mismatches
    return chromaDistance + (lumDiff * 3);
}

function calculateColorScore(photo, targetHex) {
    if (!photo.rekognition || !photo.rekognition.dominant_colors || photo.rekognition.dominant_colors.length === 0) {
        return Infinity; // Photos without color data go to the end
    }

    const targetRgb = hexToRgb(targetHex);
    if (!targetRgb) return Infinity;

    // Distance threshold: colors beyond this are considered "not a match"
    // 150 is roughly the perceptual distance between fairly different colors
    const MAX_DISTANCE = 200;

    // Get top 5 dominant colors (look at more colors for better matching)
    const dominantColors = photo.rekognition.dominant_colors.slice(0, 5);

    // Calculate weighted score for each color
    // Lower score is better (closer color + higher percentage)
    const scores = dominantColors.map(color => {
        const colorRgb = hexToRgb(color.hex);
        if (!colorRgb) return Infinity;

        const distance = colorDistance(targetRgb, colorRgb);

        // Only include reasonably distance images
        if (distance > MAX_DISTANCE) {
            return Infinity;
        }

        const percentage = parseFloat(color.pixel_percentage);

        return (distance * distance + 10) / percentage;
    });

    // Return the best (lowest) score from all dominant colors
    const bestScore = Math.min(...scores);

    // If no colors matched (all beyond threshold), return Infinity
    return bestScore;
}

function performSearch() {
    const colorPicker = document.getElementById('color-picker');
    const hexColor = colorPicker ? colorPicker.value : '#ffffff';

    // Step 1: Filter by labels (if any are selected)
    let photosToSearch;
    if (selectedLabels.size > 0) {
        photosToSearch = allPhotos.filter(photo => {
            if (!photo.rekognition || !photo.rekognition.labels) return false;
            return photo.rekognition.labels.some(label =>
                selectedLabels.has(label.name)
            );
        });
        activeSearchMode = 'active';
    } else {
        photosToSearch = allPhotos;
    }

    if (colorSelected) {
        const photosWithScores = photosToSearch.map(photo => ({
            photo,
            score: calculateColorScore(photo, hexColor)
        }));

        // Filter out photos that didn't match (score = Infinity)
        const matchedPhotos = photosWithScores.filter(item => item.score !== Infinity);

        // Sort by score (lower is better)
        matchedPhotos.sort((a, b) => a.score - b.score);

        // Extract sorted photos
        filteredPhotos = matchedPhotos.map(item => item.photo);
    } else {
        // No color filter, use label-filtered photos as-is
        filteredPhotos = photosToSearch;
    }

    // Set active search mode if either labels or color is selected
    if (selectedLabels.size > 0 || colorSelected) {
        activeSearchMode = 'active';
    }

    // Reset and reload gallery
    resetGallery();
    loadMorePhotos();
}

// Handle label checkbox selection
function handleLabelSelection(event) {
    const label = event.target.value;

    if (event.target.checked) {
        selectedLabels.add(label);
    } else {
        selectedLabels.delete(label);
    }

    updateLabelDisplay();
}

// Update the multiselect display to show selected labels
function updateLabelDisplay() {
    const display = document.getElementById('label-display');
    if (!display) return;

    if (selectedLabels.size === 0) {
        display.innerHTML = '<span class="placeholder">Select labels...</span>';
    } else {
        const labelTags = Array.from(selectedLabels).map(label => `
            <span class="selected-label-tag">${label}</span>
        `).join('');
        display.innerHTML = labelTags;
    }
}

// Reset search and show all photos
function resetSearch() {
    activeSearchMode = null;
    filteredPhotos = [];
    selectedLabels.clear();
    colorSelected = false;

    // Clear all checkboxes
    document.querySelectorAll('#label-options input[type="checkbox"]').forEach(checkbox => {
        checkbox.checked = false;
    });

    updateLabelDisplay();
    resetGallery();
    allPhotos = shuffleArray(allPhotos);
    loadMorePhotos();
}

// Reset gallery display
function resetGallery() {
    displayedPhotos = [];
    currentIndex = 0;
    photoGrid.innerHTML = '';
    if (lightbox) {
        lightbox.destroy();
        lightbox = null;
    }
}

// Load more photos
function loadMorePhotos() {
    // Use filtered photos if search is active, otherwise use all photos
    const photosToDisplay = activeSearchMode ? filteredPhotos : allPhotos;

    // Show "no results" message if search is active but no photos match
    if (activeSearchMode && photosToDisplay.length === 0 && currentIndex === 0) {
        loadingSpinner.style.display = 'none';
        photoGrid.innerHTML = '<div style="grid-column: 1/-1; text-align: center; padding: 2rem; color: var(--text-color);"><p>No photos match your search criteria. Try a different color or adjust your filters.</p></div>';
        return;
    }

    if (isLoading || currentIndex >= photosToDisplay.length) {
        loadingSpinner.style.display = 'none';
        return;
    }

    isLoading = true;
    loadingSpinner.style.display = 'flex';

    // Get next batch
    const batch = photosToDisplay.slice(currentIndex, currentIndex + PHOTOS_PER_LOAD);
    currentIndex += PHOTOS_PER_LOAD;

    // Render photos
    batch.forEach(photo => {
        displayedPhotos.push(photo);
        renderPhoto(photo);
    });

    // Initialize/refresh GLightbox after adding new photos
    if (lightbox) {
        lightbox.destroy();
    }
    lightbox = GLightbox({
        touchNavigation: true,
        loop: true,
        autoplayVideos: true,
        onOpen: () => {
            createInfoPanel();
            // Wait for GLightbox to fully initialize the slide
            setTimeout(() => {
                updateInfoPanel();
                updateDownloadMenu();
            }, 100);
        },
        onClose: () => {
            removeInfoPanel();
        }
    });

    // Add event listener for slide changes using GLightbox's event system
    lightbox.on('slide_changed', () => {
        console.log('Slide changed, updating info panel and download menu...');
        setTimeout(() => {
            updateInfoPanel();
            updateDownloadMenu();
        }, 100);
    });

    isLoading = false;
    loadingSpinner.style.display = 'none';
}

// Create info panel for lightbox
function createInfoPanel() {
    // Create info button
    const infoButton = document.createElement('div');
    infoButton.className = 'ginfo-button';
    infoButton.innerHTML = 'i';
    infoButton.onclick = toggleInfoPanel;
    document.body.appendChild(infoButton);

    // Create download button
    const downloadButton = document.createElement('div');
    downloadButton.className = 'gdownload-button';
    downloadButton.innerHTML = '↓';
    downloadButton.onclick = toggleDownloadMenu;
    document.body.appendChild(downloadButton);

    // Create download menu
    const downloadMenu = document.createElement('div');
    downloadMenu.className = 'gdownload-menu';
    downloadMenu.id = 'photo-download-menu';
    document.body.appendChild(downloadMenu);

    // Create info panel
    const infoPanel = document.createElement('div');
    infoPanel.className = 'ginfo-panel';
    infoPanel.id = 'photo-info-panel';
    document.body.appendChild(infoPanel);

    updateInfoPanel();
    updateDownloadMenu();
}

// Remove info panel and download menu
function removeInfoPanel() {
    const infoButton = document.querySelector('.ginfo-button');
    const downloadButton = document.querySelector('.gdownload-button');
    const downloadMenu = document.getElementById('photo-download-menu');
    const panel = document.getElementById('photo-info-panel');
    if (infoButton) infoButton.remove();
    if (downloadButton) downloadButton.remove();
    if (downloadMenu) downloadMenu.remove();
    if (panel) panel.remove();
}

// Toggle info panel
function toggleInfoPanel() {
    const panel = document.getElementById('photo-info-panel');
    const downloadMenu = document.getElementById('photo-download-menu');

    if (panel) {
        panel.classList.toggle('active');
    }

    // Close download menu if open
    if (downloadMenu && downloadMenu.classList.contains('active')) {
        downloadMenu.classList.remove('active');
    }
}

// Toggle download menu
function toggleDownloadMenu() {
    const downloadMenu = document.getElementById('photo-download-menu');
    const panel = document.getElementById('photo-info-panel');

    if (downloadMenu) {
        downloadMenu.classList.toggle('active');
    }

    // Close info panel if open
    if (panel && panel.classList.contains('active')) {
        panel.classList.remove('active');
    }
}

// Format dimensions for display
function formatDimensions(width, height) {
    if (!width || !height) return '';
    return `${width} × ${height}`;
}

// Update download menu content
async function updateDownloadMenu() {
    const menu = document.getElementById('photo-download-menu');
    if (!menu) return;

    // Get current slide
    const currentSlide = document.querySelector('.gslide.current');
    if (!currentSlide) return;

    const slideContainer = currentSlide.querySelector('.gslide-media');
    if (!slideContainer) return;

    const slideImage = slideContainer.querySelector('img');
    if (!slideImage) return;

    // Get photo data using the same matching logic
    const imageSrc = slideImage.src;
    let imagePath = imageSrc;
    try {
        const url = new URL(imageSrc);
        imagePath = url.pathname;
    } catch (e) {
        // Use full src if URL parsing fails
    }

    const photoCard = Array.from(document.querySelectorAll('.photo-card')).find(card => {
        const cardData = JSON.parse(card.getAttribute('data-photo-info'));
        const largePath = cardData.images.large;
        return imagePath.includes(largePath) || card.href === imageSrc;
    });

    if (!photoCard) {
        menu.innerHTML = '<div class="download-error">Unable to load download options</div>';
        return;
    }

    const photoData = JSON.parse(photoCard.getAttribute('data-photo-info'));
    const baseUrl = 'https://prod-photo-gallery.s3.us-east-1.amazonaws.com';

    // Define download options (medium -> small for display)
    const downloadOptions = [
        { label: 'Small', key: 'medium', path: photoData.images.medium },
        { label: 'Large', key: 'large', path: photoData.images.large },
        { label: 'Full', key: 'full', path: photoData.images.full }
    ];

    // Build options with dimensions from metadata
    const optionsWithDimensions = downloadOptions.map(option => {
        const fullUrl = `${baseUrl}/${option.path}`;
        const dims = photoData.dimensions?.[option.key];
        const dimensionStr = dims ? formatDimensions(dims.width, dims.height) : 'Unknown size';
        return { ...option, dimensions: dimensionStr, url: fullUrl };
    });

    // Build menu HTML
    let html = '<div class="download-menu-header">Download Image</div>';
    html += '<div class="download-options">';

    optionsWithDimensions.forEach(option => {
        html += `
            <div class="download-option" onclick="downloadImage('${option.url}', '${photoData.title}_${option.label}.jpg')">
                <span class="download-label">${option.label}</span>
                <span class="download-size">${option.dimensions}</span>
            </div>
        `;
    });

    html += '</div>';
    menu.innerHTML = html;
}

// Download image
async function downloadImage(url, filename) {
    try {
        // Show a loading indicator
        const downloadMenu = document.getElementById('photo-download-menu');
        if (downloadMenu) {
            const originalContent = downloadMenu.innerHTML;
            downloadMenu.innerHTML = '<div class="download-loading">Preparing download...</div>';

            // Fetch the image with explicit CORS mode
            const response = await fetch(url, {
                method: 'GET',
                mode: 'cors',  // Explicitly request CORS
                credentials: 'omit',  // Don't send credentials
                cache: 'no-cache'  // Bypass cache to ensure fresh CORS headers
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const blob = await response.blob();

            // Create a blob URL and trigger download
            const blobUrl = window.URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = blobUrl;
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);

            // Clean up the blob URL
            window.URL.revokeObjectURL(blobUrl);

            // Restore menu and close it
            downloadMenu.innerHTML = originalContent;
            downloadMenu.classList.remove('active');
        }
    } catch (error) {
        console.error('Error downloading image:', error);
        console.error('Error details:', {
            message: error.message,
            stack: error.stack,
            url: url
        });

        const downloadMenu = document.getElementById('photo-download-menu');
        if (downloadMenu) {
            downloadMenu.innerHTML = `
                <div class="download-error">
                    <p>Download failed. Check console for details.</p>
                    <p style="font-size: 0.8em; color: #666;">${error.message}</p>
                </div>
            `;
        }

        alert('Failed to download image. Please check the browser console for details.');
    }
}

// Update info panel content
function updateInfoPanel() {
    const panel = document.getElementById('photo-info-panel');
    if (!panel) {
        console.log('Panel not found');
        return;
    }

    // Get current slide
    const currentSlide = document.querySelector('.gslide.current');
    if (!currentSlide) {
        console.log('Current slide not found');
        return;
    }

    const slideContainer = currentSlide.querySelector('.gslide-media');
    if (!slideContainer) {
        console.log('Slide container not found');
        return;
    }

    // Find the original photo card to get data
    const slideImage = slideContainer.querySelector('img');
    if (!slideImage) {
        console.log('Slide image not found');
        return;
    }

    const imageSrc = slideImage.src;
    console.log('Looking for image:', imageSrc);

    // Extract just the path part from the URL for matching
    let imagePath = imageSrc;
    try {
        const url = new URL(imageSrc);
        imagePath = url.pathname; // Gets just the path part like "/processed/large/DSCF2987.jpg"
    } catch (e) {
        console.log('Could not parse URL, using full src');
    }

    // Try to find matching photo card by comparing image paths
    let photoCard = Array.from(document.querySelectorAll('.photo-card')).find(card => {
        const cardData = JSON.parse(card.getAttribute('data-photo-info'));
        const largePath = cardData.images.large;
        // Check if the image path ends with or contains the large image path
        return imagePath.includes(largePath) || card.href === imageSrc;
    });

    if (!photoCard) {
        console.log('Photo card not found. Available cards:', document.querySelectorAll('.photo-card').length);
        console.log('Image path we are looking for:', imagePath);
        panel.innerHTML = '<div class="info-section"><p>Unable to load photo data</p></div>';
        return;
    }

    const photoData = JSON.parse(photoCard.getAttribute('data-photo-info'));
    console.log('Photo data loaded for:', photoData.title);

    // Check if bounding boxes are currently enabled (before we rebuild the panel)
    const currentCheckbox = document.getElementById('bbox-toggle');
    const bboxWasEnabled = currentCheckbox && currentCheckbox.checked;
    console.log('Bbox was enabled:', bboxWasEnabled);

    // Remove any existing bounding boxes before updating panel
    removeBoundingBoxes();

    // Build info panel HTML with optimized grid layout
    let html = `
        <h3>${photoData.title || 'Untitled'}</h3>
        <div class="ginfo-panel-content">
    `;

    // ROW 1: Caption + Bounding Box Toggle
    const hasInstances = photoData.rekognition && photoData.rekognition.labels &&
        photoData.rekognition.labels.some(label => label.instances && label.instances.length > 0);

    html += '<div class="info-row row-caption-toggle">';

    // Caption (takes most of the row)
    if (photoData.caption) {
        html += `
            <div class="info-section caption-section">
                <div class="info-label">Caption</div>
                <div class="info-content">${photoData.caption}</div>
            </div>
        `;
    }

    // Bounding box toggle (on the right)
    if (hasInstances) {
        html += `
            <div class="info-section bbox-toggle-section">
                <div class="info-label">Object Detection</div>
                <div class="toggle-container">
                    <label class="toggle-switch">
                        <input type="checkbox" id="bbox-toggle" onchange="toggleBoundingBoxes()">
                        <span class="toggle-slider"></span>
                    </label>
                    <span class="toggle-label">Show Boxes</span>
                </div>
            </div>
        `;
    }

    html += '</div>'; // Close row 1

    // ROW 2: Dominant Colors + Labels
    html += '<div class="info-row row-colors-labels">';

    // Dominant colors (left side)
    if (photoData.rekognition && photoData.rekognition.dominant_colors && photoData.rekognition.dominant_colors.length > 0) {
        const topColors = photoData.rekognition.dominant_colors.slice(0, 3);
        html += `
            <div class="info-section colors-section">
                <div class="info-label">Dominant Colors</div>
                <div class="colors-container">
                    ${topColors.map(color => `
                        <div class="color-swatch">
                            <div class="color-box" style="background-color: ${color.hex}"></div>
                            <div class="color-info">
                                <div class="color-name">${color.simplified_color}</div>
                                <div class="color-percent">${color.pixel_percentage}%</div>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }

    // Labels (right side)
    if (photoData.rekognition && photoData.rekognition.labels && photoData.rekognition.labels.length > 0) {
        const topLabels = photoData.rekognition.labels.slice(0, 10);
        html += `
            <div class="info-section labels-section">
                <div class="info-label">Detected Labels</div>
                <div class="labels-container">
                    ${topLabels.map(label => `<div class="label-tag">${label.name}</div>`).join('')}
                </div>
            </div>
        `;
    }

    html += '</div>'; // Close row 2

    // ROW 3: Camera Settings (2x3 grid - flattened)
    if (photoData.exif_info) {
        const exif = photoData.exif_info;
        html += `
            <div class="info-row row-camera-settings">
                <div class="info-section camera-settings-section">
                    <div class="info-label">Camera & Settings</div>
                    <div class="camera-settings-grid-flat">
                        ${exif.camera ? `<div class="exif-item"><strong>Camera:</strong> ${exif.camera}</div>` : ''}
                        ${exif.focal_length ? `<div class="exif-item"><strong>Focal Length:</strong> ${exif.focal_length}mm</div>` : ''}
                        ${exif.exposure ? `<div class="exif-item"><strong>Shutter:</strong> ${exif.exposure}s</div>` : ''}
                        ${exif.lens ? `<div class="exif-item"><strong>Lens:</strong> ${exif.lens}</div>` : ''}
                        ${exif.f ? `<div class="exif-item"><strong>Aperture:</strong> ${exif.f}</div>` : ''}
                        ${exif.iso ? `<div class="exif-item"><strong>ISO:</strong> ${exif.iso}</div>` : ''}
                    </div>
                </div>
            </div>
        `; // Close row 3
    }
    html += `</div>`; // Close ginfo-panel-content

    panel.innerHTML = html;

    // Restore the bounding box toggle state and re-render boxes if they were enabled
    setTimeout(() => {
        const checkbox = document.getElementById('bbox-toggle');
        console.log('Restoring bbox state, checkbox exists:', !!checkbox, 'was enabled:', bboxWasEnabled);
        if (checkbox && bboxWasEnabled) {
            checkbox.checked = true;
            console.log('Re-rendering bounding boxes for new image');
            renderBoundingBoxes();
        }
    }, 100);
}

// Render a single photo
function renderPhoto(photo) {
    const baseUrl = 'https://prod-photo-gallery.s3.us-east-1.amazonaws.com';

    // Create photo card
    const photoCard = document.createElement('a');
    photoCard.className = 'glightbox photo-card';
    photoCard.href = `${baseUrl}/${photo.images.large}`;

    // Store photo data for info panel
    photoCard.setAttribute('data-photo-info', JSON.stringify(photo));

    // Create thumbnail image
    const img = document.createElement('img');
    img.src = `${baseUrl}/${photo.images.thumbnail}`;
    img.alt = photo.title || 'Photo';
    img.loading = 'lazy';

    photoCard.appendChild(img);
    photoGrid.appendChild(photoCard);
}

// Intersection Observer for infinite scroll
const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        const photosToDisplay = activeSearchMode ? filteredPhotos : allPhotos;
        if (entry.isIntersecting && !isLoading && currentIndex < photosToDisplay.length) {
            loadMorePhotos();
        }
    });
}, {
    rootMargin: '200px'
});

observer.observe(loadMoreTrigger);

// Toggle bounding boxes on/off
function toggleBoundingBoxes() {
    const checkbox = document.getElementById('bbox-toggle');
    if (!checkbox) return;

    if (checkbox.checked) {
        renderBoundingBoxes();
    } else {
        removeBoundingBoxes();
    }
}

// Render bounding boxes as SVG overlay
function renderBoundingBoxes() {
    // Remove any existing overlay first
    removeBoundingBoxes();

    // Get current slide and photo data
    const currentSlide = document.querySelector('.gslide.current');
    if (!currentSlide) return;

    const slideContainer = currentSlide.querySelector('.gslide-media');
    if (!slideContainer) return;

    const slideImage = slideContainer.querySelector('img');
    if (!slideImage) return;

    // Get photo data using the same matching logic as updateInfoPanel
    const imageSrc = slideImage.src;
    let imagePath = imageSrc;
    try {
        const url = new URL(imageSrc);
        imagePath = url.pathname;
    } catch (e) {
        // Use full src if URL parsing fails
    }

    const photoCard = Array.from(document.querySelectorAll('.photo-card')).find(card => {
        const cardData = JSON.parse(card.getAttribute('data-photo-info'));
        const largePath = cardData.images.large;
        return imagePath.includes(largePath) || card.href === imageSrc;
    });
    if (!photoCard) return;

    const photoData = JSON.parse(photoCard.getAttribute('data-photo-info'));

    // Get all instances with bounding boxes
    const instances = [];
    if (photoData.rekognition && photoData.rekognition.labels) {
        photoData.rekognition.labels.forEach(label => {
            if (label.instances && label.instances.length > 0) {
                label.instances.forEach(instance => {
                    if (instance.BoundingBox) {
                        instances.push({
                            label: label.name,
                            confidence: instance.Confidence,
                            box: instance.BoundingBox
                        });
                    }
                });
            }
        });
    }

    if (instances.length === 0) return;

    // Get image dimensions
    const imgRect = slideImage.getBoundingClientRect();
    const imgWidth = slideImage.naturalWidth;
    const imgHeight = slideImage.naturalHeight;
    const displayWidth = imgRect.width;
    const displayHeight = imgRect.height;

    // Create SVG overlay
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('class', 'bbox-overlay');
    svg.style.position = 'absolute';
    svg.style.top = '0';
    svg.style.left = '0';
    svg.style.width = '100%';
    svg.style.height = '100%';
    svg.style.pointerEvents = 'none';
    svg.style.zIndex = '1';

    // Calculate scale factor (handles image being scaled to fit container)
    const scaleX = displayWidth / imgWidth;
    const scaleY = displayHeight / imgHeight;
    const scale = Math.min(scaleX, scaleY);

    // Calculate offset (handles centering of scaled image)
    const scaledWidth = imgWidth * scale;
    const scaledHeight = imgHeight * scale;
    const offsetX = (displayWidth - scaledWidth) / 2;
    const offsetY = (displayHeight - scaledHeight) / 2;

    // Draw each bounding box
    instances.forEach((instance) => {
        const box = instance.box;

        // Convert normalized coordinates (0-1) to pixel coordinates
        const x = (box.Left * imgWidth * scale) + offsetX;
        const y = (box.Top * imgHeight * scale) + offsetY;
        const width = box.Width * imgWidth * scale;
        const height = box.Height * imgHeight * scale;

        // Create rectangle
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('x', x);
        rect.setAttribute('y', y);
        rect.setAttribute('width', width);
        rect.setAttribute('height', height);
        rect.setAttribute('class', 'bbox-rect');
        rect.style.fill = 'none';
        rect.style.stroke = '#00ff00';
        rect.style.strokeWidth = '3';
        rect.style.opacity = '0.8';

        // Create label background
        const labelBg = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        const labelText = `${instance.label} (${instance.confidence.toFixed(1)}%)`;
        const labelWidth = labelText.length * 8 + 16;
        const labelHeight = 24;

        // Determine label position: above box if there's room, otherwise inside at top
        const labelAboveBox = y - labelHeight >= offsetY;
        const labelY = labelAboveBox ? y - labelHeight : y;
        const textY = labelAboveBox ? y - 7 : y + 17;

        labelBg.setAttribute('x', x);
        labelBg.setAttribute('y', labelY);
        labelBg.setAttribute('width', labelWidth);
        labelBg.setAttribute('height', labelHeight);
        labelBg.setAttribute('class', 'bbox-label-bg');
        labelBg.style.fill = '#00ff00';
        labelBg.style.opacity = '0.9';

        // Create label text
        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('x', x + 8);
        text.setAttribute('y', textY);
        text.setAttribute('class', 'bbox-label-text');
        text.style.fill = '#000';
        text.style.fontSize = '14px';
        text.style.fontWeight = 'bold';
        text.style.fontFamily = 'Arial, sans-serif';
        text.textContent = labelText;

        // Add to SVG
        svg.appendChild(rect);
        svg.appendChild(labelBg);
        svg.appendChild(text);
    });

    // Add SVG to the slide container
    slideContainer.style.position = 'relative';
    slideContainer.appendChild(svg);
}

// Remove bounding boxes
function removeBoundingBoxes() {
    const overlays = document.querySelectorAll('.bbox-overlay');
    overlays.forEach(overlay => overlay.remove());
}

// Event listeners for search controls
document.addEventListener('DOMContentLoaded', () => {
    const colorPicker = document.getElementById('color-picker');
    const searchBtn = document.getElementById('search-btn');
    const labelDropdown = document.getElementById('label-dropdown');
    const labelDisplay = document.getElementById('label-display');
    const labelOptions = document.getElementById('label-options');
    const resetBtn = document.getElementById('reset-btn');

    // Track when user changes color
    colorPicker.addEventListener('input', () => {
        colorSelected = true;
    });

    // Unified search button
    searchBtn.addEventListener('click', () => {
        performSearch();
        labelDropdown.classList.remove('open');
    });

    // Toggle label dropdown
    labelDisplay.addEventListener('click', (e) => {
        e.stopPropagation();
        labelDropdown.classList.toggle('open');
    });

    // Close dropdown when clicking outside
    document.addEventListener('click', (e) => {
        if (!labelDropdown.contains(e.target)) {
            labelDropdown.classList.remove('open');
        }
    });

    // Prevent dropdown from closing when clicking inside options
    labelOptions.addEventListener('click', (e) => {
        e.stopPropagation();
    });

    // Reset button
    resetBtn.addEventListener('click', () => {
        resetSearch();
        colorPicker.value = '#ffffff';
    });
});

// Initialize
fetchGalleryData();
