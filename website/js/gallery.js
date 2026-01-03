// Configuration
const GALLERY_JSON_URL = 'https://prod-photo-gallery.s3.us-east-1.amazonaws.com/metadata/gallery.json';
const PHOTOS_PER_LOAD = 20;

// State
let allPhotos = [];
let displayedPhotos = [];
let currentIndex = 0;
let isLoading = false;
let lightbox = null;

// Elements
const photoGrid = document.getElementById('photo-grid');
const loadingSpinner = document.getElementById('loading');
const loadMoreTrigger = document.getElementById('load-more-trigger');

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

// Shuffle array (Fisher-Yates algorithm)
function shuffleArray(array) {
    const shuffled = [...array];
    for (let i = shuffled.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    return shuffled;
}

// Load more photos
function loadMorePhotos() {
    if (isLoading || currentIndex >= allPhotos.length) {
        loadingSpinner.style.display = 'none';
        return;
    }

    isLoading = true;
    loadingSpinner.style.display = 'flex';

    // Get next batch
    const batch = allPhotos.slice(currentIndex, currentIndex + PHOTOS_PER_LOAD);
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
        autoplayVideos: true
    });

    isLoading = false;
    loadingSpinner.style.display = 'none';
}

// Render a single photo
function renderPhoto(photo) {
    const baseUrl = 'https://prod-photo-gallery.s3.us-east-1.amazonaws.com';

    // Create photo card
    const photoCard = document.createElement('a');
    photoCard.className = 'glightbox photo-card';
    photoCard.href = `${baseUrl}/${photo.images.large}`;

    // Build description with caption and location
    let description = photo.title || 'Untitled';
    if (photo.caption) {
        description += `<br>${photo.caption}`;
    }
    if (photo.location && photo.location.name) {
        description += `<br>📍 ${photo.location.name}`;
    }
    photoCard.setAttribute('data-glightbox', `description: ${description}`);

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
        if (entry.isIntersecting && !isLoading && currentIndex < allPhotos.length) {
            loadMorePhotos();
        }
    });
}, {
    rootMargin: '200px'
});

observer.observe(loadMoreTrigger);

// Initialize
fetchGalleryData();
