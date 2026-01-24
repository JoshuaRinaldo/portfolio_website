// Theme toggle functionality with system preference detection

(function() {
    const STORAGE_KEY = 'theme-preference';

    // Get theme preference
    function getThemePreference() {
        // Check if user has saved preference
        const savedTheme = localStorage.getItem(STORAGE_KEY);
        if (savedTheme) {
            return savedTheme;
        }

        // Default to system preference
        if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
            return 'dark';
        }

        return 'light';
    }

    // Apply theme
    function setTheme(theme) {
        if (theme === 'dark') {
            document.documentElement.setAttribute('data-theme', 'dark');
        } else {
            document.documentElement.removeAttribute('data-theme');
        }
        localStorage.setItem(STORAGE_KEY, theme);
        updateToggleButton(theme);

        // Update particles color if particles.js is loaded
        updateParticlesColor(theme);
    }

    // Update toggle button appearance
    function updateToggleButton(theme) {
        const toggle = document.querySelector('.theme-toggle');
        if (toggle) {
            if (theme === 'dark') {
                toggle.classList.add('theme-toggle--toggled');
            } else {
                toggle.classList.remove('theme-toggle--toggled');
            }
            toggle.setAttribute('aria-label', theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode');
        }
    }

    // Update particles.js color dynamically
    function updateParticlesColor(theme) {
        if (window.pJSDom && window.pJSDom.length > 0) {
            const pJS = window.pJSDom[0].pJS;
            const color = theme === 'dark' ? '#ffffff' : '#000000';

            // Update particle color configuration
            pJS.particles.color.value = color;
            pJS.particles.line_linked.color = color;

            // Update all existing particles
            pJS.particles.array.forEach(particle => {
                particle.color.value = color;
                particle.color.rgb = hexToRgb(color);
            });

            // Trigger a redraw
            pJS.fn.particlesRefresh();
        }
    }

    // Helper function to convert hex to RGB
    function hexToRgb(hex) {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? {
            r: parseInt(result[1], 16),
            g: parseInt(result[2], 16),
            b: parseInt(result[3], 16)
        } : null;
    }

    // Toggle theme
    function toggleTheme() {
        const currentTheme = document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        setTheme(newTheme);
    }

    // Initialize theme on page load
    function init() {
        // Create toggle button with SVG markup
        const toggle = document.createElement('button');
        toggle.className = 'theme-toggle';
        toggle.type = 'button';
        toggle.title = 'Toggle theme';
        toggle.setAttribute('aria-label', 'Toggle theme');
        toggle.innerHTML = `
            <svg
                xmlns="http://www.w3.org/2000/svg"
                aria-hidden="true"
                width="1em"
                height="1em"
                fill="currentColor"
                stroke-linecap="round"
                class="theme-toggle__classic"
                viewBox="0 0 32 32"
            >
                <clipPath id="theme-toggle__classic__cutout">
                    <path d="M0-5h30a1 1 0 0 0 9 13v24H0Z" />
                </clipPath>
                <g clip-path="url(#theme-toggle__classic__cutout)">
                    <circle cx="16" cy="16" r="8.34" />
                    <g stroke="currentColor" stroke-width="1.5">
                        <path d="M16 5.5v-4" />
                        <path d="M16 30.5v-4" />
                        <path d="M1.5 16h4" />
                        <path d="M26.5 16h4" />
                        <path d="m23.4 8.6 2.8-2.8" />
                        <path d="m5.7 26.3 2.9-2.9" />
                        <path d="m5.8 5.8 2.8 2.8" />
                        <path d="m23.4 23.4 2.9 2.9" />
                    </g>
                </g>
            </svg>
        `;
        toggle.addEventListener('click', toggleTheme);

        // Append to navbar container, or fallback to body if navbar doesn't exist
        const navbarContainer = document.querySelector('.navbar .container');
        if (navbarContainer) {
            navbarContainer.appendChild(toggle);
        } else {
            document.body.appendChild(toggle);
        }

        // Set the theme
        const theme = getThemePreference();
        setTheme(theme);

        // Listen for system theme changes
        if (window.matchMedia) {
            window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
                // Only auto-switch if user hasn't manually set a preference
                if (!localStorage.getItem(STORAGE_KEY)) {
                    setTheme(e.matches ? 'dark' : 'light');
                }
            });
        }
    }

    // Apply theme immediately (before DOM ready) to prevent flash
    const initialTheme = getThemePreference();
    if (initialTheme === 'dark') {
        document.documentElement.setAttribute('data-theme', 'dark');
    } else {
        // IMPORTANT: Remove dark theme attribute if light mode
        document.documentElement.removeAttribute('data-theme');
    }

    // Expose setTheme globally so particles can call it if needed
    window.updateThemeParticles = function() {
        const currentTheme = document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
        updateParticlesColor(currentTheme);
    };

    // Run on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
