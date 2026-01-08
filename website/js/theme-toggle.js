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
            toggle.textContent = theme === 'dark' ? '☀️' : '🌙';
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
        const theme = getThemePreference();
        setTheme(theme);

        // Create toggle button
        const toggle = document.createElement('button');
        toggle.className = 'theme-toggle';
        toggle.setAttribute('aria-label', 'Toggle theme');
        toggle.addEventListener('click', toggleTheme);
        document.body.appendChild(toggle);

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
