/* -----------------------------------------------
/* How to use? : Check the GitHub README
/* ----------------------------------------------- */

/* To load a config file (particles.json) you need to host this demo (MAMP/WAMP/local)... */

// particlesJS.load('particles-js', 'particles.json', function() {
//   console.log('particles.js loaded - callback');
// });


// /* Otherwise just put the config content (json): */

// Wrap initialization to ensure CSS is fully loaded
function initParticles() {
    // Determine particle speed based on current page and accessibility preferences
    const isLandingPage = document.body.classList.contains('landing-page');
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const particleSpeed = prefersReducedMotion ? 0 : (isLandingPage ? 2 : 0.5);

    // Get particle color from CSS variable - force style recalculation
    const root = document.documentElement;
    // Trigger a forced reflow to ensure CSS variables are computed
    root.offsetHeight;
    const computedStyles = window.getComputedStyle(root);
    let particleColor = computedStyles.getPropertyValue('--particle-color').trim();

    // Fallback: if color is empty or invalid, determine from theme
    if (!particleColor || particleColor === '') {
        const isDark = root.getAttribute('data-theme') === 'dark';
        particleColor = isDark ? '#ffffff' : '#000000';
    }

    particlesJS('particles-js', {
        "particles": {
            "number": {
                "value": 80,
                "density": {
                    "enable": true,
                    "value_area": 800
                }
            },
            "color": {
                "value": particleColor || "#000000"
            },
            "shape": {
                "type": "circle",
                "stroke": {
                    "width": 0,
                    "color": particleColor
                },
                "polygon": {
                    "nb_sides": 5
                },
                "image": {
                    "src": "img/github.svg",
                    "width": 100,
                    "height": 100
                }
            },
            "opacity": {
                "value": 0.5,
                "random": false,
                "anim": {
                    "enable": false,
                    "speed": 1,
                    "opacity_min": 0.1,
                    "sync": false
                }
            },
            "size": {
                "value": 3,
                "random": true,
                "anim": {
                    "enable": false,
                    "speed": 40,
                    "size_min": 0.1,
                    "sync": false
                }
            },
            "line_linked": {
                "enable": true,
                "distance": 150,
                "color": particleColor,
                "opacity": 0.4,
                "width": 1
            },
            "move": {
                "enable": true,
                "speed": particleSpeed,
                "direction": "none",
                "random": false,
                "straight": false,
                "out_mode": "out",
                "bounce": false,
                "attract": {
                    "enable": false,
                    "rotateX": 600,
                    "rotateY": 1200
                }
            }
        },
        "interactivity": {
            "detect_on": "canvas",
            "events": {
                "onhover": {
                    "enable": true,
                    "mode": "grab"
                },
                "onclick": {
                    "enable": true,
                    "mode": "push"
                },
                "resize": true
            },
            "modes": {
                "grab": {
                    "distance": 179.82017982017982,
                    "line_linked": {
                        "opacity": 0.7041854516904497
                    }
                },
                "bubble": {
                    "distance": 400,
                    "size": 40,
                    "duration": 2,
                    "opacity": 8,
                    "speed": 3
                },
                "repulse": {
                    "distance": 200,
                    "duration": 0.4
                },
                "push": {
                    "particles_nb": 4
                },
                "remove": {
                    "particles_nb": 2
                }
            }
        },
        "retina_detect": true
    });

    // After particles are initialized, update colors based on current theme
    // Use a small delay to ensure particles are fully loaded
    setTimeout(function() {
        if (window.updateThemeParticles) {
            window.updateThemeParticles();
        }
    }, 100);
}

// Initialize particles after DOM is ready AND styles are computed
// Use double requestAnimationFrame to ensure styles are fully applied
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function() {
        requestAnimationFrame(function() {
            requestAnimationFrame(initParticles);
        });
    });
} else {
    // DOM already ready
    requestAnimationFrame(function() {
        requestAnimationFrame(initParticles);
    });
}
