// Configuration - loaded from config.js (generated during CDK deployment)
// Variables API_ENDPOINT, WARMUP_ENDPOINT, CLASSIFICATION_MODELS, and ENDPOINT_NAMES are defined in config.js

console.log('Config loaded:', {
    API_ENDPOINT,
    WARMUP_ENDPOINT,
    CLASSIFICATION_MODELS,
    ENDPOINT_NAMES
});

// Warmup endpoints on page load
function warmupEndpoints() {
    // Fire-and-forget async request to warmup endpoint
    // Don't wait for response or handle errors - we want this to be non-blocking
    fetch(WARMUP_ENDPOINT, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
    }).catch(() => {
        // Silently ignore errors - warmup is a best-effort optimization
        console.log('Warmup request sent (response not awaited)');
    });
}

// Call warmup when page loads
warmupEndpoints();

// Get DOM elements
const form = document.getElementById('explanation-form');
const inputText = document.getElementById('input-text');
const modelSelect = document.getElementById('model-select');
const submitBtn = document.getElementById('submit-btn');
const errorMessage = document.getElementById('error-message');
const loadingMessage = document.getElementById('loading-message');
const results = document.getElementById('results');

// Form submission handler
form.addEventListener('submit', async (e) => {
    e.preventDefault();

    // Clear previous results and errors
    errorMessage.style.display = 'none';
    results.style.display = 'none';
    results.innerHTML = '';

    const text = inputText.value.trim();
    const model = modelSelect.value;

    // Validate input
    if (text.length === 0) {
        showError('Please enter some text.');
        return;
    }

    if (text.length > 100) {
        showError('Your input is too long. Your input must be fewer than 100 characters.');
        return;
    }

    // Check for HTML injection
    const htmlPattern = /<[^>]*>/;
    if (htmlPattern.test(text)) {
        showError('Your input was detected as potentially containing HTML. To avoid HTML injections, this field does not accept any input that is wrapped in HTML tags: <>');
        return;
    }

    // Show loading state
    loadingMessage.style.display = 'block';
    submitBtn.disabled = true;

    try {
        const modelConfig = CLASSIFICATION_MODELS[model];
        const classificationEndpoint = ENDPOINT_NAMES[`${model.toUpperCase()}_CLASSIFICATION_MODEL`];

        // Build the generic Lambda request format
        const requestBody = {
            endpoint_name: classificationEndpoint,
            payload: {
                data: text,
                explain: true
            }
        };

        console.log('Making API request to:', API_ENDPOINT);
        console.log('Request body:', requestBody);

        // Call the API
        const response = await fetch(API_ENDPOINT, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody)
        });

        console.log('Response status:', response.status);
        console.log('Response ok:', response.ok);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log('Response data:', data);

        // Check if the Lambda invocation was successful
        if (!data.success) {
            throw new Error(data.error || 'Unknown error from model endpoint');
        }

        // Display results (data.result contains the SageMaker endpoint response)
        displayResults(data.result, model);
    } catch (error) {
        console.error('Error details:', error);
        console.error('Error stack:', error.stack);
        showError('An error occurred while generating explanations. Please try again later.');
    } finally {
        loadingMessage.style.display = 'none';
        submitBtn.disabled = false;
    }
});

// Show error message
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
}

// Create colored explanation HTML
function createHexCss(explanation, desiredLabel, undesiredLabel) {
    const greens = ['d6e6d5', 'e1ffe0', 'c8ffc7', 'a8faa7', '90ff8f', '6bff69', '40ff3d', '07fc03'];
    const reds = ['e6d5d5', 'facfcf', 'ffbaba', 'ffa1a1', 'ff8585', 'ff6666', 'ff3d3d', 'ff1919'];
    let outputStr = '';

    for (const tokenExplanation of explanation) {
        const token = tokenExplanation[0];
        const shapleyValues = tokenExplanation[1];

        // Collapse shapley values to simplify coloring
        const collapsedValue = shapleyValues[desiredLabel] - shapleyValues[undesiredLabel];
        const colorIndex = Math.min(Math.floor(Math.abs(collapsedValue) * 20), 7);

        let color;
        if (collapsedValue > 0) {
            color = greens[colorIndex];
        } else if (collapsedValue < 0) {
            color = reds[colorIndex];
        } else {
            color = 'fcfcfc';
        }

        outputStr += `<mark style="background-color: #${color};\">${escapeHtml(token)}</mark>`;
    }

    return outputStr;
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Display results
function displayResults(endpointResponse, modelType) {
    const model = CLASSIFICATION_MODELS[modelType];
    const desiredLabel = model.desired_label;
    const undesiredLabel = model.undesired_label;

    // Get prediction info from the SageMaker endpoint response
    const prediction = endpointResponse.prediction;
    let predictedLabel = '';
    let predictedScore = 0;

    for (const labelInfo of prediction) {
        if (labelInfo.score > predictedScore) {
            predictedScore = labelInfo.score;
            predictedLabel = labelInfo.label;
        }
    }

    let html = `
        <h3>Prediction</h3>
        <div class="prediction-box">
            <p><strong>Predicted Class:</strong> <span class="prediction-label">${escapeHtml(predictedLabel)}</span></p>
            <p><strong>Confidence:</strong> ${(predictedScore * 100).toFixed(1)}%</p>
        </div>
    `;

    // Display explanation
    if (endpointResponse.explanation) {
        const explanationHtml = createHexCss(
            endpointResponse.explanation,
            desiredLabel,
            undesiredLabel
        );

        html += `
            <h3>Explanation</h3>
            <div class="explanation">
                ${explanationHtml}
            </div>
            <p style="text-align: center; font-size: 0.9rem; color: #666; margin-top: 1rem;">
                (green indicates a contribution towards ${desiredLabel},
                red indicates a contribution towards ${undesiredLabel})
            </p>
        `;
    }

    results.innerHTML = html;
    results.style.display = 'block';
}
