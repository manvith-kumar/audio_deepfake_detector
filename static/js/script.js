document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('upload-form');
    const analyzeButton = document.getElementById('analyze-button');
    const fileInput = document.getElementById('audio-file-input');
    const resultContainer = document.getElementById('result-container');
    const predictionDiv = document.getElementById('prediction');
    const confidenceDiv = document.getElementById('confidence');
    const spinner = document.getElementById('loading-spinner');
    const errorMessageDiv = document.getElementById('error-message');

    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        resultContainer.classList.add('hidden');
        errorMessageDiv.classList.add('hidden');
        spinner.classList.remove('hidden');
        analyzeButton.disabled = true;
        analyzeButton.textContent = 'Analyzing...';

        const formData = new FormData();
        formData.append('audio_file', fileInput.files[0]);

        try {
            const response = await fetch('/predict', { method: 'POST', body: formData });
            const result = await response.json();
            if (response.ok) {
                displayResult(result);
            } else {
                displayError(result.error || 'An unknown error occurred.');
            }
        } catch (error) {
            displayError('Failed to connect to the server. Please try again.');
        } finally {
            spinner.classList.add('hidden');
            analyzeButton.disabled = false;
            analyzeButton.textContent = 'Analyze';
        }
    });

    function displayResult(data) {
        const { prediction, confidence } = data;
        predictionDiv.textContent = prediction;
        predictionDiv.className = prediction.toLowerCase();
        confidenceDiv.textContent = `Confidence: ${(confidence * 100).toFixed(2)}%`;
        resultContainer.classList.remove('hidden');
    }

    function displayError(message) {
        errorMessageDiv.textContent = `Error: ${message}`;
        errorMessageDiv.classList.remove('hidden');
    }
});