document.addEventListener('DOMContentLoaded', function() {
    const form = document.querySelector('.input-section');
    const submitBtn = document.getElementById('submit-btn');
    const inputText = document.getElementById('input-text');
    const resultsSection = document.querySelector('.results-section');

    submitBtn.addEventListener('click', function(e) {
        e.preventDefault();
        
        const text = inputText.value.trim();
        if (!text) {
            alert('Please enter some text to analyze');
            return;
        }

        // Disable button and show loading state
        submitBtn.disabled = true;
        submitBtn.textContent = 'Analyzing...';

        // Make the API call
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                // Show results section
                resultsSection.style.display = 'block';
                
                // Update prediction
                const predictionText = document.getElementById('prediction-text');
                predictionText.textContent = data.prediction_label;
                predictionText.style.color = data.prediction ? '#dc3545' : '#28a745';
                
                // Update confidence
                document.getElementById('confidence-score').textContent = 
                    `Confidence: ${(data.confidence * 100).toFixed(2)}%`;
                
                // Update processed text
                document.getElementById('processed-text-content').textContent = 
                    data.processed_text;
            } else {
                alert('Error: ' + (data.message || 'Unknown error occurred'));
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Error analyzing text. Please try again.');
        })
        .finally(() => {
            // Re-enable button
            submitBtn.disabled = false;
            submitBtn.textContent = 'Analyze Text';
        });
    });
});