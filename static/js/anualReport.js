document.getElementById('loadDataButton').addEventListener('click', () => {
    const fileInput = document.getElementById('csvFileInput');
    const file = fileInput.files[0];
    
    if (file) {
        const reader = new FileReader();
        
        reader.onload = (event) => {
            const data = event.target.result;
            const parsedData = parseCSV(data);
            renderChart(parsedData);
            displayTopNGOs(parsedData);
        };
        
        reader.readAsText(file);
    } else {
        alert('Please select a CSV file first.');
    }
});

function parseCSV(data) {
    const rows = data.split('\n').slice(1); // Skip header
    const ngos = [];
    const providedFood = [];
    const receivedFood = [];

    rows.forEach(row => {
        const cols = row.split(',');
        if (cols.length === 3) {
            ngos.push(cols[0]); // NGO name
            providedFood.push(parseFloat(cols[1])); // Food provided
            receivedFood.push(parseFloat(cols[2])); // Food received
        }
    });

    return { ngos, providedFood, receivedFood };
}

function renderChart(data) {
    const ctx = document.getElementById('reportChart').getContext('2d');
    const reportChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.ngos,
            datasets: [
                {
                    label: 'Food Provided (kg)',
                    data: data.providedFood,
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    fill: true,
                    tension: 0.3 // Smooth lines
                },
                {
                    label: 'Food Received (kg)',
                    data: data.receivedFood,
                    borderColor: 'rgba(255, 99, 132, 1)',
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    fill: true,
                    tension: 0.3 // Smooth lines
                }
            ]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

function displayTopNGOs(data) {
    const topNGOs = data.ngos
        .map((ngo, index) => ({
            name: ngo,
            provided: data.providedFood[index]
        }))
        .sort((a, b) => b.provided - a.provided)
        .slice(0, 3); // Get top 3 NGOs

    const topNGOsDiv = document.getElementById('topNGOs');
    topNGOsDiv.innerHTML = '<h3>Top NGOs:</h3>' + 
        topNGOs.map(ngo => `<p>${ngo.name}: ${ngo.provided} kg provided</p>`).join('');
}
