let chart;
let map;
let marker;
let ws;

const cityCoords = {
    Delhi: [28.6139, 77.2090],
    Mumbai: [19.0760, 72.8777],
    Bengaluru: [12.9716, 77.5946],
    Chennai: [13.0827, 80.2707],
    Hyderabad: [17.3850, 78.4867]
};

// ================= INITIALIZE =================

window.onload = function () {
    initMap();
    initChart();
    initLiveStream();
};


// ================= SEARCH HANDLER =================

function handleSearch() {
    const city = document.getElementById("cityInput").value.trim();

    if (!cityCoords[city]) {
        alert("City not supported.");
        return;
    }

    getForecast(city);
}


// ================= MAP =================

function initMap() {
    map = L.map('map').setView([22.5937, 78.9629], 5);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors'
    }).addTo(map);

    marker = L.circleMarker([22.5937, 78.9629], {
        radius: 12,
        fillColor: "#888",
        color: "#000",
        weight: 1,
        fillOpacity: 0.9
    }).addTo(map);
}

function updateMap(city, value) {
    const coords = cityCoords[city];
    if (!coords) return;

    marker.setLatLng(coords);

    marker.setStyle({
        fillColor: getAQIColor(value)
    });

    marker.bindPopup(`${city}<br>AQI: ${value}`).openPopup();
    pulseMarker();
}


// ================= CHART =================

function initChart() {
    const ctx = document.getElementById("forecastChart").getContext("2d");

    chart = new Chart(ctx, {
        type: "line",
        data: {
            labels: ["Day1","Day2","Day3","Day4","Day5","Day6","Day7"],
            datasets: [
                { label: "Prediction", data: [], borderWidth: 2, tension: 0.4 },
                { label: "Lower 95%", data: [], borderDash: [5,5] },
                { label: "Upper 95%", data: [], borderDash: [5,5] }
            ]
        },
        options: {
            responsive: true,
            animation: false
        }
    });
}

function updateChart(predictions, lower, upper) {
    if (!chart) return;

    chart.data.datasets[0].data = predictions;
    chart.data.datasets[1].data = lower;
    chart.data.datasets[2].data = upper;
    chart.update();
}


// ================= FORECAST (REST) =================

async function getForecast(city) {

    try {
        showLoader(true);

        const response = await fetch(`/predict?city=${city}`);
        if (!response.ok) throw new Error("Prediction failed");

        const data = await response.json();
        const forecast = data["7_day_forecast_with_uncertainty"];

        if (!forecast) return;

        const predictions = forecast.map(x => x.prediction);
        const lower = forecast.map(x => x.lower_95);
        const upper = forecast.map(x => x.upper_95);
        const risks = forecast.map(x => x.health_risk);

        updateChart(predictions, lower, upper);
        renderRisks(risks);
        updateMap(city, predictions[0]);

    } catch (error) {
        console.error(error);
    } finally {
        showLoader(false);
    }
}


// ================= RISK BADGES =================

function renderRisks(risks) {

    const container = document.getElementById("riskContainer");
    container.innerHTML = "<strong>Health Risk:</strong><br>";

    risks.forEach(risk => {
        const badge = document.createElement("span");
        badge.className = "risk-badge";
        badge.innerText = risk;
        badge.style.background = getAQIColorFromLabel(risk);
        container.appendChild(badge);
    });
}


// ================= LIVE STREAM =================

function initLiveStream() {

    const protocol = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${protocol}://${location.host}/ws/live`);

    ws.onmessage = function(event) {

        const liveData = JSON.parse(event.data);
        const selectedCity = document.getElementById("cityInput").value;

        if (!liveData[selectedCity]) return;

        const aqi = liveData[selectedCity].aqi;

        updateMap(selectedCity, aqi);
    };

    ws.onclose = function() {
        setTimeout(initLiveStream, 5000);
    };
}


// ================= LOADER =================

function showLoader(show) {
    const loader = document.getElementById("loader");
    if (!loader) return;
    loader.classList.toggle("hidden", !show);
}


// ================= ANIMATION =================

function pulseMarker() {
    marker.setStyle({ radius: 16 });
    setTimeout(() => marker.setStyle({ radius: 12 }), 300);
}


// ================= COLOR HELPERS =================

function getAQIColor(aqi) {
    if (aqi <= 50) return "#00e400";
    if (aqi <= 100) return "#9cff00";
    if (aqi <= 200) return "#ffcc00";
    if (aqi <= 300) return "#ff6600";
    if (aqi <= 400) return "#ff0000";
    return "#990000";
}

function getAQIColorFromLabel(label) {
    const mapping = {
        "Good": "#00e400",
        "Satisfactory": "#9cff00",
        "Moderate": "#ffcc00",
        "Poor": "#ff6600",
        "Very Poor": "#ff0000",
        "Severe": "#990000"
    };
    return mapping[label] || "#888";
}


// ================= THEME =================

function toggleTheme() {
    const body = document.body;

    if (body.classList.contains("dark")) {
        body.classList.remove("dark");
        body.classList.add("light");
        localStorage.setItem("theme", "light");
    } else {
        body.classList.remove("light");
        body.classList.add("dark");
        localStorage.setItem("theme", "dark");
    }
}

window.addEventListener("DOMContentLoaded", () => {
    const savedTheme = localStorage.getItem("theme");
    if (savedTheme) document.body.classList.add(savedTheme);
});