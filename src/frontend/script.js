let chart;

async function getForecast() {

    const city = document.getElementById("citySelect").value;

    const response = await fetch(`http://127.0.0.1:8000/predict?city=${city}`);
    const data = await response.json();

    const forecast = data["7_day_forecast_with_uncertainty"];

    const predictions = forecast.map(x => x.prediction);
    const lower = forecast.map(x => x.lower_95);
    const upper = forecast.map(x => x.upper_95);
    const risks = forecast.map(x => x.health_risk);

    document.getElementById("riskContainer").innerHTML =
        "<strong>Health Risk Levels:</strong><br>" + risks.join(" → ");

    const ctx = document.getElementById("forecastChart").getContext("2d");

    if (chart) chart.destroy();

    chart = new Chart(ctx, {
        type: "line",
        data: {
            labels: ["Day1","Day2","Day3","Day4","Day5","Day6","Day7"],
            datasets: [
                {
                    label: "Prediction",
                    data: predictions,
                    borderColor: "blue",
                    fill: false
                },
                {
                    label: "Lower 95%",
                    data: lower,
                    borderColor: "green",
                    borderDash: [5,5],
                    fill: false
                },
                {
                    label: "Upper 95%",
                    data: upper,
                    borderColor: "red",
                    borderDash: [5,5],
                    fill: false
                }
            ]
        }
    });
}