from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import joblib
from typing import Literal

app = FastAPI()

model = joblib.load("model.pkl")
weather_encoder = joblib.load("weather_encoder.pkl")

@app.get("/", response_class=HTMLResponse)
def form():
    html_content = """
    <html>
        <head>
            <title>Weather Prediction</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #f0f4f8;
                    padding: 40px;
                }
                .container {
                    background-color: #fff;
                    border-radius: 12px;
                    padding: 30px;
                    max-width: 500px;
                    margin: auto;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                }
                h2 {
                    text-align: center;
                    color: #333;
                }
                label {
                    font-weight: bold;
                    display: block;
                    margin-top: 15px;
                }
                input, select {
                    width: 100%;
                    padding: 10px;
                    border-radius: 6px;
                    border: 1px solid #ccc;
                    margin-top: 5px;
                }
                input[type=submit] {
                    margin-top: 20px;
                    background-color: #007BFF;
                    color: white;
                    border: none;
                    cursor: pointer;
                    transition: 0.3s;
                }
                input[type=submit]:hover {
                    background-color: #0056b3;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h2>Weather Prediction</h2>
                <form action="/predict" method="post">
                    <label>Temperature (°C):</label>
                    <input type="number" step="0.1" name="temperature" required>

                    <label>Weather:</label>
                    <select name="weather">
                        <option value="rainy">Rainy</option>
                        <option value="clear">Clear</option>
                        <option value="cloudy">Cloudy</option>
                    </select>

                    <label>Weekday (0=Sunday, 6=Saturday):</label>
                    <input type="number" name="weekday" min="0" max="6" required>

                    <input type="submit" value="Predict">
                </form>
            </div>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/predict", response_class=HTMLResponse)
def predict(temperature: float = Form(...), weather: Literal["rainy", "clear", "cloudy"] = Form(...), weekday: int = Form(...)):
    encoded_weather = weather_encoder.transform([[weather]]).toarray()[0]
    features = [temperature, *encoded_weather, weekday]
    prediction = model.predict([features])[0]

    html_response = f"""
    <html>
        <head>
            <title>Prediction Result</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background-color: #f0f4f8;
                    padding: 40px;
                }}
                .container {{
                    background-color: #fff;
                    border-radius: 12px;
                    padding: 30px;
                    max-width: 500px;
                    margin: auto;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                    text-align: center;
                }}
                h2 {{
                    color: #28a745;
                }}
                p {{
                    font-size: 18px;
                    margin: 10px 0;
                }}
                a {{
                    display: inline-block;
                    margin-top: 20px;
                    text-decoration: none;
                    color: white;
                    background-color: #007BFF;
                    padding: 10px 20px;
                    border-radius: 6px;
                }}
                a:hover {{
                    background-color: #0056b3;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h2>Prediction Result</h2>
                <p><b>Temperature:</b> {temperature} °C</p>
                <p><b>Weather:</b> {weather}</p>
                <p><b>Weekday:</b> {weekday}</p>
                <h3> Predicted Value: <span style='color:blue'>{round(prediction, 2)}</span></h3>
                <a href="/">Try Again</a>
            </div>
        </body>
    </html>
    """
    return HTMLResponse(content=html_response)