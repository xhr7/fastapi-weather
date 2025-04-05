from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
import joblib


# define input schema
class ModelInput(BaseModel):
    temperature: float  # Temperature in Celsius
    weather: Literal["rainy", "clear", "cloudy"]  # Categorical weather condition
    weekday: int  # Integer representing the day of the week (0=Sunday, 6=Saturday)


# initialize FastAPI
app = FastAPI()

# load model and encoder
print("Model loading...")

model = joblib.load("model.pkl")
print("Model loaded.")

print("Encoder loading...")

weather_encoder = joblib.load("weather_encoder.pkl")
print("Encoder loaded.")


# define prediction endpoint
@app.post("/predict")
def predict(data: ModelInput):
    # encode categorical input
    encoded_weather = weather_encoder.transform([[data.weather]]).toarray()[0]

    # create feature vector
    features = [data.temperature, *encoded_weather, data.weekday]

    # generate prediction
    prediction = model.predict([features])

    return {"prediction": prediction.tolist()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("solved:app", host="0.0.0.0", port=8000, reload=True)
