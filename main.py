from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pickle
import os

app = FastAPI()

# Load model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)


# Home page
@app.get("/", response_class=HTMLResponse)
def home():
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")

    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


# Student input
class StudentData(BaseModel):
    study_hours: float
    attendance: float
    previous_score: float
    assignments_completed: int
    sleep_hours: float


# Prediction
@app.post("/predict")
def predict(data: StudentData):

    input_data = [[
        data.study_hours,
        data.attendance,
        data.previous_score,
        data.assignments_completed,
        data.sleep_hours
    ]]
 
    prediction = model.predict(input_data)[0]

    return {
        "predicted_score": round(float(prediction), 2)
    }