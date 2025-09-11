from fastapi import FastAPI, File, UploadFile
import pandas as pd
import re, time, os
from pydantic import BaseModel
from . import LinearRegression
import joblib
from fastapi.staticfiles import StaticFiles

class AnalyzeDataset(BaseModel):
    file_path: str
    split_amount: float
    visualization: bool
    first_row_header: bool

class ValidateModel(BaseModel):
    x_value: int
    model_path: str

MAX_FILE_SIZE = 1 * 1024 * 1024  # MAX 1MB file size
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(root_path="/linear")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            return {"error": "Maximum 1MB file size is allowed!"}

        if not file.filename.lower().endswith(".csv"):
            return {"error": "Only .csv files are allowed to upload!"}

        from io import StringIO
        df = pd.read_csv(StringIO(contents.decode("utf-8")))

        if len(df.columns) != 2:
            return {"error": "Upload .csv with only 1 feature and 1 target!"}

        df = df.astype(float)

        clean_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', file.filename.lower())
        unique_name = f"{int(time.time())}_{clean_name}"
        file_path = os.path.join(UPLOAD_DIR, unique_name)

        with open(file_path, "wb") as myfile:
            myfile.write(contents)

        return {"message": "CSV uploaded successfully!", "file_path": file_path}

    except Exception as e:
        return {"error": f"File upload failed: {str(e)}"}


@app.post("/predict")
def get_response(file: AnalyzeDataset):
    try:
        file_path = file.file_path
        split_amount = file.split_amount
        first_row_header = file.first_row_header
        visualize = file.visualization

        if not os.path.exists(file_path):
            return {"error": "File not found!"}

        x, y = LinearRegression.read_csv(file_path, first_row_header)
        x_train, y_train, x_val, y_val = LinearRegression.data_split(x, y, split_amount)

        model = LinearRegression.SimpleLinearRegression()
        model.fit(x_train, y_train)
        model.predict(x_val)
        metrics = model.evaluate(x_val, y_val, "Validation Set")

        png_path = file_path.removesuffix(".csv") + ".png"
        if visualize:
            model.visualize(x_val, y_val, png_path)

        model_path = png_path.removesuffix(".png") + ".joblib"
        equation = model.equation()
        joblib.dump(model, model_path)

        return {
            "equation": equation,
            "metrics": metrics,
            "file_path": png_path,
            "model_path": model_path
        }

    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}


@app.post("/validate")
def validate_x(data: ValidateModel):
    try:
        x_value = float(data.x_value)
        model_path = data.model_path

        if not os.path.exists(model_path):
            return {"error": "Model file not found!"}

        loaded_model = joblib.load(model_path)
        y = round(loaded_model.validate(x_value), 2)

        return {"message": "Successfully predicted!", "y": y}

    except Exception as e:
        return {"error": f"Validation failed: {str(e)}"}
