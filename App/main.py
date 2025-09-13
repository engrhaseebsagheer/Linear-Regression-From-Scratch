from fastapi import FastAPI, File, UploadFile
import pandas as pd
import re, time, os
from pydantic import BaseModel
from . import LinearRegression
import joblib
from fastapi.staticfiles import StaticFiles

# ====== BASE DIR for Production ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ====== Pydantic Models ======
class AnalyzeDataset(BaseModel):
    file_path: str
    split_amount: float
    visualization: bool
    first_row_header: bool

class ValidateModel(BaseModel):
    x_value: int
    model_path: str

MAX_FILE_SIZE = 1 * 1024 * 1024  # MAX 1MB file size

# ====== FastAPI App ======
app = FastAPI(root_path="/linear")
app.mount(
    "/uploads",  # URL path
    StaticFiles(directory=UPLOAD_DIR),  # real folder
    name="linear-uploads"
)

# Helper function to create public URL
def get_public_url(filename: str):
    return f"/linear/uploads/{filename}"

# ====== Upload Endpoint ======
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

        # Return public URL instead of disk path
        return {"message": "CSV uploaded successfully!", "file_path": get_public_url(unique_name)}

    except Exception as e:
        return {"error": f"File upload failed: {str(e)}"}

# ====== Predict Endpoint ======
@app.post("/predict")
def get_response(file: AnalyzeDataset):
    try:
        file_path = file.file_path
        split_amount = file.split_amount
        first_row_header = file.first_row_header
        visualize = file.visualization

        # Convert public URL back to real path for processing
        filename = file_path.split("/")[-1]
        real_path = os.path.join(UPLOAD_DIR, filename)

        if not os.path.exists(real_path):
            return {"error": "File not found!"}

        x, y = LinearRegression.read_csv(real_path, first_row_header)
        x_train, y_train, x_val, y_val = LinearRegression.data_split(x, y, split_amount)

        model = LinearRegression.SimpleLinearRegression()
        model.fit(x_train, y_train)
        model.predict(x_val)
        metrics = model.evaluate(x_val, y_val, "Validation Set")

        png_path = real_path.removesuffix(".csv") + ".png"
        if visualize:
            model.visualize(x_val, y_val, png_path)

        model_path = png_path.removesuffix(".png") + ".joblib"
        equation = model.equation()
        joblib.dump(model, model_path)

        # Return public URLs instead of disk paths
        return {
            "equation": equation,
            "metrics": metrics,
            "file_path": get_public_url(os.path.basename(png_path)),
            "model_path": get_public_url(os.path.basename(model_path))
        }

    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

# ====== Validate Endpoint ======
@app.post("/validate")
def validate_x(data: ValidateModel):
    try:
        x_value = float(data.x_value)

        # Convert public URL back to real path
        filename = data.model_path.split("/")[-1]
        model_real_path = os.path.join(UPLOAD_DIR, filename)

        if not os.path.exists(model_real_path):
            return {"error": "Model file not found!"}

        loaded_model = joblib.load(model_real_path)
        y = round(loaded_model.validate(x_value), 2)

        return {"message": "Successfully predicted!", "y": y}

    except Exception as e:
        return {"error": f"Validation failed: {str(e)}"}
