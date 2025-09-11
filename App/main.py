from fastapi import FastAPI, File, UploadFile
import pandas as pd
import re, time, os
from pydantic import BaseModel
from . import LinearRegression
import joblib


class AnalyzeDataset(BaseModel):
    file_path:str
    split_amount:float
    visualization:bool

    first_row_header:bool
class ValidateModel(BaseModel):
    x_value:int
    model_path:str
MAX_FILE_SIZE = 1 * 1024 * 1024  # MAX 1MB file size
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(root_path="/linear")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # ==================
    # Checking File Size
    # ==================
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        return {"message": "Maximum 1MB file size is allowed!"}

    # Check file extension
    if not file.filename.lower().endswith(".csv"):
        return {"message": "Only .csv files are allowed to upload!"}
    
    # Read CSV from contents, not file.file
    try:
        from io import StringIO
        df = pd.read_csv(StringIO(contents.decode("utf-8")))
    except:
        return {"message": "Invalid .csv file!"}
    
    # Check columns count
    if len(df.columns) != 2:
        return {"message": "Upload .csv with only 1 feature and 1 target!"}

    # Check numeric values
    try:
        df = df.astype(float)
    except:
        return {"message": "Your dataset contains non-numerical values!"}

    # Generate unique safe filename
    clean_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', file.filename.lower())
    unique_name = f"{int(time.time())}_{clean_name}"
    file_path = os.path.join(UPLOAD_DIR, unique_name)

    # Save the file
    with open(file_path, "wb") as myfile:
        myfile.write(contents)

    return {
        "message": "CSV uploaded successfully!",
        "file_path": file_path
    }

@app.post("/predict")
def get_response(file:AnalyzeDataset):
    file_path = file.file_path
    split_amount = file.split_amount
    first_row_header = file.first_row_header
    visualize = file.visualization
    if os.path.exists(file_path):

        x,y = LinearRegression.read_csv(file_path,first_row_header)
        x_train,y_train,x_val,y_val = LinearRegression.data_split(x,y,split_amount)
        model = LinearRegression.SimpleLinearRegression()
        model.fit(x_train,y_train)
        model.predict(x_val)
        metrics = model.evaluate(x_val,y_val,"Validation Set")
        png_path = file_path.removesuffix(".csv") + ".png"
        if visualize:
            model.visualize(x_val,y_val,png_path)



        model_path = png_path.removesuffix(".png") + ".joblib"
        
        equation = model.equation()
        joblib.dump(model,model_path)
        results = {"equation":equation,
                   "metrics":metrics
                   ,
                   "file_path":png_path,
                   "model_path": model_path}
        return results
    
    else:
        return
@app.post("/validate")
def validate_x(data:ValidateModel):
    x_value  = float(data.x_value)
    model_path = data.model_path
    loaded_model = joblib.load(model_path)
    y = round(loaded_model.validate(x_value),2)
    return{"message":"Successfully predicted!"
        ,"y":y}
