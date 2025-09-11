from fastapi import FastAPI, File, UploadFile
import pandas as pd
import re, time, os

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

@app.get("/predict")
def get_response():
    return "Analyzing dataset..."
