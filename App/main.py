from fastapi import FastAPI, UploadFile, File, HTTPException
import os
import LinearModel.LinearRegression as LinearRegression
upload_directory = "uploads"
os.makedirs(upload_directory,exist_ok=True)


app = FastAPI()
@app.post("/upload")
async def upload_file(file:UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    allowed_exts = [".csv", ".xls", ".xlsx"]
    if ext not in allowed_exts:
        raise HTTPException(status_code=400, detail="Only CSV or Excel files allowed")
    

    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as myfile:
        myfile.write(await file.read())
    return {"message": "File uploaded successfully", "file": file.filename}



@app.get("/")
def home():
    return {"message":"Hello World!"}