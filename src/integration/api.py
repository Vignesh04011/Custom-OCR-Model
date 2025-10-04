from fastapi import FastAPI, UploadFile
from src.main import process_pdf

app = FastAPI()

@app.post("/infer")
async def infer(file: UploadFile):
    output = process_pdf(file.file)
    return {"results": output}
