import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import uvicorn
from ImageCaptioning.pipeline.prediction import PredictionPipeline

# Bypass NLTK import security check for the prediction app
os.environ["NLTK_DISABLE_IMPORT_SECURITY"] = "1"

app = FastAPI()

os.makedirs("uploads", exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <html>
        <head>
            <title>Image Captioning</title>
            <style>
                body { font-family: Arial, sans-serif; padding: 50px; background-color: #f4f4f9; }
                .container { max-width: 500px; margin: auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
                input[type="file"] { margin: 20px 0; }
                input[type="submit"] { background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
                input[type="submit"]:hover { background-color: #45a049; }
            </style>
        </head>
        <body>
            <div class="container">
                <h2>Upload an Image for Captioning</h2>
                <form action="/predict" enctype="multipart/form-data" method="post">
                    <input name="file" type="file" required>
                    <br>
                    <input type="submit" value="Generate Caption">
                </form>
            </div>
        </body>
    </html>
    """

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_path = os.path.join("uploads", file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
        
    pipeline = PredictionPipeline(file_path)
    caption = pipeline.predict()
    
    return {"filename": file.filename, "generated_caption": caption}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
