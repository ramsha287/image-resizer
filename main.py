from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import io

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/resize/")
async def resize_image(file: UploadFile = File(...), size: int = Form(...)):
    # Read the uploaded file as a NumPy array
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

    # Resize the image
    new_width = int(image.shape[1] * size / 100)
    new_height = int(image.shape[0] * size / 100)
    resized_image = cv2.resize(image, (new_width, new_height))

    # Encode the resized image as a JPEG file
    _, buffer = cv2.imencode(".jpg", resized_image)
    file_stream = io.BytesIO(buffer)

    # Return the resized image as a downloadable file
    return StreamingResponse(file_stream, media_type="image/jpeg", headers={
        "Content-Disposition": "attachment; filename=resized-image.jpg"
    })
