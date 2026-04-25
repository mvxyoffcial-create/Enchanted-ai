import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from diffusers import StableDiffusionXLControlNetImg2ImgPipeline, EulerDiscreteScheduler
from PIL import Image
import io
import uvicorn

app = FastAPI()

# Allow your frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the fastest 4-step model available in 2026
MODEL_ID = "ByteDance/SDXL-Lightning-4step"
pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, variant="fp16"
).to("cuda")

# Speed optimization for Euler scheduler
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

@app.post("/upscale")
async def upscale_image(file: UploadFile = File(...)):
    # 1. Load and Resize
    content = await file.read()
    init_image = Image.open(io.BytesIO(content)).convert("RGB").resize((1024, 1024))
    
    # 2. Run AI Repair (4 steps for lightning speed)
    with torch.inference_mode():
        result = pipe(
            prompt="ultra high definition, 4k, sharp details, photorealistic, cinematic",
            image=init_image,
            num_inference_steps=4,
            guidance_scale=0,
            strength=0.35 # High enough to add detail, low enough to keep the original look
        ).images[0]
    
    # 3. Export as high-quality JPEG
    img_byte_arr = io.BytesIO()
    result.save(img_byte_arr, format='JPEG', quality=90)
    return img_byte_arr.getvalue()

if __name__ == "__main__":
    # Host on Port 8080 for Koyeb
    uvicorn.run(app, host="0.0.0.0", port=8080)
          
