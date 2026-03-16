from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

# choose device
device = "cuda"

# load model
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
model.to(device)
model.eval()

# load image
image = Image.open("test5.jpg").convert("RGB")

# preprocess image
pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

# inference
with torch.no_grad():
    generated_ids = model.generate(pixel_values)

# decode
text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("Recognized Text:", text)