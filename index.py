from transformers import MobileViTImageProcessor, MobileViTForImageClassification
import torch
from PIL import Image

model = MobileViTForImageClassification.from_pretrained("apple/mobilevit-small")
image_processor = MobileViTImageProcessor.from_pretrained("apple/mobilevit-small")


image = Image.open("car.png").convert("RGB")

inputs = image_processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()

predicted_class = model.config.id2label[predicted_class_idx]
print(f"Predicted class: {predicted_class}")

