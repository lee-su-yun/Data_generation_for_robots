import torch
from transformers import AutoProcessor, AutoModelForVision2Text, BitsAndBytesConfig
from PIL import Image

model_path = "/sda1/InternVL3-14B"

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True
)

model = AutoModelForVision2Text.from_pretrained(
    model_id,
    device_map="cuda:1",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    trust_remote_code=True
).eval()

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

image_paths = [
    "/home/sylee/codes/Data_generation_for_robots/image/task_1/init/top_Color.png",
    "/home/sylee/codes/Data_generation_for_robots/image/task_1/init/side_Color.png"
]
images = [Image.open(p).convert("RGB") for p in image_paths]

text_prompt = "Describe the differences between the two images."

inputs = processor(text=text_prompt, images=images, return_tensors="pt").to(model.device)

with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=128)

output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("Generated Answer:\n", output)
