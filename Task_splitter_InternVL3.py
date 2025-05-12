import torch
from transformers import AutoProcessor, AutoModel, BitsAndBytesConfig
from PIL import Image

model_path = "/sda1/InternVL3-14B"

# 1. 모델 로드 (8bit + bfloat16)
bnb_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="cuda:1",
    torch_dtype=torch.bfloat16,
    quantization_config=BitsAndBytesConfig(load_in_8bit=True)
).eval()

# 2. 프로세서 로드
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

# 3. 이미지 입력
image_paths = [
    "/home/sylee/codes/Data_generation_for_robots/image/task_1/init/top_Color.png",
    "/home/sylee/codes/Data_generation_for_robots/image/task_1/init/side_Color.png"
]
images = [Image.open(p).convert("RGB") for p in image_paths]

# 4. 프롬프트 입력
text_prompt = "Describe the differences between the two images."

# 5. 입력 처리 및 추론
inputs = processor(text=text_prompt, images=images, return_tensors="pt").to(model.device)

with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=128)

# 6. 출력
output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("Generated Answer:\n", output)