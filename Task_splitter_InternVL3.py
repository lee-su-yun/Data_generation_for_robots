import torch
from transformers import AutoProcessor, AutoModel, AutoTokenizer, BitsAndBytesConfig
from PIL import Image

# 1. 모델 경로와 로드
model_path = '/sda1/InternVL3-14B'
model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map='cuda:1'  # 자동 분산
).eval()

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

# 2. 이미지 두 장 불러오기
img_paths = [
    '/home/sylee/codes/Data_generation_for_robots/image/task_1/init/top_Color.png',
    '/home/sylee/codes/Data_generation_for_robots/image/task_1/final/top_Color.png'
]
images = [Image.open(p).convert("RGB") for p in img_paths]

# 3. 이미지 텍스트 명령 구성
question = 'Image-1: <image>\nImage-2: <image>\nDescribe the similarities and differences between these two images.'

# 4. 이미지 전처리
pixel_values_list = []
num_patches_list = []
for img in images:
    tiles = processor.preprocess(img, max_num=12, image_size=448, use_thumbnail=True)
    pixel_values = tiles["pixel_values"].to(torch.bfloat16).cuda()
    pixel_values_list.append(pixel_values)
    num_patches_list.append(pixel_values.size(0))

pixel_values = torch.cat(pixel_values_list, dim=0)

# 5. 생성 config
generation_config = dict(max_new_tokens=512, do_sample=False)

# 6. 대화 수행
response, _ = model.chat(
    tokenizer,
    pixel_values=pixel_values,
    query=question,
    generation_config=generation_config,
    num_patches_list=num_patches_list,
    return_history=True
)

print(f"User: {question}\nAssistant: {response}")