import torch
from transformers import AutoProcessor, AutoModel, AutoTokenizer
from PIL import Image

# 1. 모델 경로와 로드
model_path = '/sda1/InternVL3-14B'
model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    load_in_8bit=True,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    use_flash_attn=True,
    device_map='cuda:1'  # 자동 분산
).eval()

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

pixel_values1 = load_image('/home/sylee/codes/Data_generation_for_robots/image/task_1/init/top_Color.png', max_num=12).to(torch.bfloat16).cuda()
pixel_values2 = load_image('/home/sylee/codes/Data_generation_for_robots/image/task_1/final/top_Color.png', max_num=12).to(torch.bfloat16).cuda()
pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)


# 3. 이미지 텍스트 명령 구성
question = '<image>\nDescribe the two images in detail.'


# 5. 생성 config
generation_config = dict(max_new_tokens=1024, do_sample=True)

response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                               history=None, return_history=True)
print(f'User: {question}\nAssistant: {response}')