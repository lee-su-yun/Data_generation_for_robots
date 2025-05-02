import json
import os
from tqdm import tqdm
import torch
from diffusers import StableDiffusion3Pipeline
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel

model_path = "/sda1/stable-diffusion-3.5-large"

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model_nf4 = SD3Transformer2DModel.from_pretrained(
    model_path,
    subfolder="transformer",
    quantization_config=nf4_config,
    torch_dtype=torch.bfloat16
)

pipeline = StableDiffusion3Pipeline.from_pretrained(
    model_path,
    transformer=model_nf4,
    torch_dtype=torch.bfloat16
)
pipeline.enable_model_cpu_offload()


def make_prompt(state_text):
    return (
        "On the table, Wide top-down view."
        f"{state_text}."
    )

def generate_and_save_image(pipeline, prompt, save_path):
    image = pipeline(
        prompt=prompt,
        num_inference_steps=28,
        guidance_scale=4.5,
        max_sequence_length=512,
    ).images[0]
    image.save(save_path)




pipe = StableDiffusion3Pipeline.from_pretrained(
    "/sda1/stable-diffusion-3.5-large",
    torch_dtype=torch.bfloat16
).to("cuda:1")

with open("tasks1.json", "r") as f:
    tasks = json.load(f)


# 경로 생성 및 이미지 저장
for task_key in tqdm(tasks.keys()):
    task = tasks[task_key]
    folder = f"./{task_key}"
    os.makedirs(folder, exist_ok=True)

    # 프롬프트 생성
    init_prompt = make_prompt(task["initial_setup"])
    final_prompt = make_prompt(task["final_state"])

    # 이미지 생성 및 저장
    generate_and_save_image(pipeline, init_prompt, os.path.join(folder, "init.png"))
    generate_and_save_image(pipeline, final_prompt, os.path.join(folder, "final.png"))

    # task 내용 저장
    with open(os.path.join(folder, "task.json"), "w") as f:
        json.dump({task_key: task}, f, indent=2)

    exit()