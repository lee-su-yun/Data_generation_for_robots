import json
import os
from tqdm import tqdm
import torch
from diffusers import StableDiffusion3Pipeline
"""
pipe = StableDiffusion3Pipeline.from_pretrained("/sda1/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda:1")

image = pipe(
    "A capybara holding a sign that reads Hello World",
    num_inference_steps=28,
    guidance_scale=3.5,
).images[0]
image.save("capybara.png")
"""

pipe = StableDiffusion3Pipeline.from_pretrained(
    "/sda1/stable-diffusion-3.5-large",
    torch_dtype=torch.bfloat16
).to("cuda:1")

with open("tasks.json", "r") as f:
    tasks = json.load(f)


##########
def make_prompt(state_text, viewpoint):
    return (
        f"A tabletop robotic manipulation scene showing a one-arm robot with a gripper, "
        f"executing a task on a white table. {state_text}. "
        f"The camera angle is from a top-down view at a slight tilt, capturing the robot and the entire table clearly. "
        f"The environment resembles a lab setting. {viewpoint}"
    )

def generate_and_save_image(prompt, save_path):
    image = pipe(prompt, num_inference_steps=28, guidance_scale=3.5).images[0]
    image.save(save_path)
##########


# 경로 생성 및 이미지 저장
for task_key in tqdm(tasks.keys()):
    task = tasks[task_key]
    folder = f"./{task_key}"
    os.makedirs(folder, exist_ok=True)

    # 프롬프트 생성
    init_prompt = make_prompt(task["initial_setup"], "This is the initial setup.")
    final_prompt = make_prompt(task["final_state"], "This is the final state after the task.")

    # 이미지 생성 및 저장
    generate_and_save_image(init_prompt, os.path.join(folder, "init.png"))
    generate_and_save_image(final_prompt, os.path.join(folder, "final.png"))

    # task 내용 저장
    with open(os.path.join(folder, "task.json"), "w") as f:
        json.dump({task_key: task}, f, indent=2)