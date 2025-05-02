import json
import os
from tqdm import tqdm
import torch
from diffusers import StableDiffusion3Pipeline
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from compel import Compel

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

#compel = Compel(tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder)
#prompt = "A whimsical and creative image depicting a hybrid creature that is a mix of a waffle and a hippopotamus, basking in a river of melted butter amidst a breakfast-themed landscape. It features the distinctive, bulky body shape of a hippo. However, instead of the usual grey skin, the creature's body resembles a golden-brown, crispy waffle fresh off the griddle. The skin is textured with the familiar grid pattern of a waffle, each square filled with a glistening sheen of syrup. The environment combines the natural habitat of a hippo with elements of a breakfast table setting, a river of warm, melted butter, with oversized utensils or plates peeking out from the lush, pancake-like foliage in the background, a towering pepper mill standing in for a tree.  As the sun rises in this fantastical world, it casts a warm, buttery glow over the scene. The creature, content in its butter river, lets out a yawn. Nearby, a flock of birds take flight"

prompt = (
    #"A robotic manipulation scene showing a one-arm robot with a parallel gripper on a white table,"
    #"The scene is viewed from a top-down angle, showing the robot and table clearly,"
    "The red tissue is crumpled and lying flat on the center of the table. The blue plastic cup is upright on the left side, 20 cm from the edge. The green paper cup is upright on the right side, 20 cm from the edge. A yellow tissue is folded and placed near the red tissue."
    #"The red plastic cup is upright on the left side of the table, 20 cm from the edge. The blue paper cup is on the right side, 20 cm from the edge. A folded white tissue lies at the table center, 10 cm from the red cup."
)

image = pipeline(
    prompt=prompt,
    num_inference_steps=28,
    guidance_scale=4.5,
    max_sequence_length=512,
).images[0]

image.save("whimsical.png")


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

"""