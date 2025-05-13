import json
import torch
from transformers import AutoProcessor, MllamaForConditionalGeneration

import requests
from PIL import Image

model_path = "/sda1/llama3.2-11b-vision"
image_path = "/home/sylee/codes/Data_generation_for_robots/task/stack_cups1.jpg"

image = Image.open(image_path)


processor = AutoProcessor.from_pretrained(model_path)
model = MllamaForConditionalGeneration.from_pretrained(
    model_path,
    #attn_implementation="flex_attention",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

"""
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "I have a robot arm with a gripper, operating on a white table. "
                    "Assume the robot can interact with any general household objects, such as colored cups, tissue, paper cups, or a permanent marker. "
                    "Please suggest 30 practical tabletop manipulation tasks suitable for fine-tuning a foundation model for robotics. "
                    "Each task should involve physical interaction and test useful robotic skills such as planning, perception, or tool use. "
                    "Respond in the following JSON format:\n\n"
                    "{\n"
                    "  \"task_1\": {\n"
                    "    \"description\": \"~~\",\n"
                    "    \"required_objects\": \"~~\",\n"
                    "    \"initial_setup\": \"~~\",\n"
                    "  },\n"
                    "  \"task_2\": { ... },\n"
                    "  ...\n"
                    "}\n\n"
                    "Make sure all fields are filled. Output only the JSON structure, with no extra commentary."
                )
            }
        ]
    }
]
"""
prompt = (
    "<|image|>"
    "<|begin_of_text|>"
    "Ignore the image."
    "[INST] I have a robot arm with a gripper, operating on a white table. "
    "Assume the robot can interact with plastic cups in the following colors: blue, white, pink, purple, light green, and yellow. "
    "There are 3 sets of these colored plastic cups randomly placed on the table (i.e., 18 cups total). "
    "Please suggest 30 practical tabletop manipulation tasks based on these colored cups. "
    "Each task should involve physical interaction and test useful robotic skills such as planning, perception, categorization, or spatial reasoning. "
    "Tasks should encourage the robot to perform operations like sorting, stacking, grouping, or pattern-based placement. "
    "Respond in JSON format with fields: description, required_objects, initial_setup, and difficulty. [/INST]"
    "{\n"
    "  \"task_1\": {\n"
    "    \"description\": \"...\",\n"
    "    \"required_objects\": \"...\",\n"
    "    \"initial_setup\": \"...\",\n"
    "    \"difficulty\": \"easy / medium / hard\"\n"
    "  },\n"
    "  \"task_2\": {\n"
    "    \"description\": \"...\",\n"
    "    \"required_objects\": \"...\",\n"
    "    \"initial_setup\": \"...\",\n"
    "    \"difficulty\": \"...\"\n"
    "  },\n"
    "  ...\n"
    "}\n\n"
)

inputs = processor(image, prompt, return_tensors="pt").to(model.device)

"""
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)
"""
outputs = model.generate(
    **inputs,
    max_new_tokens=512,
)

response_text = processor.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])[0]

try:
    response_json = json.loads(response_text)
except json.JSONDecodeError:
    print("JSON parsing failed. Raw output:")
    print(response_text)
    raise

with open("/home/sylee/codes/Data_generation_for_robots/30tasks.json", "w") as f:
    json.dump(response_json, f, indent=2)

print("Saved to suggested_tasks.json")