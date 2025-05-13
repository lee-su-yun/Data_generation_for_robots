import json
import torch
from transformers import AutoProcessor, MllamaForConditionalGeneration
from PIL import Image

model_path = "/sda1/llama3.2-11b-vision"
image_path = "/home/sylee/codes/Data_generation_for_robots/image/side_Color.png"
output_path = "/home/sylee/codes/Data_generation_for_robots/30tasks.json"

image = Image.open(image_path).convert("RGB")

processor = AutoProcessor.from_pretrained(model_path)
model = MllamaForConditionalGeneration.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

prompt = (
    "<|image|>"
    "<|begin_of_text|>"
    "[INST] The image shows a side view of a robot arm operating on a white table. "
    "Assume the robot can interact with plastic cups in the following colors: blue, white, pink, purple, light green, and yellow. "
    "There are 3 sets of these colored plastic cups randomly placed on the table (i.e., 18 cups total). "
    "Please suggest 30 practical tabletop manipulation tasks based on these colored cups. "
    "Each task should involve physical interaction and test useful robotic skills such as planning, perception, categorization, or spatial reasoning. "
    "Tasks should encourage the robot to perform operations like sorting, stacking, grouping, or pattern-based placement. "
    "Respond only in JSON format with 30 tasks. Each should include fields: description, required_objects, and initial_setup. "
    "Do not repeat the prompt. Do not include any explanation or commentary. Just return the final JSON object with 30 tasks. [/INST]"
    # "Respond in JSON format with fields: description, required_objects, and initial_setup. [/INST]"
    # "{\n"
    # "  \"task_1\": {\n"
    # "    \"description\": \"...\",\n"
    # "    \"required_objects\": \"...\",\n"
    # "    \"initial_setup\": \"...\"\n"
    # "  },\n"
    # "  \"task_2\": {\n"
    # "    \"description\": \"...\",\n"
    # "    \"required_objects\": \"...\",\n"
    # "    \"initial_setup\": \"...\"\n"
    # "  },\n"
    # "  ...\n"
    # "}\n\n"
)

inputs = processor(image, prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=1024,
)

response_text = processor.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])[0]
print("=== RAW OUTPUT ===")
print(response_text[:1000])

try:
    response_json = json.loads(response_text)
except json.JSONDecodeError:
    print("JSON parsing failed. Raw output:")
    print(response_text)
    raise

# 파일 저장
with open(output_path, "w") as f:
    json.dump(response_json, f, indent=2)

print(f"Saved to {output_path}")