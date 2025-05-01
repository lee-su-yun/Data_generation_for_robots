import json
import torch
from transformers import AutoProcessor, MllamaForConditionalGeneration

model_path = "/sda1/llama3.2-11b-vision"
processor = AutoProcessor.from_pretrained(model_path)
model = MllamaForConditionalGeneration.from_pretrained(
    model_path,
    #attn_implementation="flex_attention",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "I have a robot arm with a gripper, operating on a white table. "
                    "Assume the robot can interact with any general household objects, such as colored cups, tissue, paper cups, or a permanent marker. "
                    "Please suggest 5 practical tabletop manipulation tasks suitable for fine-tuning a foundation model for robotics. "
                    "Each task should involve physical interaction and test useful robotic skills such as planning, perception, or tool use. "
                    "Respond in the following JSON format:\n\n"
                    "{\n"
                    "  \"task_1\": {\n"
                    "    \"description\": \"~~\",\n"
                    "    \"required_objects\": \"~~\",\n"
                    "    \"initial_setup\": \"~~\",\n"
                    "    \"difficulty\": \"easy / medium / hard\"\n"
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

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

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

with open("/home/sylee/codes/Data_generation_for_robots/suggested_tasks.json", "w") as f:
    json.dump(response_json, f, indent=2)

print("Saved to suggested_tasks.json")