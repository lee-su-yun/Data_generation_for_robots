
import re
import json
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_task_blocks(text: str) -> dict:
    tasks = {}
    current_task = None
    fields = {}

    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("task_"):
            if current_task and fields:
                tasks[current_task] = fields
            parts = line.split(":", 1)
            current_task = parts[0].strip()
            description = parts[1].strip()
            fields = {"description": description}
        elif ":" in line:
            key, val = line.split(":", 1)
            fields[key.strip()] = val.strip()

    if current_task and fields:
        tasks[current_task] = fields

    return tasks


model_path = "/sda1/Qwen3-8B"

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="cuda:0",
    torch_dtype="auto",
    local_files_only=True
)

messages = [
    {
        "role": "user",
        "content": (
            "I have a robot arm with a gripper, operating on a white table. "
            "Assume the robot can interact with any general household objects, such as colored cups, tissue, paper cups, or a permanent marker. "
            "Please suggest 2 practical tabletop manipulation tasks suitable for fine-tuning a foundation model for robotics. "
            "Each task should involve physical interaction and test useful robotic skills such as planning, perception, or tool use. "
            "Use the following format clearly:\n\n"
            "task_1: <description>\n"
            "required_objects: <objects>\n"
            "initial_setup: <~~>\n"
            "difficulty: <easy/medium/hard>\n"
        )
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

outputs = model.generate(
    **model_inputs,
    max_new_tokens=32768,
)

output_ids = outputs[0][len(model_inputs.input_ids[0]):].tolist()

try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()

# thinking_content
formatted_text = thinking_content.replace("\\n", "\n")
with open("/home/sylee/codes/Data_generation_for_robots/thinking_task.json", "w", encoding="utf-8") as f:
    f.write(formatted_text)

# content
parsed = parse_task_blocks(content)

with open("/home/sylee/codes/Data_generation_for_robots/suggested_task.json", "w", encoding="utf-8") as f:
    json.dump(parsed, f, indent=2, ensure_ascii=False)

print("Saved to suggested_tasks.json")
