
import re
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

def clean_llm_json_output(raw_text: str, output_path: str):
    parts = re.split(r'}\s*"?\s*"?\s*{', raw_text)
    json_chunks = []
    for i, part in enumerate(parts):
        if i == 0:
            json_chunks.append(part + "}")
        elif i == len(parts) - 1:
            json_chunks.append("{" + part)
        else:
            json_chunks.append("{" + part + "}")

    merged_tasks = {}
    task_idx = 1

    for chunk in json_chunks:
        try:
            parsed = json.loads(chunk)
            for key, val in parsed.items():
                merged_tasks[f"task_{task_idx}"] = val
                task_idx += 1
        except json.JSONDecodeError as e:
            print(f"Skipping malformed chunk:\n{chunk}\nError: {e}")
            continue

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged_tasks, f, indent=2, ensure_ascii=False)

    print(f"Clean JSON saved to {output_path}")



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

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")


# thinking_content
try:
    response_json = json.loads(content)
except json.JSONDecodeError:
    print("JSON parsing failed. Raw output:")
    print(thinking_content)

with open("/home/sylee/codes/Data_generation_for_robots/suggested_thinking.json", "w") as f:
    json.dump(content, f, indent=2)

# content
clean_llm_json_output(content, "/home/sylee/codes/Data_generation_for_robots/suggested_tasks.json")
"""
cleaned_content = extract_and_merge_json_objects(content)

with open("/home/sylee/codes/Data_generation_for_robots/suggested_tasks.json", "w") as f:
    json.dump(cleaned_content, f, indent=2)


print("Saved to suggested_tasks.json")
"""
