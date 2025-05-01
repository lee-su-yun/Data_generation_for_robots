
import json
from transformers import AutoModelForCausalLM, AutoTokenizer


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
        "content": [
            {
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
            }
        ]
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
    response_json = json.loads(output_ids)
except json.JSONDecodeError:
    print("JSON parsing failed. Raw output:")
    print(output_ids)
    raise

with open("/home/sylee/codes/Data_generation_for_robots/suggested_tasks.json", "w") as f:
    json.dump(output_ids, f, indent=2)

print("Saved to suggested_tasks.json")