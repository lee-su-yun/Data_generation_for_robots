
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
    device_map="auto",
    torch_dtype="auto",
    local_files_only=True
)
"""
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

messages = [
    {
        "role": "user",
        "content": (
            "I have a robot arm with a simple gripper, operating on a flat white table. "
            "Assume the robot has standard capabilities for visual perception, object localization, and basic grasping. "
            "It can manipulate common household objects such as colored plastic cups, tissues, paper cups, and a permanent marker. "
            "Please suggest 100 **realistic and practical tabletop manipulation tasks** suitable for fine-tuning a foundation model for 1-armed household robots. "
            "These tasks should be feasible with current robot hardware and relevant to everyday human activities. "
            "Each task should require meaningful **physical interaction** and test at least one useful skill such as planning, visual perception, sequential action, or tool use. "
            "Avoid tasks that involve deformable object modeling or complex multi-step assembly that is not suitable for a single arm robot. "
            "Use the following format exactly:\n\n"
            "task_1: <short clear description>\n"
            "required_objects: <list of objects>\n"
            "initial_setup: <clear initial condition of objects on the table>\n"
            "difficulty: <easy / medium / hard>\n"
        )
    }
]

messages = [
    {
        "role": "user",
        "content": (
            "I have a robot arm with a simple gripper, operating on a flat white table. "
            "The robot has standard capabilities for visual perception, object localization, and basic grasping. "
            #"It can manipulate common household objects such as colored plastic cups, tissues, paper cups, and a permanent marker. "
            "It can manipulate common household objects such as 6 colored plastic cups (e.g., light blue, white, pink, lavender, mint, yellow), and three white paper cups which have numbers (1, 2, 3). "
            "The robot can lift and place objects by using its gripper.\n\n"

            "Please suggest 50 **realistic and reasoning-focused tabletop manipulation tasks** suitable for fine-tuning a foundation model for 1-armed household robots. "
            "Each task should be physically feasible with current robot hardware and involve **meaningful physical interaction**. "
            "However, more importantly, each task must also require **reasoning, decision-making, or planning** by the robot. "
            "That means the robot must determine the correct course of action based on object properties (like color, position, or type), task constraints, or implicit rules. "
            "For example, tasks like selecting the right item based on a condition (e.g., color, materials), prioritizing actions, or avoiding specific objects. "
            "Tasks that only involve repetition or precision placement without reasoning should be avoided.\n\n"
            
            "Avoid using exact measurements such as '10 cm from the edge'. Instead, describe object locations using **simple relative spatial terms** like 'on the left side of the table', 'in the center', 'next to the blue cup'.\n\n"
           # "Avoid tasks that require manipulation of deformable objects, complex tool use, or multi-step mechanical assembly.\n\n"
            "For each task, you must describe the initial and final state of the tabletop **in sufficient detail that a human could visualize or reconstruct the setup**. "
            "That includes positions, orientations, relative positions (e.g. right, left, above, upright), and any relevant spatial arrangements. Mention specific colors, object states (e.g. upright, upside down), and where they are located on the table.\n\n"
            
            "**Important: You must generate all 50 tasks in full. Do not skip, summarize, or say that the remaining tasks follow the same pattern. "
            "Every task must be explicitly written out in the required format.**\n\n"
            
            "Use the following format exactly:\n\n"
            "task_1: <short clear description>\n"
            "required_objects: <list of objects>\n"
            "initial_setup: <clear, detailed initial condition of the tabletop>\n"
            "final_state: <clear, detailed expected tabletop state after the task is complete>\n"
            "difficulty: <easy / medium / hard>\n"
        )
    }
]
"""

messages = [
    {
        "role": "user",
        "content": (
            "I have a robot arm with a simple gripper, operating on a flat white table. "
            "The robot has standard capabilities for visual perception, object localization, and basic grasping. "
            "It can manipulate 3 sets of 6 colored plastic cups (e.g., blue, white, pink, purple, green, yellow). "  # , and three white paper cups which have numbers (1, 2, 3). "
            "The robot can lift and place objects by using its gripper.\n\n"

            "Please suggest 30 practical tabletop manipulation tasks based on these colored cups, suitable for fine-tuning a foundation model for 1-armed robots. "
            "Each task should involve physical interaction and test useful robotic skills such as planning, perception, categorization, or spatial reasoning. "
            "Tasks should encourage the robot to perform operations like sorting, stacking, grouping, or pattern-based placement. "
            "Respond in JSON format with fields: description, required_objects, and initial_setup. [/INST]"
            "{\n"
            "  \"task_1\": {\n"
            "    \"description\": \"...\",\n"
            "    \"required_objects\": \"...\",\n"
            "    \"initial_setup\": \"...\"\n"
            "  },\n"
            "  \"task_2\": {\n"
            "    \"description\": \"...\",\n"
            "    \"required_objects\": \"...\",\n"
            "    \"initial_setup\": \"...\"\n"
            "  },\n"
            "  ...\n"
            "}\n\n"
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
with open("/home/sylee/codes/Data_generation_for_robots/task/reason50.txt", "w") as f:
    f.write(thinking_content)

# content
parsed = parse_task_blocks(content)

with open("/home/sylee/codes/Data_generation_for_robots/task/tasks50.json", "w", encoding="utf-8") as f:
    json.dump(parsed, f, indent=2, ensure_ascii=False)

print("Saved to suggested_tasks.json")
