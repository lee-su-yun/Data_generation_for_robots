import os
import json
import asyncio
with open('/home/sylee/codes/Data_generation_for_robots/splitted_task/50tasks.json', 'r', encoding='utf-8') as f:
    all_tasks = json.load(f)
from typing import List, Optional
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info  # Qwen2.5-VL util

class RobotTaskPlanner:
    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device

    async def generate_plan(self, task: str, images: List[str], system_prompt: Optional[str] = None) -> str:
        # Prepare message
        content = []
        for image in images:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": task})

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]

        # Prepare model input
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.device)

        # Inference
        generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]

    def parse_generated_text(self, text: str) -> dict:
        result = {
            "task": "",
            "description": "",
            "plan": "",
            "planning_reason": "",
            "subtasks": []
        }
        current = {}
        lines = text.split('\n')

        def commit():
            if current and "step" in current:
                result["subtasks"].append(current.copy())

        current_field = None

        for line in lines:
            line = line.strip()
            if line.startswith("### Task:"):
                current_field = "task"
                result["task"] = ""
            elif line.startswith("### Description:"):
                current_field = "description"
                result["description"] = ""
            elif line.startswith("### Plan:"):
                current_field = "plan"
                result["plan"] = ""
            elif line.startswith("### Reasoning:"):
                current_field = "planning_reason"
                result["planning_reason"] = ""
            elif line.startswith("[Step"):
                commit()
                current_field = None
                current = {"step": line.strip("[]")}
            elif line.strip().upper() == "FINISHED":
                commit()
                break
            elif line.startswith("<SUBTASK>:"):
                current["subtask"] = line[len("<SUBTASK>:"):].strip()
            elif line.startswith("<SUBTASK_reason>:"):
                current["subtask_reason"] = line[len("<SUBTASK_reason>:"):].strip()
            elif line.startswith("<MOVE>:"):
                current["move"] = line[len("<MOVE>:"):].strip()
            elif line.startswith("<MOVE_reason>:"):
                current["move_reason"] = line[len("<MOVE_reason>:"):].strip()
            elif line.startswith("<ISSUE>:"):
                current["issue"] = line[len("<ISSUE>:"):].strip()
            elif line.startswith("<SOLUTION>:"):
                current["solution"] = line[len("<SOLUTION>:"):].strip()
            else:
                if current_field and current_field in result:
                    result[current_field] += (line + " ")

        return result

async def create_robot_plan_and_save(
    model, processor, device,
    task: str,
    image_paths: List[str],
    image_ids: List[int],
    output_json_path: str
):
    planner = RobotTaskPlanner(model, processor, device)

    system_prompt = (
        "You are tasked with inferring and annotating a robot arm trajectory given only the initial images, final images, and a task description. "
        "You must reason through the missing sequence of actions required to transition from the initial state to the final state. Your output should include high-level planning and per-step detailed annotations with justifications.\n\n"
        "# Experimental Setup:\n"
        "- You are an expert in robotics and reinforcement learning.\n"
        "- A robot arm must complete a manipulation task using the information from:\n"
        "  - 3 initial images:\n"
        "    - 2 ego-view images (top view, side view)\n"
        "    - 1 exo-view image (wrist camera)\n"
        "  - 3 final images: same format as above\n"
        "  - A natural language task instruction\n"
        "- You must infer the trajectory of steps and actions the robot would take to solve the task.\n\n"
        "# Instructions:\n\n"
        "## 1. Start by Describing the Task:\n"
        "Give a comprehensive description of the task using the given instruction and the initial/final images. Include:\n"
        "- Object identities and positions\n"
        "- Relative spatial relationships\n"
        "- Any potential occlusions, stackings, or flipped objects\n"
        "- A high-level sequence of subtasks needed to complete the task\n"
        "Then provide a full high-level plan (e.g., \"Pick cup 3\", \"Move cup 3 to right\", etc.), and assign each plan to a step interval (e.g., step 1: Pick cup 3, step 2: Move cup 3 to right). "
        "Include a brief explanation of the reasoning behind the plan structure based on spatial relationships or task constraints.\n\n"
        "## 2. For Each Step, Write the Following:\n"
        "Each action step must include:\n"
        "[Step X]\n"
        "<SUBTASK> The high-level subtask that should be executed now.\n"
        "<SUBTASK_reason> Why this subtask should be executed now, referencing spatial cues in the current setup.\n"
        "<MOVE> Primitive movement to execute this subtask (e.g., “move down and close gripper”).\n"
        "<MOVE_reason> Why this movement is necessary at this moment.\n"
        "<ISSUE> A potential issue that may arise while performing this step.\n"
        "<SOLUTION> A brief reasoning or method to avoid or correct the issue.\n\n"
        "## 3. After All Steps Are Annotated:\n"
        "End the output with:\n"
        "FINISHED\n\n"
        "## 4. Use the following format clearly.\n"
        "<TASK>:~~\n"
        "<DESCRIPTION>:~~\n"
        "<PLAN>:~~\n"
        "<PLANNING_reason>:~~\n"
        "[Step X]\n"
        "<SUBTASK>:~~\n"
        "<SUBTASK_reason>:~~\n"
        "<MOVE>:~~\n"
        "<MOVE_reason>:~~\n"
        "<ISSUE>:~~\n"
        "<SOLUTION>:~~\n\n"
        "IMPORTANT: You MUST use the exact format shown above, with angle brackets and colons.\n"
        "Do NOT use Markdown headers like '### Task' or '### Plan'. That is strictly forbidden.\n"
        "If you do not follow the format, the output will be discarded and considered invalid.\n\n"
        "# Notes:\n"
        "- Be descriptive and natural in language, but maintain a clear logical structure.\n"
        "- You are working purely from visual state comparison and task description, without access to in-between frames.\n"
        "- Your goal is to generate a plausible and logically sound sequence that could have transformed the initial to the final state.\n"
        "- The output should simulate what an expert robot policy planner would do.\n\n"
        "# Scene Description:\n"
        "The robot arm is in an environment with the following objects:\n"
        "- Multiple colored plastic cups (light blue, white, pink, lavender, mint, yellow).\n"
        "- Three white paper cups labeled with numbers 1, 2, and 3.\n"
        "These cups may be upright, flipped, or stacked.\n"
        "All objects are on a white table.\n"
        "The robot arm is black and has a silver camera attached to its gripper, and images may come from:\n"
        "- Ego-top view (top-down),\n"
        "- Ego-side view (angled),\n"
        "- Exo-wrist view (camera on the gripper).\n\n"
        "# Image Convention:\n"
        "- You will receive 6 images in total:\n"
        "  - The first 3 images represent the initial state of the environment:\n"
        "    - Ego-top view, Ego-side view, Exo-wrist view (from the gripper)\n"
        "  - The last 3 images represent the final state after the task is completed:\n"
        "    - Ego-top view, Ego-side view, Exo-wrist view\n"
        "Use these images, along with the task description, to infer the full action plan and annotate the steps."
    )

    # 1. Generate response
    generated_text = await planner.generate_plan(task, image_paths, system_prompt)

    print(generated_text)

    # 2. Parse into structured format
    parsed = planner.parse_generated_text(generated_text)

    # 3. Assemble final output
    output = {
        "task": parsed["task"],
        "description": parsed["description"],
        "images_used": image_paths,
        "plan": parsed["plan"],
        "planning_reason": parsed["planning_reason"],
        "subtasks": parsed["subtasks"]
    }


    # 4. Save JSON
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    print(f"Saved robot plan to {output_json_path}")

if __name__ == "__main__":
    # Device setup
    model_path = "/sda1/hub/Qwen2.5-VL-7B-Instruct"
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(device)
    #quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        #torch_dtype="auto",
        attn_implementation="flash_attention_2",
        device_map=device
        #quantization_config=quantization_config
    ).eval().to(device)

    processor = AutoProcessor.from_pretrained(model_path)

    loop = asyncio.get_event_loop()
    for i in range(3, 5):
        task_id = f"task_{i}"
        task = all_tasks[task_id]["description"]

        image_paths = [
            f"/home/sylee/codes/Data_generation_for_robots/image/{task_id}/init/top_Color.png",
            f"/home/sylee/codes/Data_generation_for_robots/image/{task_id}/init/side_Color.png",
            f"/home/sylee/codes/Data_generation_for_robots/image/{task_id}/init/wrist_Color.png",
            f"/home/sylee/codes/Data_generation_for_robots/image/{task_id}/final/top_Color.png",
            f"/home/sylee/codes/Data_generation_for_robots/image/{task_id}/final/side_Color.png",
            f"/home/sylee/codes/Data_generation_for_robots/image/{task_id}/final/wrist_Color.png"
        ]

        output_json_path = f"/home/sylee/codes/Data_generation_for_robots/splitted_task/{task_id}.json"

        loop.run_until_complete(
            create_robot_plan_and_save(
                model,
                processor,
                device,
                task,
                image_paths,
                [1, 2, 3],
                output_json_path
            )
        )