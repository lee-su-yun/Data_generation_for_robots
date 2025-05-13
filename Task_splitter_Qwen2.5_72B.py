from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from transformers import BitsAndBytesConfig

model_path = "/sda1/Qwen2.5-VL-72B-Instruct"
#model_path = "/sda1/hub/Qwen2.5-VL-7B-Instruct"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,              # 또는 load_in_8bit=True
    bnb_4bit_compute_dtype=torch.bfloat16,  # 또는 float16
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",      # fp4보다 일반적임
)
# default: Load the model on the available device(s)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="flash_attention_2",
    # llm_int8_enable_fp32_cpu_offload=True,
)


# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     model_path,
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained(model_path)

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

image_paths = [
    "/home/sylee/codes/Data_generation_for_robots/image/task_1/init/top_Color.png",
    "/home/sylee/codes/Data_generation_for_robots/image/task_1/init/side_Color.png",
    "/home/sylee/codes/Data_generation_for_robots/image/task_1/init/wrist_Color.png",
    "/home/sylee/codes/Data_generation_for_robots/image/task_1/final/top_Color.png"
    "/home/sylee/codes/Data_generation_for_robots/image/task_1/final/side_Color.png",
    "/home/sylee/codes/Data_generation_for_robots/image/task_1/final/wrist_Color.png"
]

task = "Move the pink plastic cup to the center behind of the table without knocking over any other cups."

system_prompt = (
    "You are tasked with inferring and annotating a robot arm trajectory given only the initial images, final images, and a task description. "
    "You must reason through the missing sequence of *fine-grained* actions required to transition from the initial state to the final state. "
    "Your output must include high-level planning and per-step *primitive-level* annotations with justifications.\n\n"

    "# Experimental Setup:\n"
    "- You are an expert in robotics and reinforcement learning.\n"
    "- A robot arm must complete a tabletop manipulation task using:\n"
    "  - 3 initial images:\n"
    "    - Ego-top view, Ego-side view, Exo-wrist view\n"
    "  - 3 final images: same format\n"
    "  - A natural language task instruction\n"
    "- You must infer and annotate the *minimal, sequential motor-level subtasks* to complete the task.\n\n"

    "# Instructions:\n\n"
    "## 1. Describe the Task:\n"
    "- Briefly explain the overall goal using the instruction and the visual difference between initial and final states.\n"
    "- Mention:\n"
    "  - Object identities and colors\n"
    "  - Initial and final positions\n"

    "## 2. Provide Detailed Planning:\n"
    "<PLAN>:\n"
    "- Decompose the entire task into a sequence of *subtasks*, where each subtask represents a meaningful action unit (e.g., moving one cup).\n"
    "- For each <SUBTASK>, provide the corresponding <MOVE> list that contains fine-grained, robot arm-level primitive actions.\n"
    "- Format:\n"
    "    <SUBTASK>: [brief natural language description of the objective, e.g., 'Move blue cup to bottom-right corner']\n"
    "    <MOVE>: ['Raise arm', 'Move arm left', 'Move arm forward', 'Open gripper', 'Lower arm', 'Close gripper (grab blue cup)', 'Raise arm', 'Move arm right', 'Move arm down', 'Lower arm', 'Open gripper (release cup)']\n"

    "- Number the subtasks (e.g., Subtask 1, Subtask 2, ...) and make sure each <MOVE> is specific, step-by-step, and physical.\n"

    "<PLANNING_reason>:\n"
    "- Explain the reasoning behind the **ordering of subtasks** (not individual moves).\n"
    "- Mention factors such as: minimizing collision, avoiding occlusion, simplifying alignment, or optimizing travel distance.\n"
    "- Example: 'Starting with the blue cup minimizes the risk of disturbing other nearby cups and clears space for the red cup placement.'\n"

    "## 4. Finish with:\n"
    "FINISHED\n\n"

    "# Notes:\n"
    "- DO NOT skip primitive steps. Decompose compound actions (e.g., 'sort all cups') into multiple atomic actions.\n"
    "- Do not assume any hidden motion. You only know the initial and final images.\n"
    "- All reasoning must be *visually grounded* and task-specific.\n"
    "- Write like an expert roboticist explaining every decision.\n\n"

    "# Scene Description:\n"
    "Objects include:\n"
    "- Plastic cups in various colors (blue, red, pink, lavender, mint, yellow)\n"
    "- White paper cups labeled 1, 2, 3\n"
    "- Cups may be upright, flipped, or stacked on each other\n"
    "All items are on a white table.\n"
    "The robot arm is black with a silver wrist camera.\n\n"

    "# Image Format:\n"
    "- 6 total images:\n"
    "  - First 3: Initial state (top, side, wrist view)\n"
    "  - Last 3: Final state (top, side, wrist view)\n"
    "Use these + the task instruction to generate a detailed and *step-wise executable* plan.\n"
)

content = []
for image in image_paths:
    content.append({"type": "image", "image": image})
content.append({"type": "text", "text": task})

messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]


# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
