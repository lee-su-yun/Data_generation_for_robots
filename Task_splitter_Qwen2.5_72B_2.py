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
    "/home/sylee/codes/Data_generation_for_robots/image/task_25/init/top_Color.png",
    "/home/sylee/codes/Data_generation_for_robots/image/task_25/init/side_Color.png",
   # "/home/sylee/codes/Data_generation_for_robots/image/task_1/init/wrist_Color.png",
    "/home/sylee/codes/Data_generation_for_robots/image/task_25/final/top_Color.png",
    "/home/sylee/codes/Data_generation_for_robots/image/task_25/final/side_Color.png",
   # "/home/sylee/codes/Data_generation_for_robots/image/task_1/final/wrist_Color.png",
]


system_prompt = (
    "You are an expert robotics planner. Given an initial and final image of a tabletop and a task instruction, "
    "you must infer the robot arm’s motion plan: decompose it into subtasks, then provide low-level primitive actions with reasoning.\n"
    "In the <MOVE> section, do not just state the destination. Instead, describe the **detailed movement path**, "
)

user_input = (
    "# Task Instruction:\n"
    "Move the white paper cup numbered 3 to the position where the light blue cup is currently located.\n\n"

    "# Initial Image: [image1.png], [image2.png]\n"
    "# Final Image: [image3.png], [image4.png]\n\n"

    "# Please output in the following format:\n"
    "## 1. Describe the Task\n"
    "- Object identities and colors\n"
    "- Initial and final positions\n\n"

    "## 2. <PLAN>\n"
    "- Subtask 1: ...\n"
    "- <MOVE>: [...]\n\n"

    "<PLANNING_reason>: ...\n\n"
    "FINISHED"
)

content = []
for image in image_paths:
    content.append({"type": "image", "image": image})
content.append({"type": "text", "text": user_input})

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

# Greedy
generated_ids = model.generate(**inputs, max_new_tokens=1024)

# Sampling
# generated_ids = model.generate(
#     **inputs,
#     do_sample=True,
#     temperature=0.7,
#     top_p=0.9,
#     max_new_tokens=1024,
# )

# Beam Search
# generated_ids = model.generate(
#     **inputs,
#     do_sample=False,
#     num_beams=6,
#     num_beam_groups=3,
#     diversity_penalty=1.0,
#     max_new_tokens=1024,
# )

# Diverse Beam Search
# generated_ids = model.generate(
#     **inputs,
#     do_sample=False,
#     num_beams=6,
#     num_beam_groups=3,
#     diversity_penalty=1.0,
#     max_new_tokens=1024,
# )

generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
print('\n')
print('task25')

# with open("/home/sylee/codes/Data_generation_for_robots/task/reason30.txt", "w") as f:
#     f.write(thinking_content)