import os
import math
import torch
import torch.distributed as dist
from torch.multiprocessing import spawn
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig, BitsAndBytesConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

if __name__ == "__main__":

    # device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    # quantization_config = BitsAndBytesConfig(load_in_4bit=True)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model_path = '/sda1/InternVL3-14B'
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype="auto",
        load_in_8bit=True,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        use_flash_attn=True,
        device_map="auto",
        # quantization_config=quantization_config,
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    pixel_values1 = load_image('/home/sylee/codes/Data_generation_for_robots/image/task_5/init/top_Color.png', max_num=12).to(torch.bfloat16).to(device)
    pixel_values2 = load_image('/home/sylee/codes/Data_generation_for_robots/image/task_5/final/top_Color.png', max_num=12).to(torch.bfloat16).to(device)
    pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)

    # Separate images
    num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]

    # 3. 이미지 텍스트 명령 구성
    question = 'Image-1: <image>\nImage-2: <image>\nDescribe the two images in detail.'


    # 5. 생성 config
    generation_config = dict(max_new_tokens=1024, do_sample=True)

    response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                                   num_patches_list=num_patches_list,
                                   history=None, return_history=True)

    print(f'User: {question}\nAssistant: {response}')


