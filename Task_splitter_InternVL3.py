from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig
from lmdeploy.vl import load_image
from lmdeploy.vl.constants import IMAGE_TOKEN

model = '/sda1/InternVL3-14B'
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=4096, tp=1), chat_template_config=ChatTemplateConfig(model_name='internvl2_5'))

image_urls=[
    '/home/sylee/codes/Data_generation_for_robots/image/task_1/init/top_Color.png',
    '/home/sylee/codes/Data_generation_for_robots/image/task_1/final/top_Color.png'
]

images = [load_image(img_url) for img_url in image_urls]
# Numbering images improves multi-image conversations
response = pipe((f'Image-1: {IMAGE_TOKEN}\nImage-2: {IMAGE_TOKEN}\ndescribe these two images', images))
print(response.text)