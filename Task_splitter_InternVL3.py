
from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig
from lmdeploy.vl import load_image

model = "/sda1/InternVL3-14B"

pipe = pipeline(
    model,
    backend_config=TurbomindEngineConfig(session_len=16384, tp=1),
    chat_template_config=ChatTemplateConfig(model_name='internvl2_5')
)

image1 = load_image('/home/sylee/codes/Data_generation_for_robots/image/task_1/init/top_Color.png')
image2 = load_image('/home/sylee/codes/Data_generation_for_robots/image/task_1/init/side_Color.png')

images = [image1, image2]

response = pipe(('describe these images', images))

print(response.text)
