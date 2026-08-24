import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
from transformers import pipeline
qa = pipeline("text-generation", model="TheBloke/zephyr-7b-beta-GPTQ")
prompt = "Explain how deforestation affects climate change."
print(qa(prompt, max_new_tokens=200)[0]['generated_text'])