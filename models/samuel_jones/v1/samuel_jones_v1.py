# from transformers import AutoModel, AutoTokenizer, pipeline
# import torch
#
# model_name = "eci-io/climategpt-7b"
#
#
# qa = pipeline(
#     "text2text-generation",
#     model=model_name,
#     device=0  # set -1 for CPU
# )
#
# def generate_climate_answer(prompt, max_tokens=128):
#     result = qa(prompt, max_length=max_tokens, do_sample=True)
#     return result[0]['generated_text']
#
# print("Enter your climate-related question:")
# prompt_input = input()
# print("\nAnswer:\n")
# print(generate_climate_answer(prompt_input))

from transformers import pipeline


model_name = "google/flan-t5-base"

qa = pipeline(
    "text2text-generation",
    model=model_name,
    device=0  # or -1 for CPU
)

# Number of tokens is deciding factor
def generate_climate_answer(prompt, max_tokens=256):
    result = qa(prompt, max_new_tokens=max_tokens)
    return result[0]['generated_text']

print("Enter your climate-related question:")
prompt_input = input()
print("\nAnswer:\n")
print(generate_climate_answer(prompt_input))

#Look at streamlit