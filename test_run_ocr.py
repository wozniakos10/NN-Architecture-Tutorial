from transformers import AutoModel, AutoTokenizer
import torch
import os
model_name = './DeepSeek-OCR/'
import time

#prompts examples:
# document: <image>\n<|grounding|>Convert the document to markdown.
# other image: <image>\n<|grounding|>OCR this image.
# without layouts: <image>\nFree OCR.
# figures in document: <image>\nParse the figure.
# general: <image>\nDescribe this image in detail.
# rec: <image>\nLocate <|ref|>xxxx<|/ref|> in the image.
# '先天下之忧而忧'

start = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, _attn_implementation='eager', trust_remote_code=True, use_safetensors=True)
model = model.eval()
end = time.time()
print(f"Model and tokenizer loaded in {end - start:.2f} seconds.")

prompt = "<image>\nLocate <|ref|>fish<|/ref|> in the image."
image_file = './fish_image.jpg'
output_path = './output/fish'

# infer(self, tokenizer, prompt='', image_file='', output_path = ' ', base_size = 1024, image_size = 640, crop_mode = True, test_compress = False, save_results = False):

# Tiny: base_size = 512, image_size = 512, crop_mode = False
# Small: base_size = 640, image_size = 640, crop_mode = False
# Base: base_size = 1024, image_size = 1024, crop_mode = False
# Large: base_size = 1280, image_size = 1280, crop_mode = False

start = time.time()
# Gundam: base_size = 1024, image_size = 640, crop_mode = True
res = model.infer(tokenizer, prompt=prompt, image_file=image_file, output_path = output_path, base_size = 1024, image_size = 640, crop_mode=True, save_results = True, test_compress = True)
end = time.time()
print(f"Inference completed in {end - start:.2f} seconds.")
