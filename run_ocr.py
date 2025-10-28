import time

import torch
from transformers import AutoModel, AutoTokenizer

model_name = "./DeepSeek-OCR/"

start = time.time()

# prompts examples:
# document: <image>\n<|grounding|>Convert the document to markdown.
# other image: <image>\n<|grounding|>OCR this image.
# without layouts: <image>\nFree OCR.
# figures in document: <image>\nParse the figure.
# general: <image>\nDescribe this image in detail.
# rec: <image>\nLocate <|ref|>xxxx<|/ref|> in the image.
# '先天下之忧而忧'

if torch.cuda.is_available():
    print("✅ Using CUDA GPU")
    model = AutoModel.from_pretrained(
        model_name,
        # flash attention requires modern hardware to utilize
        # FlashAttention requires Ampere or newer:
        # ✅ Works on: A100, RTX 3090, RTX 4080/4090, H100, L40
        # ❌ Doesn’t work on: T4, V100, RTX 2080, GTX 1080
        # _attn_implementation='flash_attention_2',
        _attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        use_safetensors=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = model.eval().cuda()
else:
    print("⚠️ Using CPU (no GPU detected)")
    model = AutoModel.from_pretrained(
        model_name, _attn_implementation="eager", trust_remote_code=True, use_safetensors=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = model.eval().to(torch.bfloat16)
end = time.time()
print(f"Model and tokenizer loaded in {end - start:.2f} seconds.")

prompt = "<image>\n<|grounding|>Convert the document to markdown."
image_file = "./math_notes.jpg"
output_path = "./output/math_notes"


def parse_ocr_output(raw_output: str) -> str:
    """Parse raw OCR output to remove debug info and format cleanly"""
    lines = raw_output.split("\n")
    parsed_lines = []
    in_content = False

    # Patterns to skip (debug/metadata)
    skip_patterns = [
        "BASE:",
        "PATCHES:",
        "NO PATCHES",
        "directly resize",
        "image size:",
        "valid image tokens:",
        "output texts tokens",
        "compression ratio:",
        "save results:",
        "====",
        "===",
    ]

    for line in lines:
        stripped = line.strip()

        # Skip empty lines and debug patterns
        if not stripped or any(pattern in line for pattern in skip_patterns):
            continue

        # Handle ref/det structured data
        if "<|ref|>" in line:
            # Extract all reference-detection pairs from this line
            import re

            pattern = r"<\|ref\|>(.*?)<\|/ref\|>(?:<\|det\|>\[\[(.*?)\]\]<\|/det\|>)?"
            matches = re.findall(pattern, line)

            if matches:
                for ref_text, coords in matches:
                    if coords:
                        # Format with coordinates
                        parsed_lines.append(f"• **{ref_text}** → `[{coords}]`")
                    else:
                        # Just the reference text
                        parsed_lines.append(ref_text.strip())
            continue

        # Regular content - add as is
        parsed_lines.append(stripped)

    result = "\n".join(parsed_lines)
    return result if result.strip() else raw_output


# infer(self, tokenizer, prompt='', image_file='', output_path = ' ', base_size = 1024, image_size = 640, crop_mode = True, test_compress = False, save_results = False):

# Tiny: base_size = 512, image_size = 512, crop_mode = False
# Small: base_size = 640, image_size = 640, crop_mode = False
# Base: base_size = 1024, image_size = 1024, crop_mode = False
# Large: base_size = 1280, image_size = 1280, crop_mode = False

# Gundam: base_size = 1024, image_size = 640, crop_mode = True
start = time.time()
res = model.infer(
    tokenizer,
    prompt=prompt,
    image_file=image_file,
    output_path=output_path,
    base_size=1280,
    image_size=1280,
    crop_mode=False,
    save_results=True,
    test_compress=False,
)
end = time.time()
print(f"Inference completed in {end - start:.2f} seconds.")
print("===== DeepSeek-OCR Result =====")
# print(res.strip())
print(res)
