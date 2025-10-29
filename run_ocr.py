import time

import torch
from transformers import AutoModel, AutoTokenizer

# =========================
# Configuration
# =========================

MODEL_NAME = "./DeepSeek-OCR/"
PROMPT = "<image>\nLocate <|ref|>dog<|/ref|> in the image."
IMAGE_FILE = "./dog.jpg"
OUTPUT_PATH = "./output/dog"

# =========================
# Prompt Examples
# =========================
# document: <image>\n<|grounding|>Convert the document to markdown.
# other image: <image>\n<|grounding|>OCR this image.
# without layouts: <image>\nFree OCR.
# figures in document: <image>\nParse the figure.
# general: <image>\nDescribe this image in detail.
# rec: <image>\nLocate <|ref|>xxxx<|/ref|> in the image.
# '先天下之忧而忧'

# =========================
# Model and Tokenizer Loading
# =========================

start = time.time()

if torch.cuda.is_available():
    print("Using CUDA GPU")
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        _attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        use_safetensors=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = model.eval().cuda()
else:
    print("Using CPU (no GPU detected)")
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        _attn_implementation="eager",
        trust_remote_code=True,
        use_safetensors=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = model.eval().to(torch.bfloat16)

end = time.time()
print(f"Model and tokenizer loaded in {end - start:.2f} seconds.")

# =========================
# OCR Output Parsing Utility
# =========================


def parse_ocr_output(raw_output: str) -> str:
    """
    Parse raw OCR output to remove debug info and format cleanly.
    """
    lines = raw_output.split("\n")
    parsed_lines = []
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
        if not stripped or any(pattern in line for pattern in skip_patterns):
            continue
        if "<|ref|>" in line:
            import re

            pattern = r"<\\|ref\\|>(.*?)<\\|/ref\\|>(?:<\\|det\\|>\\[\\[(.*?)\\]\\]<\\|/det\\|>)?"
            matches = re.findall(pattern, line)
            for ref_text, coords in matches:
                if coords:
                    parsed_lines.append(f"• **{ref_text}** → `[{coords}]`")
                else:
                    parsed_lines.append(ref_text.strip())
            continue
        parsed_lines.append(stripped)
    result = "\n".join(parsed_lines)
    return result if result.strip() else raw_output


# =========================
# Inference
# =========================

# Model size presets:
# Tiny:   base_size=512,  image_size=512,  crop_mode=False
# Small:  base_size=640,  image_size=640,  crop_mode=False
# Base:   base_size=1024, image_size=1024, crop_mode=False
# Large:  base_size=1280, image_size=1280, crop_mode=False
# Gundam: base_size=1024, image_size=640,  crop_mode=True

start = time.time()
res = model.infer(
    tokenizer,
    prompt=PROMPT,
    image_file=IMAGE_FILE,
    output_path=OUTPUT_PATH,
    base_size=1024,
    image_size=640,
    crop_mode=True,
    save_results=True,
    test_compress=False,
)
end = time.time()
print(f"Inference completed in {end - start:.2f} seconds.")
print("===== DeepSeek-OCR Result =====")
print(res)
