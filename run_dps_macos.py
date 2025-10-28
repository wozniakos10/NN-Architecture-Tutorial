import os
import sys
from pathlib import Path
import time

import torch
from transformers import AutoModel, AutoTokenizer


def main() -> None:
    """
    CPU-only test runner for DeepSeek-OCR on the bundled test image.

    - Loads tokenizer and model from the current repository directory.
    - Forces CPU execution (no CUDA) and casts weights to float32 for compatibility.
    - Runs `model.infer` on `test.png` and prints the recognized text.
    - Saves any visual outputs to `output/images`.
    """

    # Force CPU usage and make sure nothing tries to grab a GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    repo_dir = Path(__file__).resolve().parent

    model_dir = repo_dir  # model & tokenizer files live in the repo root
    image_path = repo_dir / "barca_photo.jpg"
    output_dir = repo_dir / "output" / "images.txt"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not image_path.exists():
        print(f"ERROR: test image not found at: {image_path}")
        sys.exit(1)

    start = time.time()
    # Load tokenizer and model from local files. Keep everything on CPU.
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

    model = AutoModel.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.float32,
        # device_map="cpu",
    )

    end = time.time()
    print(f"Model and tokenizer loaded in {end - start:.2f} seconds.")

    model = model.eval()  # ensure inference mode

    # Prompt directing the model to convert the document into markdown.
    prompt = "<image>\nDescribe this image in detail."
    start = time.time()
    # Run inference. `eval_mode=True` makes `infer` return the text string.
    with torch.inference_mode():
        result_text = model.infer(
            tokenizer,
            prompt=prompt,
            image_file=str(image_path),
            output_path=str(output_dir),
            base_size=1024,
            image_size=640,
            crop_mode=True,
            save_results=True,
            eval_mode=True,
        )
    end = time.time()
    print(f"Inference completed in {end - start:.2f} seconds.")
    print("===== DeepSeek-OCR (CPU) Result =====")
    if isinstance(result_text, str):
        print(result_text.strip())
    else:
        print("No textual result returned by model.")


if __name__ == "__main__":
    main()
