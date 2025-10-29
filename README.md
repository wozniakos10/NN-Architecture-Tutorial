## DeepSeek-OCR: Quick Start Guide

---

### 1. Google Colab (Recommended for Quick Testing)

If you want to test DeepSeek-OCR without using local resources:

- Use the notebook at `notebooks/deepsek_ocr.ipynb`.
- Open it in Google Colab.
- **Important:** Select a runtime with GPU. This configuration will not work with CPU.

---

### 2. Local Runtime

#### 2.1. Install Git LFS

```sh
git lfs install
```

#### 2.2. Clone the DeepSeek-OCR Repository

```sh
git clone https://huggingface.co/deepseek-ai/DeepSeek-OCR
```

#### 2.3. GPU/CPU Support

- **Default:**  
  DeepSeek-OCR is configured to use CUDA and requires a GPU.

- **To run on CPU:**  
  Download the `modeling_deepseekocr.py` file from [this discussion](https://huggingface.co/deepseek-ai/DeepSeek-OCR/discussions/21/files#d2h-465181) and replace your local file with it.

#### 2.4. Installing PyTorch with CUDA via `pyproject.toml`

To set up your `pyproject.toml` to download a CUDA-compatible version of torch, add:

```toml
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = [
  { index = "pytorch-cu128", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]
torchvision = [
  { index = "pytorch-cu128", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]
```

For more information about specific CUDA versions, see the [official uv documentation](https://docs.astral.sh/uv/guides/integration/pytorch/).

#### 2.5. Set Up the Environment

```sh
uv sync
```

#### 2.6. Testing DeepSeek-OCR

- Modify the configuration parameters (image path, prompt type, etc.) in the `run_ocr.py` file.
- Run the script to test DeepSeek-OCR.