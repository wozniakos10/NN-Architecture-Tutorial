## Cloning the DeepSeek-OCR Repository

1. Install Git LFS (Large File Storage):
   ```
   git lfs install
   ```

2. Clone the DeepSeek-OCR repository:
   ```
   git clone https://huggingface.co/deepseek-ai/DeepSeek-OCR
   ```

**Note:**  
By default, DeepSeek-OCR is configured to use CUDA and requires a GPU.

If you want to run it on a CPU, download the `modeling_deepseekocr.py` file from [this discussion](https://huggingface.co/deepseek-ai/DeepSeek-OCR/discussions/21/files#d2h-465181) and replace your local file with it.