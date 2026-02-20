# 🎬 GPT-Neo 125M — Movie Review Generator (LoRA)

A fine-tuned version of **GPT-Neo 125M** using **LoRA (Low-Rank Adaptation)** trained on the Cornell Movie Review dataset (Rotten Tomatoes). The model generates fluent, sentiment-aware movie review continuations given a text prompt.

---

## 📋 Model Details

| | |
|---|---|
| **Base Model** | [EleutherAI/gpt-neo-125M](https://huggingface.co/EleutherAI/gpt-neo-125M) |
| **Method** | LoRA (Parameter-Efficient Fine-Tuning) |
| **Task** | Causal Language Modeling / Text Generation |
| **Language** | English |
| **License** | Apache-2.0 |

---

## 📦 Files

| File | Description |
|---|---|
| `adapter_model.safetensors` | LoRA adapter weights |
| `adapter_config.json` | LoRA configuration (rank, alpha, modules) |
| `tokenizer.json` | Tokenizer vocabulary |
| `tokenizer_config.json` | Tokenizer settings |

---

## 🏋️ Training Details

| | |
|---|---|
| **Dataset** | cornell-movie-review-data/rotten_tomatoes |
| **Train size** | 8,530 samples |
| **Validation size** | 1,066 samples |
| **Epochs** | 3 |
| **Learning rate** | 2e-4 |
| **Batch size** | 8 |
| **LoRA rank (r)** | 8 |
| **LoRA alpha** | 16 |
| **LoRA dropout** | 0.05 |
| **Target modules** | q_proj, v_proj |
| **Trainable params** | 294,912 (0.235% of total) |
| **Perplexity** | 59.72 |

---

## 🚀 Quick Start

### Install dependencies
```bash
pip install transformers peft torch
```

### Run inference
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load
tokenizer  = AutoTokenizer.from_pretrained("your-username/gptneo-movies-lora")
base_model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
model      = PeftModel.from_pretrained(base_model, "your-username/gptneo-movies-lora")
model.eval()

# Generate
prompt = "The cinematography was breathtaking but"
inputs = tokenizer(prompt, return_tensors="pt")

output = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.8,
    top_p=0.95,
)

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

### Example outputs
```
Prompt:  "The cinematography was breathtaking but"
Output:  "The cinematography was breathtaking but very few of its viewers
          would recognize it as a masterpiece of the genre..."

Prompt:  "I would not recommend this film because"
Output:  "I would not recommend this film because it's so boring and
          because it's not the kind of movie that would satisfy a general audience..."
```

---

## 🌐 Deploy with FastAPI

```python
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

app       = FastAPI()
tokenizer = AutoTokenizer.from_pretrained("your-username/gptneo-movies-lora")
base      = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
model     = PeftModel.from_pretrained(base, "your-username/gptneo-movies-lora")

class Request(BaseModel):
    prompt: str

@app.post("/generate")
def generate(req: Request):
    inputs = tokenizer(req.prompt, return_tensors="pt")
    out    = model.generate(**inputs, max_new_tokens=100, do_sample=True)
    return {"generated_text": tokenizer.decode(out[0], skip_special_tokens=True)}
```

```bash
uvicorn app:app --reload
```

---

## ⚠️ Limitations

- Small model (125M parameters) — output quality is limited compared to larger models
- Trained only on short movie review snippets — may struggle with long-form generation
- May produce repetitive text on longer generations
- Not suitable for factual question answering

---

## 📚 Citation

```bibtex
@misc{gptneo-movies-lora,
  author    = {your-username},
  title     = {GPT-Neo 125M Movie Review Generator (LoRA)},
  year      = {2026},
  publisher = {HuggingFace},
  url       = {https://huggingface.co/your-username/gptneo-movies-lora}
}
```
