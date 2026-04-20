# 5542-StableDiffusion
# CS 5542 Quiz Challenge 1 - Controlled Product Image Generation

## Project Overview
This project uses Stable Diffusion v1.5 to generate e-commerce-style product images from structured metadata and using different prompt strategies.

The system includes:
- Structured and negative prompt design
- Multiple generation strategies (naive → refined)
- Evaluation using metrics (alignment, quality, consistency, diversity)
- 
## Products Tested
- Red Running Shoes
- Luxury Perfume Bottle
- Black Witch Hat

## Control Mechanisms
- Structured prompt templates
- Negative prompts

## Evaluation Metrics
- Prompt Alignment
- Quality
- Consistency
- Diversity



The system includes:
- Structured and negative prompt design
- Multiple generation strategies (naive → refined)
- Evaluation using metrics (alignment, quality, consistency, diversity)

---

## IMPORTANT: Recommended Way to Run

Running the Python scripts locally may require GPU setup and dependencies.

**Recommended approach is to run the notebook in Google Colab**

---

## 🚀 How to Run in Google Colab (Best Option)

### Step 1: Download the Notebook
- Download this file from the repository "5542_challenge.ipynb"
- 
---

### Step 2: Open Colab
https://colab.research.google.com

---

### Step 3: Upload Notebook
- Click **"Upload"**
- Select `5542_challenge.ipynb`

---

### Step 4: Enable GPU (IMPORTANT)
- I have been using T4 GPU, Python 3

### Step 5: Run all Cells
  The notebook will
- Generate product images
- Display outputs
- Compute evaluation metrics

  ## 📊 Evaluation Metrics

- Prompt Alignment (CLIP)
- Quality (sharpness + contrast)
- Consistency (SSIM + embedding similarity)
- Diversity (1 − SSIM)

---

## AI Tools Used

- ChatGPT (code assistance, prompt design)
- Hugging Face Diffusers (Stable Diffusion pipeline)
- PyTorch / Torchvision (model + embeddings)
- Google Colab (execution environment)

---

## Key Insight

Structured prompts improve image quality and alignment,  
but tradeoffs remain between consistency and diversity.

---

## Demo Video
(video link here - do not forget this girl)

---
