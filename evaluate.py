import os
import itertools
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from transformers import CLIPProcessor, CLIPModel
from skimage.metrics import structural_similarity as ssim

device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

resnet = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
resnet.eval()
feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1]).to(device)

img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def load_image(path):
    return Image.open(path).convert("RGB")


def pil_to_np(img, size=(256, 256)):
    return np.array(img.resize(size)).astype(np.float32) / 255.0


def cosine_similarity(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def get_clip_alignment_score(image, prompt):
    inputs = clip_processor(text=[prompt], images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        img_emb = outputs.image_embeds[0]
        txt_emb = outputs.text_embeds[0]

    img_emb = img_emb / img_emb.norm()
    txt_emb = txt_emb / txt_emb.norm()
    return float((img_emb * txt_emb).sum().item())


def get_resnet_embedding(image):
    x = img_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = feature_extractor(x).flatten().cpu().numpy()
    return feat


def get_ssim_score(img1, img2):
    a = pil_to_np(img1)
    b = pil_to_np(img2)
    return float(ssim(a, b, channel_axis=2, data_range=1.0))


def get_sharpness_score(image):
    gray = np.array(image.convert("L").resize((256, 256))).astype(np.float32)
    gy, gx = np.gradient(gray)
    gyy, _ = np.gradient(gy)
    _, gxx = np.gradient(gx)
    lap = gxx + gyy
    return float(np.var(lap))


def get_contrast_score(image):
    gray = np.array(image.convert("L").resize((256, 256))).astype(np.float32)
    return float(np.std(gray))


def evaluate_group(group_name, image_paths, prompts, labels):
    rows = []
    for path, prompt, label in zip(image_paths, prompts, labels):
        img = load_image(path)
        rows.append({
            "group": group_name,
            "label": label,
            "file": path,
            "prompt_alignment_clip": get_clip_alignment_score(img, prompt),
            "quality_sharpness": get_sharpness_score(img),
            "quality_contrast": get_contrast_score(img),
        })

    df = pd.DataFrame(rows)

    pair_rows = []
    for i, j in itertools.combinations(range(len(image_paths)), 2):
        img1 = load_image(image_paths[i])
        img2 = load_image(image_paths[j])
        emb1 = get_resnet_embedding(img1)
        emb2 = get_resnet_embedding(img2)

        ssim_val = get_ssim_score(img1, img2)

        pair_rows.append({
            "group": group_name,
            "pair": f"{labels[i]} vs {labels[j]}",
            "ssim_consistency": ssim_val,
            "embedding_consistency": cosine_similarity(emb1, emb2),
            "diversity_1_minus_ssim": 1.0 - ssim_val,
        })

    return df, pd.DataFrame(pair_rows)


def summarize_metrics(df, pair_df, group_name):
    sharp_norm = df["quality_sharpness"] / df["quality_sharpness"].max()
    contrast_norm = df["quality_contrast"] / df["quality_contrast"].max()

    summary = pd.DataFrame([{
        "Prompt Alignment": df["prompt_alignment_clip"].mean(),
        "Quality": (sharp_norm.mean() + contrast_norm.mean()) / 2,
        "Consistency": (pair_df["ssim_consistency"].mean() + pair_df["embedding_consistency"].mean()) / 2,
        "Diversity": pair_df["diversity_1_minus_ssim"].mean(),
        "group": group_name
    }])

    return summary.round(3)
