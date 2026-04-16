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

from products import products
from prompts import get_shoe_prompts, get_perfume_prompts, get_witch_prompts

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
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing image: {path}")
    return Image.open(path).convert("RGB")


def pil_to_np(img, size=(256, 256)):
    return np.array(img.resize(size)).astype(np.float32) / 255.0


def cosine_similarity(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def get_clip_alignment_score(image, prompt):
    inputs = clip_processor(
        text=[prompt],
        images=image,
        return_tensors="pt",
        padding=True
    ).to(device)

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

    pair_df = pd.DataFrame(pair_rows)
    return df, pair_df


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


def save_metrics(df, pair_df, summary_df, group_name):
    os.makedirs("metrics", exist_ok=True)
    df.to_csv(f"metrics/{group_name}_per_image_metrics.csv", index=False)
    pair_df.to_csv(f"metrics/{group_name}_pairwise_metrics.csv", index=False)
    summary_df.to_csv(f"metrics/{group_name}_summary.csv", index=False)


def main():
    os.makedirs("metrics", exist_ok=True)

    # =========================
    # SHOES
    # =========================
    shoe_prompt_items = get_shoe_prompts(products[0])
    shoe_paths = [f"outputs/shoes/{i+1}_{item['label']}.png" for i, item in enumerate(shoe_prompt_items)]
    shoe_prompts = [item["prompt"] for item in shoe_prompt_items]
    shoe_labels = [item["label"] for item in shoe_prompt_items]

    shoe_df, shoe_pair_df = evaluate_group("shoes", shoe_paths, shoe_prompts, shoe_labels)
    shoe_summary = summarize_metrics(shoe_df, shoe_pair_df, "shoes")
    save_metrics(shoe_df, shoe_pair_df, shoe_summary, "shoes")

    # =========================
    # PERFUME
    # =========================
    perfume_prompt_items = get_perfume_prompts(products[1])
    perfume_paths = [f"outputs/perfume/{i+1}_{item['label']}.png" for i, item in enumerate(perfume_prompt_items)]
    perfume_prompts = [item["prompt"] for item in perfume_prompt_items]
    perfume_labels = [item["label"] for item in perfume_prompt_items]

    perfume_df, perfume_pair_df = evaluate_group("perfume", perfume_paths, perfume_prompts, perfume_labels)
    perfume_summary = summarize_metrics(perfume_df, perfume_pair_df, "perfume")
    save_metrics(perfume_df, perfume_pair_df, perfume_summary, "perfume")

    # =========================
    # WITCH HAT
    # =========================
    witch_prompt_items = get_witch_prompts(products[2])
    witch_paths = [f"outputs/witch_hat/{i+1}_{item['label']}.png" for i, item in enumerate(witch_prompt_items)]
    witch_prompts = [item["prompt"] for item in witch_prompt_items]
    witch_labels = [item["label"] for item in witch_prompt_items]

    witch_df, witch_pair_df = evaluate_group("witch_hat", witch_paths, witch_prompts, witch_labels)
    witch_summary = summarize_metrics(witch_df, witch_pair_df, "witch_hat")
    save_metrics(witch_df, witch_pair_df, witch_summary, "witch_hat")

    # =========================
    # FINAL SUMMARY
    # =========================
    final_summary = pd.concat(
        [shoe_summary, perfume_summary, witch_summary],
        ignore_index=True
    )
    final_summary.to_csv("metrics/final_summary.csv", index=False)

    print("\n=== FINAL SUMMARY ===")
    print(final_summary.to_string(index=False))


if __name__ == "__main__":
    main()
