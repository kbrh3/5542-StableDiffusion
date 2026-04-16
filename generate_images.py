import os
import torch
from diffusers import StableDiffusionPipeline
from products import products
from prompts import get_shoe_prompts, get_perfume_prompts, get_witch_prompts

MODEL_ID = "runwayml/stable-diffusion-v1-5"


def make_pipe():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype
    )
    pipe = pipe.to(device)
    return pipe


def save_group(pipe, group_name, prompt_items):
    out_dir = os.path.join("outputs", group_name)
    os.makedirs(out_dir, exist_ok=True)

    for i, item in enumerate(prompt_items, start=1):
        kwargs = {"prompt": item["prompt"]}
        if item["negative_prompt"]:
            kwargs["negative_prompt"] = item["negative_prompt"]

        generator = torch.manual_seed(42)
        image = pipe(**kwargs,generator=generator).images[0]
        filename = f"{i}_{item['label']}.png"
        image.save(os.path.join(out_dir, filename))
        print(f"Saved {group_name}/{filename}")


def main():
    pipe = make_pipe()

    save_group(pipe, "shoes", get_shoe_prompts(products[0]))
    save_group(pipe, "perfume", get_perfume_prompts(products[1]))
    save_group(pipe, "witch_hat", get_witch_prompts(products[2]))


if __name__ == "__main__":
    main()
