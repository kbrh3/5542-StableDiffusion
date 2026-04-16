import streamlit as st
import torch
from diffusers import StableDiffusionPipeline

st.title("CS 5542 - Controlled Product Image Generation")

@st.cache_resource
def load_pipe():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=dtype
    )
    return pipe.to(device)

pipe = load_pipe()

prompt = st.text_area("Prompt")
negative_prompt = st.text_area("Negative Prompt", value="blurry, low quality, distorted")

if st.button("Generate"):
    image = pipe(prompt, negative_prompt=negative_prompt).images[0]
    st.image(image, caption="Generated output", use_container_width=True)
