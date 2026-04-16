
def get_shoe_prompts(product):
    return [
        {
            "label": "naive",
            "prompt": product["title"],
            "negative_prompt": None
        },
        {
            "label": "structured_1",
            "prompt": f"""
High-quality e-commerce product photo of {product['title']},
category: {product['category']},
features: {product['attributes']},
white background, studio lighting, centered, professional product photography
""".strip(),
            "negative_prompt": "blurry, low quality, distorted, messy background"
        },
        {
            "label": "structured_2",
            "prompt": f"""
High-quality e-commerce product photo of {product['title']},
category: {product['category']},
features: {product['attributes']},
pure white background, isolated product, centered, no shadows,
professional studio lighting, catalog style, Amazon product image
""".strip(),
            "negative_prompt": """
blurry, low quality, distorted, colorful background, clutter,
multiple objects, artistic, dramatic lighting
""".strip()
        },
        {
            "label": "refined",
            "prompt": f"""
High-quality e-commerce product photo of {product['title']},
single product only, isolated object, pure white background,
centered, full product visible, no cropping,
professional catalog photography, Amazon listing style
""".strip(),
            "negative_prompt": """
multiple objects, extra shoes, cropped, zoomed, artistic,
colorful background, shadows, reflections, blurry
""".strip()
        }
    ]


def get_perfume_prompts(product):
    return [
        {
            "label": "naive",
            "prompt": product["title"],
            "negative_prompt": None
        },
        {
            "label": "mid",
            "prompt": f"""
High-quality photo of {product['title']},
{product['attributes']},
studio lighting, clean background
""".strip(),
            "negative_prompt": "blurry, low quality, distorted"
        },
        {
            "label": "structured",
            "prompt": f"""
High-quality e-commerce product photo of {product['title']},
category: {product['category']},
features: {product['attributes']},
white background, studio lighting, centered, professional product photography
""".strip(),
            "negative_prompt": "blurry, low quality, distorted, messy background"
        },
        {
            "label": "refined",
            "prompt": f"""
High-quality e-commerce product photo of {product['title']},
single product only, isolated object, pure white background,
centered, full product visible, no cropping,
professional catalog photography, Amazon listing style
""".strip(),
            "negative_prompt": """
multiple objects, extra bottles, cropped, zoomed, artistic,
colorful background, clutter, reflections, blurry
""".strip()
        }
    ]


def get_witch_prompts(product):
    return [
        {
            "label": "naive",
            "prompt": product["title"],
            "negative_prompt": None
        },
        {
            "label": "mid",
            "prompt": f"""
High-quality photo of {product['title']},
{product['attributes']},
studio lighting, clean background
""".strip(),
            "negative_prompt": "blurry, low quality, distorted"
        },
        {
            "label": "structured",
            "prompt": f"""
High-quality e-commerce product photo of {product['title']},
category: {product['category']},
features: {product['attributes']},
white background, studio lighting, centered, professional product photography
""".strip(),
            "negative_prompt": "blurry, low quality, distorted, messy background"
        },
        {
            "label": "refined",
            "prompt": f"""
High-quality e-commerce product photo of {product['title']},
single product only, isolated object, pure white background,
centered, full product visible, no cropping,
professional catalog photography, Amazon listing style
""".strip(),
            "negative_prompt": """
person, human, model, face, head,
multiple objects, extra hats,
costume scene, Halloween scene,
cropped, zoomed, artistic,
colorful background, clutter, blurry
""".strip()
        },
        {
            "label": "shape_focused",
            "prompt": """
High-quality e-commerce product photo of a black pointed witch hat,
tall cone-shaped structured hat, firm shape, wide brim,
solid black color, classic witch hat design,
single product only, isolated object, pure white background,
centered, full product visible, no cropping,
professional catalog photography
""".strip(),
            "negative_prompt": """
person, human, model, face,
top hat, fedora, soft hat, wrinkled fabric,
white hat, light color,
multiple objects, clutter, artistic
""".strip()
        }
    ]
