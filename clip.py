import time

import requests
import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizerFast


def process_images_clip(num_iterations=1):
    # Determine device - prefer MPS if available, fallback to CUDA or CPU
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Initialize model and processor
    model_name_or_path = "../LLM2CLIP-Openai-L-14-336"
    # model_name_or_path = "openai/clip-vit-large-patch14-336"
    image_size = 336
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    model = (
        CLIPModel.from_pretrained(
            model_name_or_path, torch_dtype=torch.float16, trust_remote_code=True
        )
        .to(device)
        .eval()
    )

    # Initialize tokenizer along with model and processor
    tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-large-patch14-336")

    # Load image once
    url = "https://d1ja9tyo8nbkbc.cloudfront.net/51411227_S0426/S0426/S0426-R0100/618865296/618865296-28.jpg?version=1731175261&width=640"
    image = Image.open(requests.get(url, stream=True).raw)

    start_time = time.time()

    # Generate image embeddings
    input_pixels = processor(images=image, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        image_features = model.get_image_features(input_pixels)

    # List of objective captions to compare
    captions = [
        "a storage shed in the backyard",
        "a shed with windows",
        "a metal storage shed",
        "a shed with double doors",
        "a storage shed",
        "a garden shed",
        "a storage unit in the backyard",
        "a shed with a metal roof",
        "a shed for lawn equipment",
        "an outdoor storage building",
    ]

    # Process text embeddings
    text_inputs = tokenizer(captions, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)

    print(f"Number of image embeddings: {image_features.shape[1]}")

    print(f"Number of text embeddings: {text_features.shape[1]}")

    # Calculate similarity scores
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    similarity = (100 * image_features @ text_features.T).softmax(dim=-1)

    # Print results sorted by similarity
    results = [(captions[i], similarity[0][i].item()) for i in range(len(captions))]
    results.sort(key=lambda x: x[1], reverse=True)

    print("\nTop matches:")
    for caption, score in results:
        print(f"{caption}: {score:.2f}%")

    end_time = time.time()
    total_time = end_time - start_time

    # Calculate and display statistics
    images_per_second = num_iterations / total_time
    ms_per_image = (total_time * 1000) / num_iterations

    print("\nProcessing complete:")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average speed: {images_per_second:.2f} images/second")
    print(f"Average time per image: {ms_per_image:.2f} ms")


if __name__ == "__main__":
    process_images_clip()
