import time
from typing import List

import numpy as np
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers.image_utils import load_image

DEVICE = "cuda" if torch.cuda.is_available() else "mps"
BATCH_SIZE = 1
NUM_RUNS = 1


def load_sample_images(num_images: int) -> List:
    # Sample image URLs to test with
    image_urls = [
        "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg",
        "https://huggingface.co/spaces/merve/chameleon-7b/resolve/main/bee.jpg",
    ]

    # Repeat images to reach desired batch size
    images = []
    while len(images) < num_images:
        for url in image_urls:
            if len(images) < num_images:
                images.append(load_image(url))
    return images


def run_batch_inference(model, processor, images: List, device: str):
    # Create input messages for batch
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"} for _ in range(len(images))]
            + [{"type": "text", "text": "Describe the image."}],
        },
    ]

    # Prepare inputs
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=images, return_tensors="pt")
    inputs = inputs.to(device)

    # Generate outputs
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=500)
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_texts


def benchmark():
    print("Loading model and processor...")
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
    model = AutoModelForVision2Seq.from_pretrained(
        "HuggingFaceTB/SmolVLM-Instruct",
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
    ).to(DEVICE)

    print(f"Loading {BATCH_SIZE} sample images...")
    images = load_sample_images(BATCH_SIZE)

    # Warmup run
    print("Performing warmup run...")
    _ = run_batch_inference(model, processor, images, DEVICE)

    # Benchmark runs
    print(f"\nStarting benchmark: {NUM_RUNS} runs with batch size {BATCH_SIZE}")
    times = []
    total_images = 0

    for i in range(NUM_RUNS):
        start_time = time.time()
        _ = run_batch_inference(model, processor, images, DEVICE)
        end_time = time.time()

        run_time = end_time - start_time
        times.append(run_time)
        total_images += BATCH_SIZE

        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{NUM_RUNS} runs...")

    # Calculate statistics
    total_time = sum(times)
    avg_time_per_run = np.mean(times)
    std_time = np.std(times)
    images_per_second = total_images / total_time

    print("\nBenchmark Results:")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per batch: {avg_time_per_run:.2f} seconds (±{std_time:.2f})")
    print(f"Images per second: {images_per_second:.2f}")
    print(f"Total images processed: {total_images}")


if __name__ == "__main__":
    benchmark()
