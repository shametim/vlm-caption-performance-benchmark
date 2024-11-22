import requests
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import time
from tqdm import tqdm

model_id = "vikhyatk/moondream2"
revision = "2024-08-26"
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision
)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

def process_images(num_iterations=1000):
    # Load image once
    url = "https://d1ja9tyo8nbkbc.cloudfront.net/51411227_S0426/S0426/S0426-R0100/618865296/618865296-28.jpg?version=1731175261&width=640"
    image = Image.open(requests.get(url, stream=True).raw)
    
    start_time = time.time()
    
    # Process in loop with progress bar
    for _ in tqdm(range(num_iterations), desc="Processing images"):
        enc_image = model.encode_image(image)
        result = model.answer_question(enc_image, "What room is this?", tokenizer)
        print(result)
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate and display statistics
    images_per_second = num_iterations / total_time
    ms_per_image = (total_time * 1000) / num_iterations
    
    print(f"\nProcessing complete:")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average speed: {images_per_second:.2f} images/second")
    print(f"Average time per image: {ms_per_image:.2f} ms")

if __name__ == "__main__":
    process_images()
