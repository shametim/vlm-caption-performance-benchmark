import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import time
from tqdm import tqdm
import gc
from typing import List, Optional
import torch.cuda.amp as amp

# Configure device and dtype
device = "cuda:0" if torch.cuda.is_available() else "mps"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load model and processor with optimizations
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-base",
    torch_dtype=torch_dtype,
    trust_remote_code=True,
    device_map=device
).to(device)

# Compile model if using CUDA
if torch.cuda.is_available():
    model = torch.compile(model)
    
processor = AutoProcessor.from_pretrained(
    "microsoft/Florence-2-base",
    trust_remote_code=True
)

def process_batch(images: List[Image.Image], task_prompts: List[str], batch_size: int = 8) -> List[str]:
    """Process a batch of images efficiently"""
    try:
        with torch.no_grad(), amp.autocast(device_type='cuda', dtype=torch_dtype):
            # Prepare inputs
            inputs = processor(
                text=task_prompts,
                images=images,
                return_tensors="pt",
                padding=True
            ).to(device, torch_dtype)
            
            # Generate with optimized settings
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                length_penalty=1.0,
                early_stopping=True,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id
            )
            
            # Process outputs
            generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=False)
            
            parsed_answers = [
                processor.post_process_generation(
                    text, task=task_prompts[0],
                    image_size=(images[0].width, images[0].height)
                )
                for text in generated_texts
            ]
            
            return parsed_answers
    except Exception as e:
        print(f"Error processing batch: {str(e)}")
        return ["Error"] * len(images)
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def process_images(num_iterations: int = 1000, batch_size: int = 8) -> List[str]:
    """Process multiple images with batching"""
    try:
        # Load image once
        url = "https://d1ja9tyo8nbkbc.cloudfront.net/51411227_S0426/S0426/S0426-R0100/618865296/618865296-28.jpg?version=1731175261&width=640"
        image = Image.open(requests.get(url, stream=True).raw)
        
        # Create batches
        images = [image] * batch_size
        prompts = ["<OD>"] * batch_size
        
        start_time = time.time()
        results = []
        
        # Process in batches with progress bar
        for i in tqdm(range(0, num_iterations, batch_size), desc="Processing batches"):
            current_batch_size = min(batch_size, num_iterations - i)
            batch_results = process_batch(
                images[:current_batch_size],
                prompts[:current_batch_size],
                current_batch_size
            )
            results.extend(batch_results)
            
            # Periodic cleanup
            if i % (batch_size * 10) == 0:
                gc.collect()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate and display statistics
        images_per_second = num_iterations / total_time
        ms_per_image = (total_time * 1000) / num_iterations
        
        print(f"\nProcessing complete:")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average speed: {images_per_second:.2f} images/second")
        print(f"Average time per image: {ms_per_image:.2f} ms")
        
        return results
    
    except Exception as e:
        print(f"Error in process_images: {str(e)}")
        return []

def cleanup():
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    try:
        results = process_images()
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        cleanup()
