import os
# import logging
# logging.basicConfig(level=logging.DEBUG)
import sys, time

sys.path.append(os.path.abspath(__file__).rsplit("/", 2)[0])
import argparse

from PIL import Image
import torch
torch.set_float32_matmul_precision('high')
torch.manual_seed(3407)
if torch.cuda.is_available():
    torch.cuda.manual_seed(3407)
    torch.cuda.manual_seed_all(3407)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
from inference_solver import FlexARInferenceSolver
from xllmx.util.misc import random_seed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='Alpha-VLLM/Lumina-mGPT-7B-768')
    parser.add_argument("--save_path", type=str, default='/mnt/workspace/Lumina-mGPT')
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=2000)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("-n", type=int, default=1)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--zipar_window_size", type=int, default=16)

    args = parser.parse_args()

    print("args:\n", args)
    
    prompt = 'Image of a bustling downtown street in Tokyo at night, with neon signs, crowded sidewalks, and tall skyscrapers.'
    l_prompts = [prompt]

    t = args.temperature
    top_k = args.top_k
    cfg = args.cfg
    n = args.n
    w, h = args.width, args.height
    file_name = 'sample.jpg'

    inference_solver = FlexARInferenceSolver(
        model_path=args.model_path,
        precision="bf16",
        target_size=args.width
    )
    inference_solver.model.cfg = cfg
    inference_solver.model.zipar_window_size = args.zipar_window_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.reset_max_memory_allocated(device)  # Reset the max memory counter

    with torch.no_grad():
        l_generated_all = []
        for i, prompt in enumerate(l_prompts):
            for repeat_idx in range(n):
                random_seed(repeat_idx)
                torch.cuda.synchronize()
                start = time.perf_counter()
                generated = inference_solver.generate(
                    images=[],
                    qas=[[f"Generate an image of {w}x{h} according to the following prompt:\n{prompt}", None]],
                    max_gen_len=2500,
                    temperature=t,
                    logits_processor=inference_solver.create_logits_processor(cfg=cfg, image_top_k=top_k),
                )
                inference_solver.model.reset_zipar()
                end = time.perf_counter()
                print('Time for one image: ', end-start, 's')
                try:
                    l_generated_all.append(generated[1][0])
                except:
                    l_generated_all.append(Image.new("RGB", (w, h)))

        result_image = inference_solver.create_image_grid(l_generated_all, len(l_prompts), n)
        result_image.save(os.path.join(args.save_path, file_name))

    # Get the maximum GPU memory usage
    max_memory_used = torch.cuda.max_memory_allocated(device)
    print(f"Maximum GPU memory used: {max_memory_used / 1024 ** 2:.2f} MB")
