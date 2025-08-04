import os
import json
import torch
from tqdm import tqdm
import numpy as np
import PIL.Image
from transformers import AutoProcessor
from blip3o.model.builder import load_pretrained_model
from blip3o.utils import disable_torch_init
from blip3o.constants import *
from blip3o.conversation import conv_templates, SeparatorStyle

from diffusers import DiffusionPipeline
from blip3o.mm_utils import get_model_name_from_path
import re, random
import glob


os.environ["TOKENIZERS_PARALLELISM"] = "true"
from accelerate.utils import set_seed

from omegaconf import OmegaConf

def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf

def add_template(prompt):
   conv = conv_templates['qwen'].copy()
   conv.append_message(conv.roles[0], prompt[0])
   conv.append_message(conv.roles[1], None)
   prompt = conv.get_prompt()
   return [prompt]

def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Grid and save images for each prompt
def make_grid(images, grid_size=(2,2)):
    w, h = images[0].size
    grid_img = PIL.Image.new('RGB', (w * grid_size[1], h * grid_size[0]))
    for idx, img in enumerate(images):
        row = idx // grid_size[1]
        col = idx % grid_size[1]
        grid_img.paste(img, (col * w, row * h))
    return grid_img

def main(config, shard_id=0, num_shards=1):
    # If passed along, set the seed now.
    if config.experiment.seed is not None:
        set_seed(config.experiment.seed)

    # Load all prompts
    prompt_files = glob.glob(os.path.join(config.experiment.metadata_file, "*.txt"))
    metadatas = [
        {"prompt": open(pf).read().strip(), "file": pf}
        for pf in prompt_files
    ]

    # Shard the dataset
    total_samples = len(metadatas)
    shard_size = (total_samples + num_shards - 1) // num_shards
    start_idx = shard_id * shard_size
    end_idx = min(start_idx + shard_size, total_samples)
    metadatas = metadatas[start_idx:end_idx]

    # Assign GPU based on shard_id
    device = torch.device(f"cuda:{shard_id % torch.cuda.device_count()}")
    torch.cuda.set_device(device)
    print(f"Shard {shard_id}: Processing {len(metadatas)} prompts on {device}")

    # Load BLIP3o model
    import time
    start_time = time.time()
    model_path = "/scratch-shared/dvarghese/models/hub/models--BLIP3o--BLIP3o-Model-4B/snapshots/5af652f9d947d4128c6fdf9ebad470e17f067fc2/"
    diffusion_path = model_path + "/diffusion-decoder"
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    disable_torch_init()
    model_path = os.path.expanduser(model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, multi_model, context_len = load_pretrained_model(model_path, None, model_name)

    pipe = DiffusionPipeline.from_pretrained(
        diffusion_path,
        custom_pipeline="pipeline_llava_gen",
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        variant="bf16",
        multimodal_encoder=multi_model,
        tokenizer=tokenizer,
        safety_checker=None
    )
    end_time = time.time()
    print(f"Time taken to load model: {end_time - start_time} seconds")

    pipe.vae.to(f'cuda:{shard_id}')
    pipe.unet.to(f'cuda:{shard_id}')
    set_global_seed(seed=42)

    prompts_per_sample = 4  # Always generate 4 images per prompt
    output_dir = config.experiment.output_dir
    os.makedirs(output_dir, exist_ok=True)

    for metadata in tqdm(metadatas, desc=f"GPU {shard_id} Processing batches"):
        pil_images = []
        prompt = metadata['prompt']
        prompt_file = metadata['file']
        prompt_name = os.path.splitext(os.path.basename(prompt_file))[0]
        out_file = os.path.join(output_dir, f"{prompt_name}.png")
        if os.path.exists(out_file):
            continue

        for _ in range(prompts_per_sample):
            pil_images.append(pipe(prompt, guidance_scale=3.0).image)

        grid_img = make_grid(pil_images, grid_size=(2,2))
        grid_img.save(out_file)
    print(f"GPU {shard_id} - Done.")



if __name__ == "__main__":
    config = get_config()

    num_shards = torch.cuda.device_count()
    shard_id = int(os.environ.get("LOCAL_RANK", 0))

    main(config, shard_id, num_shards)
