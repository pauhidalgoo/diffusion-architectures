import os
from datasets import load_dataset
from PIL import Image

import torch
from torchvision import transforms
from diffusers import AutoencoderDC
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader


from tqdm import tqdm
# would be nice to visualize progress :)

from config import Config
Cfg = Config()

transform = transforms.Compose([
    transforms.Resize((Cfg.image_size, Cfg.image_size)),
    transforms.ToTensor(),
])


def make_dataset_subset(max_items=2000):
    dset = load_dataset(Cfg.dataset)
    split = 'train' if 'train' in dset else list(dset.keys())[0] # If dataset doesn't have train split use first one
    ds = dset[split]
    # Note: this may not be viable for large datasets, may change
    ds = ds.shuffle(seed=Cfg.seed).select(range(min(len(ds), max_items)))
    return ds





def preprocess_and_precompute(save_dir='precomputed', max_items=2000, device='cuda'):
    os.makedirs(save_dir, exist_ok=True)
    ds = make_dataset_subset(max_items)

    ae = AutoencoderDC.from_pretrained(
        "mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers",
        torch_dtype=torch.bfloat16 if Cfg.model_dtype == "bfloat16" else torch.float32,
                                       ).to(device).eval()
    
    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B",
                                model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"}, tokenizer_kwargs={"padding_side": "left"},
                                device=device).eval()

    dataloader = DataLoader(ds, batch_size=Cfg.precompute_batch, shuffle=False)

    all_img_latents = []
    all_text_embeddings = []

    for batch in dataloader:
        imgs = [transform(Image.fromarray(img)) for img in batch["image"]]
        imgs = torch.stack(imgs).to(device, dtype=torch.float16)
        with torch.no_grad():
            latents = ae.encode(imgs).latent_dist.sample() # (B, latent_dim, H, W)
            all_img_latents.append(latents.cpu())

        captions = batch["caption"]
        with torch.no_grad():
            text_embeds = model.encode(captions, convert_to_tensor=True, device=device)
            all_text_embeddings.append(text_embeds.cpu())

    all_img_latents = torch.cat(all_img_latents, dim=0)
    all_text_embeddings = torch.cat(all_text_embeddings, dim=0)

    torch.save(all_img_latents, os.path.join(save_dir, f"{Cfg.dataset}_latents.pt"))
    torch.save(all_text_embeddings, os.path.join(save_dir, f"{Cfg.dataset}_text_embeddings.pt"))

    print(f"Saved image latents and text embeddings for {Cfg.dataset} to {save_dir}")
    print('Saved', latents.shape)
    return os.path.join(save_dir, 'dataset_latents.pt')
