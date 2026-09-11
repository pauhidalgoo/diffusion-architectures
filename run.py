import argparse, os
from data import preprocess_and_precompute
from config import Config
Cfg = Config()

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['precompute','train','sample'], required=True)
parser.add_argument('--latents_path', type=str, default='precomputed/dataset_latents.pt')
args = parser.parse_args()

if args.mode == 'precompute':
    path = preprocess_and_precompute(save_dir='precomputed', max_items=2000)
    print('precomputed latents saved to', path)