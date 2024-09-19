import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import os
from tqdm import tqdm
import argparse

import open_clip
from dataset import load_dataset, CUBDataset

def main():
    device = torch.device('cuda:0')
    clip_model_name = "ViT-L/14" 
    
    # out_path = f"./CLIP_prefix_caption/data/birds/ViT-L_14_train.pkl"
    out_path = f"./CLIP_prefix_caption/data/birds/ViT-L_14_test.pkl"

    clip_model, _, preprocess = open_clip.create_model_and_transforms(clip_model_name, pretrained="datacomp_xl_s13b_b90k")
    clip_model = clip_model.to(device)
    
    all_embeddings = []
    all_captions = []
    all_labels = []

    # LOAD CUB DATASET
    cub_test_dataset = CUBDataset(preprocess, split="test")
    batch_size = 128

    # for i in tqdm(range(0, len(cub_train_dataset), batch_size)):
    for i in tqdm(range(0, len(cub_test_dataset), batch_size)):
        
        # d = [cub_train_dataset[j][0] for j in range(i, min(i + batch_size, len(cub_train_dataset)))]
        d = [cub_test_dataset[j][0] for j in range(i, min(i + batch_size, len(cub_test_dataset)))]
        d = torch.stack(d).to(device)
        with torch.no_grad():
            prefix = clip_model.encode_image(d).cpu()
        
        all_embeddings.append(prefix)
        # labels = [cub_train_dataset[j][1] for j in range(i, min(i + batch_size, len(cub_train_dataset)))]
        labels = [cub_test_dataset[j][1] for j in range(i, min(i + batch_size, len(cub_test_dataset)))]
        # all_captions.append(d["description"])
        
        # all_captions += [cub_train_dataset.captions[j] for j in range(i, min(i + batch_size, len(cub_train_dataset)))]
        all_captions += [cub_test_dataset.captions[j] for j in range(i, min(i + batch_size, len(cub_test_dataset)))]
        
        all_labels += labels
        
        if (i + 1) % 10000 == 0:
            with open(out_path, 'wb') as f:
                pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions, "labels": torch.tensor(all_labels)}, f)

    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions, "labels": torch.tensor(all_labels)}, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    return 0


if __name__ == '__main__':
    main()