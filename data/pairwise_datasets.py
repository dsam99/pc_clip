import torch
import open_clip
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import sys 
import pickle
from typing import Tuple
import clip

import skimage.io as io
from PIL import Image

from transformers import CLIPTokenizerFast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CPU = torch.device("cpu")

class COCO_Comparative_CLIP_Tokens(Dataset):

    '''
    Dataset of COCO comparatives: (fixed) prefix and comparative tokens -> used for finetuning text only
    '''

    def __len__(self) -> int:
        return len(self.pair_inds)


    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        
        ind1, ind2 = self.pair_inds[item]
        ind1, ind2 = self.ind_map.index(ind1), self.ind_map.index(ind2)
        
        tokens1 = self.captions_tokens[ind1]
        tokens2 = self.captions_tokens[ind2]
        prefix1 = self.prefixes[ind1]
        prefix2 = self.prefixes[ind2]

        # return tokenized comparative
        comp_tokens = self.comparatives_tokens[item]

        return tokens1, prefix1, tokens2, prefix2, comp_tokens 

    def __init__(self, data_path: str, comparatives_path="llm_diffs_v2_filtered.pkl", openclip=True, hf=False):
        
        comp_dict = pickle.load(open(comparatives_path, "rb"))
        self.pair_inds = comp_dict["indices"]
        self.comparatives = comp_dict["llm_diffs"]
        
        if openclip:
            self.tokenizer = open_clip.get_tokenizer("ViT-L-14")
        elif hf:
            self.tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-large-patch14")

        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)

        print("Data size is %0d" % len(all_data["clip_embedding"]))
        sys.stdout.flush()

        self.prefixes = all_data["clip_embedding"]
        captions_raw = all_data["captions"]
        self.image_ids = [caption["image_id"] for caption in captions_raw]
        self.ind_map = all_data["inds"] # list of indices of used images

        if os.path.isfile(f"{data_path[:-4]}_tokens.pkl"):
            with open(f"{data_path[:-4]}_clip_tokens.pkl", 'rb') as f:
                self.captions_tokens, self.caption2embedding, self.max_seq_len = pickle.load(f)
        else:
            self.captions_tokens = []
            self.caption2embedding = []
            max_seq_len = 0
            for caption in tqdm(captions_raw, total=len(captions_raw)):
                if openclip:
                    tokens = self.tokenizer(caption['caption'])[0]
                elif hf:
                    tokens = self.tokenizer(caption['caption'], return_tensors="pt").input_ids[0]
                else:
                    tokens = clip.tokenize(caption['caption'])[0]
                self.captions_tokens.append(tokens)
                self.caption2embedding.append(caption["clip_embedding"])
            with open(f"{data_path[:-4]}_clip_tokens.pkl", 'wb') as f:
                pickle.dump([self.captions_tokens, self.caption2embedding, max_seq_len], f)
        
        # turn comparatives into tokens
        self.comparatives_tokens = []
        for comp in tqdm(self.comparatives):
            if openclip:
                comp_tokens = self.tokenizer(comp)[0]
            else:
                comp_tokens = clip.tokenize(comp)[0]
            self.comparatives_tokens.append(comp_tokens)



class COCO_Comparative_CLIP_Tokens_HF(Dataset):

    '''
    Dataset of COCO comparatives: (fixed) prefix and comparative tokens -> used for finetuning text only
    
    compatible with huggingface model version -> here we only store used images in data and load this as a dict
    since otherwise there are memory issues
    
    '''

    def __len__(self) -> int:
        return len(self.pair_inds)


    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        
        ind1, ind2 = self.pair_inds[item]

        data_inds1, data_inds2 = self.ind_map.index(ind1), self.ind_map.index(ind2)

        tokens1 = self.captions_tokens[data_inds1]
        tokens2 = self.captions_tokens[data_inds2]

        prefix1 = self.prefixes[data_inds1]
        prefix2 = self.prefixes[data_inds2]

        # return tokenized comparative
        comp_tokens = self.comparatives_tokens[item]        
        return tokens1, prefix1, tokens2, prefix2, comp_tokens 

    def __init__(self, data_path: str, comparatives_path="llm_diffs_v2_filtered.pkl"):
        
        comp_dict = pickle.load(open(comparatives_path, "rb"))
        self.pair_inds = comp_dict["indices"]
        self.comparatives = comp_dict["llm_diffs"]
        self.tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-large-patch14")

        print("loaded model")

        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)

        print("Data size is %0d" % len(all_data["clip_embedding"]))
        sys.stdout.flush()
        self.prefixes = all_data["clip_embedding"]
        captions_raw = all_data["captions"]
        self.image_ids = [caption["image_id"] for caption in captions_raw]
        self.captions = [caption['caption'] for caption in captions_raw]

        self.ind_map = all_data["inds"] # list of indices of used images

        if os.path.isfile(f"{data_path[:-4]}_tokens.pkl"):
            with open(f"{data_path[:-4]}_clip_tokens.pkl", 'rb') as f:
                self.captions_tokens, self.caption2embedding, self.max_seq_len = pickle.load(f)
        else:
            self.captions_tokens = []
            self.caption2embedding = []
            max_seq_len = 0
            for caption in tqdm(captions_raw, total=len(captions_raw)):
                tokens = self.tokenizer(caption['caption'], return_tensors="pt", truncation=True, padding="max_length", max_length=64).input_ids[0]
                self.captions_tokens.append(tokens)
                self.caption2embedding.append(caption["clip_embedding"])
            with open(f"{data_path[:-4]}_clip_tokens.pkl", 'wb') as f:
                pickle.dump([self.captions_tokens, self.caption2embedding, max_seq_len], f)
        
        # turn comparatives into tokens
        self.comparatives_tokens = []
        for comp in tqdm(self.comparatives):
            
            comp_tokens = self.tokenizer(comp, padding="max_length", return_tensors="pt", max_length=64, truncation=True).input_ids[0]
            self.comparatives_tokens.append(comp_tokens)



class COCO_Comparative_CLIP_Tokens_Image(Dataset):

    '''
    Dataset of images + comparatives -> used for finetuning both image and text encoders / not required
    for the results in our paper
    '''

    def __len__(self) -> int:
        return len(self.pair_inds)
    
    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
            
            ind1, ind2 = self.pair_inds[item]

            tokens1 = self.captions_tokens[ind1]
            tokens2 = self.captions_tokens[ind2]
            
            im1 = self.images[self.caption2embedding[ind1]]
            im2 = self.images[self.caption2embedding[ind2]]

            im1 = io.imread(im1)
            im2 = io.imread(im2)
    
            image1 = self.preprocess(Image.fromarray(im1)).to(device)
            image2 = self.preprocess(Image.fromarray(im2)).to(device)

            # return tokenized comparative
            comp_tokens = self.comparatives_tokens[item]
            return tokens1, image1, tokens2, image2, comp_tokens

    def __init__(self, data_path: str, comparatives_path="llm_diffs_v2_filtered.pkl", openclip=True):

        comp_dict = pickle.load(open(comparatives_path, "rb"))
        self.pair_inds = comp_dict["indices"]
        self.comparatives = comp_dict["llm_diffs"]

        _, _, self.preprocess = open_clip.create_model_and_transforms("ViT-L/14", pretrained=None)

        if openclip:
            self.tokenizer = open_clip.get_tokenizer("ViT-L-14")
    
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        print("Data size is %0d" % len(all_data["clip_embedding"]))
        sys.stdout.flush()
        self.images = all_data["clip_embedding"]
        captions_raw = all_data["captions"]
        self.image_ids = [caption["image_id"] for caption in captions_raw]
        self.captions = [caption['caption'] for caption in captions_raw]


        self.images = []
        for img_id in tqdm(self.image_ids, total=len(self.image_ids)):
            filename = f"./CLIP_prefix_caption/data/coco/train2014/COCO_train2014_{int(img_id):012d}.jpg"
            if not os.path.isfile(filename):
                filename = f"./CLIP_prefix_caption/data/coco/val2014/COCO_val2014_{int(img_id):012d}.jpg"
            # image = io.imread(filename)
            # self.images.append(image)
            self.images.append(filename)

        if os.path.isfile(f"{data_path[:-4]}_tokens.pkl"):
            with open(f"{data_path[:-4]}_clip_tokens.pkl", 'rb') as f:
                self.captions_tokens, self.caption2embedding, self.max_seq_len = pickle.load(f)
        else:
            self.captions_tokens = []
            self.caption2embedding = []
            max_seq_len = 0
            for caption in captions_raw:
                
                if openclip:
                    tokens = self.tokenizer(caption['caption'])[0]
                else:
                    tokens = clip.tokenize(caption['caption'])[0]
                self.captions_tokens.append(tokens)
                self.caption2embedding.append(caption["clip_embedding"])
            with open(f"{data_path[:-4]}_clip_tokens.pkl", 'wb') as f:
                pickle.dump([self.captions_tokens, self.caption2embedding, max_seq_len], f)
        
        # turn comparatives into tokens
        self.comparatives_tokens = []
        for comp in tqdm(self.comparatives):
            if openclip:
                comp_tokens = self.tokenizer(comp)[0]
            else:
                comp_tokens = clip.tokenize(comp)[0]
            self.comparatives_tokens.append(comp_tokens)