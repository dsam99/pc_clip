'''
File for performing PC-CLIP finetuning.
'''


import torch
import open_clip
from tqdm import tqdm 
import numpy as np
import torch.nn.functional as F

from data.dataset import get_coco_dataloader, get_coco_rewritten_dataloader
from src.model import load_clip_model
from data.pairwise_datasets import COCO_Comparative_CLIP_Tokens, COCO_Comparative_CLIP_Tokens_HF
from data.dataset import CUBDifferenceClassificationTokens
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CPU = torch.device("cpu")

def clip_loss(image_embed, text_embed, temperature=0.1):
    '''
    Function to compute CLIP loss
    '''
    # normalize embeddings
    image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)
    text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)

    # compute loss
    labels = torch.arange(image_embed.shape[0]).to(device)
    logits = image_embed @ text_embed.t() / temperature

    loss_i = F.cross_entropy(logits, labels, reduction="none")
    loss_t = F.cross_entropy(logits.t(), labels, reduction="none")
    loss_i = loss_i.mean()
    loss_t = loss_t.mean()

    loss = (loss_i + loss_t) / 2
    
    if torch.isnan(loss): # for numerical precision issues
        print("Loss is nan")
        return 0
    
    return loss

def finetune_gt_captions_diff(ft_all=False, openclip=True, use_clip_loss=True):
    '''
    Function to finetune CLIP on the COCO ground truth captions and the LLM generated differences
    '''

    model, _ = load_clip_model("ViT-L/14", openclip=openclip) 

    clip_tokenizer = open_clip.get_tokenizer("ViT-L-14")
    lr = 1e-8
    epochs=20
    batch_size = 512

    # define optimizer for textual params
    optimizer = torch.optim.Adam(model.transformer.parameters(), lr=lr)

    # lr scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    dataset = COCO_Comparative_CLIP_Tokens("./data/CLIP_ViT-L_14_train.pkl",
                                            comparatives_path="llm_diffs/coco_diffs_1000_filtered_v2.pkl")
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # define MSE loss function
    loss_func = torch.nn.MSELoss()

    # if clip model -> convert to float
    if not openclip:
        model = model.float()

    for epoch_ind in tqdm(range(epochs), total=epochs):

        avg_loss = 0

        for i, (_, prefix1, _, prefix2, comp_tokens) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            model.zero_grad()
            prefix1, prefix2 = prefix1.to(device), prefix2.to(device)

            if not openclip:
                prefix1 = prefix1.float()
                prefix2 = prefix2.float()

            else:
                prefix1 = prefix1.float()
                prefix2 = prefix2.float()

            # normalize all embeddings
            prefix1 = prefix1 / prefix1.norm(dim=-1, keepdim=True)
            prefix2 = prefix2 / prefix2.norm(dim=-1, keepdim=True)

            # encode comp_tokens
            comp_tokens = comp_tokens.to(device)
            comp_tokens_clip = model.encode_text(comp_tokens)

            # normalize
            comp_tokens_clip = comp_tokens_clip / comp_tokens_clip.norm(dim=-1, keepdim=True)

            if not use_clip_loss:
                loss = loss_func(comp_tokens_clip, prefix1 - prefix2)
            else:    
                loss = clip_loss(comp_tokens_clip, prefix1 - prefix2, temperature=1)
                # loss = clip_loss(comp_tokens_clip, prefix1 - prefix2, temperature=0.5)
                
            # skip if loss is 0
            if loss == 0:
                pass
            else:
                loss.backward()
                optimizer.step()

                avg_loss += loss.item()
            
            # free memory
            del prefix1, prefix2, comp_tokens, comp_tokens_clip
            torch.cuda.empty_cache()


        # save model per epoch
        if epoch_ind in [9, 19]:   
            if openclip:
                if ft_all:
                    if use_clip_loss:
                        torch.save(model.state_dict(), f"checkpoints/pc_clip/clip-loss-coco-diff-all-batch-{batch_size}-lr-{lr}-sched-{epoch_ind + 1}-epoch.pt")
                    else:
                        torch.save(model.state_dict(), f"checkpoints/pc_clip/coco-diff-all-lr-{lr}-sched-{epoch_ind + 1}-epoch.pt")
                else:
                    if use_clip_loss:
                        torch.save(model.state_dict(), f"checkpoints/pc_clip_test/clip-loss-coco-diff-text-batch-{batch_size}-lr-{lr}-sched-{epoch_ind + 1}-epoch.pt")
                    else:
                        torch.save(model.state_dict(), f"checkpoints/pc_clip/mseloss-coco-diff-text-batch-{batch_size}-lr-{lr}-sched-{epoch_ind + 1}-epoch.pt")
            else:
                torch.save(model.state_dict(), f"checkpoints/clip/coco-batch-{batch_size}-lr-{lr}-sched-{epoch_ind + 1}-epoch.pt")
        scheduler.step()

        print("Epoch: " + str(epoch_ind) + " " + "Loss: " + str(avg_loss / len(train_dataloader)))


def finetune_coco():
    '''
    Function to finetune CLIP on the COCO ground truth captions
    '''

    model, preprocess = load_clip_model("ViT-L/14", openclip=True)
    num_epochs = 50
    lr = 1e-6
    batch_size = 128

    train_loader = get_coco_dataloader(batch_size=batch_size)
    optimizer = torch.optim.Adam(model.transformer.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    for epoch in tqdm(range(num_epochs), total=num_epochs):
        for i, (image_embs, captions) in enumerate(train_loader):

            optimizer.zero_grad()

            image_embs = image_embs.to(device).float()
            captions = captions.to(device)

            # get image features
            text_features = model.encode_text(captions)
            loss = clip_loss(image_embs, text_features)

            # skip if loss is nan
            if loss == 0:
                continue
            

            # backprop
            loss.backward()
            optimizer.step()

            # if i % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

        if epoch in [9, 19, 29, 49]:
            torch.save(model.state_dict(), f"checkpoints/coco_ft/coco-batch-{batch_size}-lr-{lr}-sched-{epoch + 1}-epoch.pt")
        scheduler.step()

def finetune_coco_rewritten():

    '''
    Function to finetune CLIP on the COCO rewritten captions
    '''

    model, preprocess = load_clip_model("ViT-L/14", openclip=True)
    num_epochs = 20
    lr = 1e-6
    batch_size = 128

    train_loader = get_coco_rewritten_dataloader(batch_size=batch_size)
    optimizer = torch.optim.Adam(model.transformer.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    for epoch in tqdm(range(num_epochs), total=num_epochs):
        for i, (image_embs, captions) in enumerate(train_loader):

            optimizer.zero_grad()

            image_embs = image_embs.to(device).float()
            captions = captions.to(device)

            # get image features
            text_features = model.encode_text(captions)
            loss = clip_loss(image_embs, text_features)

            # skip if loss is nan
            if loss == 0:
                continue
            
            # backprop
            loss.backward()
            optimizer.step()

            # if i % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

        if epoch in [9, 19]:
            torch.save(model.state_dict(), f"checkpoints/coco_rewrite_ft/coco-batch-{batch_size}-lr-{lr}-sched-{epoch + 1}-epoch.pt")
        scheduler.step()

if __name__ == "__main__":

    # set random seeds
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # baselines
    # finetune_coco()
    # finetune_coco_rewritten()

    # OUR METHOD
    finetune_gt_captions_diff(ft_all=False, openclip=True, use_clip_loss=True)