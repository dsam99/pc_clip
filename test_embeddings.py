'''
File for running analysis on learned embeddings from PC-CLIP.
'''

import torch
import open_clip
from src.model import load_clip_model
from dataset import get_dataloaders, get_comparative_dataloaders, get_difference_dataloaders
from caption_finetune import COCO_Comparative
from tqdm import tqdm
import numpy as np

from utils import get_comparatives
from dataset import get_class_labels

def test_embeddings(model):
    _, preprocess = load_clip_model("ViT-L/14", openclip=True) 
    clip_tokenizer = open_clip.get_tokenizer("ViT-L-14")

    # looping through options
    dataset = COCO_Comparative("./data/oscar_split_ViT-L_14_train.pkl", prefix_length=40)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    

    diff_im = 0
    diff_text = 0

    for i in tqdm(range(len(dataset))):

        with torch.no_grad():
            comparative = dataset.comparatives[i]
            inds = dataset.pair_inds[i]
            image1, image2 = dataset.prefixes[dataset.caption2embedding[inds[0]]], dataset.prefixes[dataset.caption2embedding[inds[1]]]
            caption1, caption2 = dataset.captions[inds[0]], dataset.captions[inds[1]]

            # generate text embeddings
            text1, text2 = clip_tokenizer(caption1), clip_tokenizer(caption2)
            text1, text2 = text1.to(device), text2.to(device)
            text1, text2 = model.encode_text(text1), model.encode_text(text2)

            im_diff = image1 - image2
            im_diff = im_diff.to(device)
            text_diff = text1 - text2

            # generate embedding for comparative
            comp = clip_tokenizer(comparative)
            comp = comp.to(device)
            comp = model.encode_text(comp)

            # normalize
            comp = comp / torch.norm(comp)
            im_diff = im_diff / torch.norm(im_diff)
            text_diff = text_diff / torch.norm(text_diff)

            # print difference in text space and in image space form the comparative in terms of cosine similarity
            diff_comp_im_diff = 1 - torch.cosine_similarity(comp, im_diff)
            diff_comp_text_diff = 1 - torch.cosine_similarity(comp, text_diff)

            diff_im += diff_comp_im_diff.item()
            diff_text += diff_comp_text_diff.item()

    print("Diff comp and image diff", diff_im / len(dataset))
    print("Diff comp and text diff", diff_text / len(dataset))

def test_comparative_prompts(checkpoint, dataset, weight_avg=False):

    class_labels = get_class_labels(dataset)
    comparatives = get_comparatives(dataset)
    
    model, preprocess = load_clip_model("ViT-L/14", openclip=True, checkpoint=checkpoint, weight_avg=weight_avg)    

    clip_tokenizer = open_clip.get_tokenizer("ViT-L-14")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    prompts = ["A photo of a {}".format(x) for x in class_labels]
    prompt_embeddings = model.encode_text(clip_tokenizer(prompts).to(device))

    # getting comparative embeddings
    comparatives_embeddings = [(pair, model.encode_text(clip_tokenizer(x).to(device))) for (pair, x) in comparatives]

    avg_diff = []
    avg_diff_rev = []


    for pair, embed in comparatives_embeddings:
        print(pair)

        # see how far the comparative is from the prompt in square distance
        prompt_diff = prompt_embeddings[pair[0]] - prompt_embeddings[pair[1]]
        prompt_diff = prompt_diff / torch.norm(prompt_diff)

        embed = embed / torch.norm(embed)

        # compute cosine similarity 
        avg_diff.append(1 - torch.cosine_similarity(prompt_diff, embed).item())
        avg_diff_rev.append(1 - torch.cosine_similarity(-prompt_diff, embed).item())

    print("Avg cosine dist", sum(avg_diff) / len(avg_diff))
    print("Avg cosine dist reverse", sum(avg_diff_rev) / len(avg_diff_rev))

def test_text_differences(checkpoint, dataset, weight_avg=False):

    class_labels = get_class_labels(dataset)    
    model, preprocess = load_clip_model("ViT-L/14", openclip=True, checkpoint=checkpoint, weight_avg=weight_avg)    
    clip_tokenizer = open_clip.get_tokenizer("ViT-L-14")
    test_loader = get_difference_dataloaders(dataset, preprocess, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    prompts = ["A photo of a {}".format(x) for x in class_labels]
    prompt_embeddings = model.encode_text(clip_tokenizer(prompts).to(device))

    # getting comparative embeddings=
    avg_diff = []
    avg_diff_rev = []

    for _, (_, _, difference_tokens, label1, label2) in tqdm(enumerate(test_loader)):

        # see how far the comparative is from the prompt in square distance
        prompt_diff = prompt_embeddings[label1] - prompt_embeddings[label2]
        prompt_diff = prompt_diff / torch.norm(prompt_diff)

        embed = model.encode_text(difference_tokens.to(device))
        embed = embed / torch.norm(embed)

        # compute cosine similarity
        avg_diff.append(1 - torch.cosine_similarity(prompt_diff, embed).item())
        avg_diff_rev.append(1 - torch.cosine_similarity(-prompt_diff, embed).item())

    print("Avg cosine distance", sum(avg_diff) / len(avg_diff))
    print("Avg cosine distance reverse", sum(avg_diff_rev) / len(avg_diff_rev))

def test_linear_probe(checkpoint, dataset, weight_avg=False):

    class_labels = get_class_labels(dataset)
    model, preprocess = load_clip_model("ViT-L/14", openclip=True, checkpoint=checkpoint, weight_avg=weight_avg)    
    
    clip_tokenizer = open_clip.get_tokenizer("ViT-L-14")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # num_shots = 2
    num_shots = 5


    train_dataset, test_dataset = get_dataloaders(dataset, preprocess, batch_size=32, num_shots=num_shots)

    seeds = 5
    accs = []
    for i in range(seeds):

        # train linear head off of models image encoder
        linear_head = torch.nn.Linear(768, len(class_labels))
        linear_head.to(device)

        epochs = 10
        lr = 0.001

        optimizer = torch.optim.Adam(linear_head.parameters(), lr=lr)

        for epoch in range(epochs):
            for _, (images, labels) in enumerate(train_dataset):
                images = images.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    image_embeddings = model.encode_image(images)

                logits = linear_head(image_embeddings)
                loss = torch.nn.functional.cross_entropy(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print("Epoch: %d, Loss: %f" % (epoch, loss.item()))

        # test linear head
        correct = 0
        total = 0

        with torch.no_grad():
            for _, (images, labels) in enumerate(test_dataset):
                images = images.to(device)
                labels = labels.to(device)

                image_embeddings = model.encode_image(images)
                logits = linear_head(image_embeddings)
                predictions = torch.argmax(logits, dim=1)

                correct += torch.sum(predictions == labels).item()
                total += len(labels)

        print("Accuracy: %f" % (correct / total))
        accs.append(correct / total)

    print("Average Accuracy: %f" % (sum(accs) / len(accs)))
    print("Std Err", np.std(accs) / np.sqrt(len(accs)))

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="embeddings", help="diff, comparative, embeddings")
    parser.add_argument("--dataset", type=str, default="cifar10", help="cifar10, cifar100")
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint to load")
    parser.add_argument("--weight_avg", action="store_true", help="weight average embeddings")
    args = parser.parse_args()


    print("Dataset:", args.dataset)
    print("Checkpoint:", args.checkpoint)
    if args.mode == "diff":
        print("Running Text Differences")
        test_text_differences(checkpoint=args.checkpoint, dataset=args.dataset, weight_avg=args.weight_avg)
    elif args.mode == "comparative":
        print("Running Comparatives")
        test_comparative_prompts(checkpoint=args.checkpoint, dataset=args.dataset, weight_avg=args.weight_avg)
    elif args.mode == "lp":
        print("Linear Probe")
        test_linear_probe(checkpoint=args.checkpoint, dataset=args.dataset, weight_avg=args.weight_avg)