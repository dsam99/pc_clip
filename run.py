'''
File for computing accuracy of PC-CLIP models.
'''

import torch
import clip
from tqdm import tqdm
import numpy as np
import open_clip

from data.dataset import get_dataloaders, get_comparative_dataloaders, get_difference_dataloaders
from src.model import load_clip_model, load_llm
from src.utils import get_extended_class_labels, get_descriptions, get_comparatives
from data.dataset import get_class_labels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=np.inf)

def run_zero_shot(dataset, openclip=False, checkpoint=None, weight_avg=False, weight_alpha=None):

    # load model
    model, preprocess = load_clip_model("ViT-L/14", openclip=openclip, checkpoint=checkpoint, weight_avg=weight_avg, weight_alpha=weight_alpha) 
    _, test_loader = get_dataloaders(dataset, preprocess, batch_size=256)

    # get class labels
    class_labels = get_class_labels(dataset)
    prompts = [f"This is a photo of a {label}" for label in class_labels]

    # encode prompts
    if openclip:
        tokenizer = open_clip.get_tokenizer("ViT-L-14")
        prompts_token = tokenizer(prompts).to(device)
    else:
        prompts_token = clip.tokenize(prompts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(prompts_token)
        text_features /= text_features.norm(dim=-1, keepdim=True) # normalize tensors

    # evaluate zero-shot accuracy on cifar10 test set
    correct = 0
    total = 0

    # check confusion matrix
    confusion_matrix = torch.zeros((len(prompts), len(prompts)))

    for _, (images, labels) in tqdm(enumerate(test_loader), total=len(test_loader)):
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            probs = (image_features @ text_features.T).softmax(dim=-1).cpu().numpy()
            predicted_labels = probs.argmax(axis=1)
            correct += (predicted_labels == labels.cpu().numpy()).sum()
            total += len(labels)

            # add to confusion matrix
            for i in range(len(labels)):
                confusion_matrix[labels[i]][predicted_labels[i]] += 1
    
    print("Prompts Zeroshot Accuracy: ", correct / total)
    if len(class_labels) < 11:
       print(confusion_matrix)

    # find 3 largest values in confusion matrix that are not along diagonal
    cm_copy = confusion_matrix.clone()
    for i in range(len(class_labels)):
        cm_copy[i][i] = 0
    
    print("Top 3 Confusions")
    for i in range(3):
        max_val = cm_copy.max()
        max_index = torch.argmax(cm_copy)
        row = max_index // len(class_labels)
        col = max_index % len(class_labels)
        print("c1", class_labels[row], "c2", class_labels[col], "inds", row, col)
        cm_copy[row][col] = 0

    # compute top-3 worst group accuracy and corresponding names of classes
    group_accs = []
    for i in range(len(class_labels)):
        group_acc = confusion_matrix[i][i] / confusion_matrix[i].sum()
        group_accs.append(group_acc)
    group_accs = np.array(group_accs)
    sorted_group_accs = np.sort(group_accs)
    print("Top-3 Worst Group Accuracies: ", sorted_group_accs[:3])

    # get indices of top-3 worst groups and print names
    top_3_worst_group_indices = np.argsort(group_accs)[:3]
    print("Top-3 Worst Group Indices: ", top_3_worst_group_indices)
    print("Top-3 Worst Group Names: ", [class_labels[i] for i in top_3_worst_group_indices])

    # print best group accuracy and index and name
    print("Best Group Accuracy: ", sorted_group_accs[-1])
    best_group_index = np.argmax(group_accs)
    print("Best Group Index: ", best_group_index)
    print("Best Group Name: ", class_labels[best_group_index])

    return correct / total



def run_comparative(dataset, openclip=True, checkpoint=None, alpha=0.8, weight_avg=False, weight_alpha=None):

    # load model
    model, preprocess = load_clip_model("ViT-L/14", openclip=openclip, checkpoint=checkpoint, weight_avg=weight_avg, weight_alpha=weight_alpha) 
    _, test_loader = get_dataloaders(dataset, preprocess, batch_size=32)

    class_labels = get_class_labels(dataset)
    prompts = [f"This is a photo of a {label}" for label in class_labels]

    # get comparatives
    if checkpoint is not None:
        comparative_prompts = get_comparatives(dataset, ft=True) 
    else:
        comparative_prompts = get_comparatives(dataset)
        
    # encode prompts
    if openclip:
        tokenizer = open_clip.get_tokenizer("ViT-L-14")
        prompts_token = tokenizer(prompts).to(device)
    else:
        prompts_token = clip.tokenize(prompts).to(device)

    text_features = model.encode_text(prompts_token)
    text_features /= text_features.norm(dim=-1, keepdim=True) # normalize tensors

    # loop through and encode comparatives
    comparative_features = []
    for _, comparative in comparative_prompts:
        if openclip:
            comparative_token = tokenizer(comparative).to(device)
        else:
            comparative_token = clip.tokenize(comparative).to(device)
        comparative_features.append(model.encode_text(comparative_token))
    comparative_features = torch.cat(comparative_features, dim=0)
    comparative_features /= comparative_features.norm(dim=-1, keepdim=True) # normalize tensors

    # average comparative - other class with original class embedding
    sum_comparative_features = torch.zeros_like(text_features)
    class_counts = np.zeros(len(class_labels))
    for index, (pair, _) in enumerate(comparative_prompts):
        sum_comparative_features[pair[1]] += text_features[pair[0]] - comparative_features[index]
        class_counts[pair[1]] += 1
    
    # add one to class_counts at 0 indices so that divide by 0 error does not happen
    class_counts[class_counts == 0] = 1

    # average  
    sum_comparative_features /= torch.tensor(class_counts).unsqueeze(-1).to(device)
    comp_text_features = (alpha) * text_features + (1 - alpha) * sum_comparative_features
    comp_text_features /= comp_text_features.norm(dim=-1, keepdim=True) # normalize tensors

    # get standard zs performance
    zs_text_features = text_features / text_features.norm(dim=-1, keepdim=True) # normalize tensors

    # evaluate zero-shot accuracy on cifar10 test set
    correct = 0
    total = 0

    # check confusion matrix
    confusion_matrix = torch.zeros((len(prompts), len(prompts)))

    all_labels = []
    all_predictions = []
    all_zs_predictions = []
    for _, (images, labels) in tqdm(enumerate(test_loader)):
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            probs = (image_features @ comp_text_features.T).softmax(dim=-1).cpu().numpy()
            predicted_labels = probs.argmax(axis=1)

            zs_probs = (image_features @ zs_text_features.T).softmax(dim=-1).cpu().numpy()
            zs_predicted_labels = zs_probs.argmax(axis=1)

            correct += (predicted_labels == labels.cpu().numpy()).sum()
            total += len(labels)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted_labels)
            all_zs_predictions.extend(zs_predicted_labels)

            # add to confusion matrix
            for i in range(len(labels)):
                confusion_matrix[labels[i]][predicted_labels[i]] += 1
    
    # convert predictions and labels to numpy
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_zs_predictions = np.array(all_zs_predictions)

    print("ZS Accuracy: ", (all_labels == all_zs_predictions).sum() / len(all_labels))
    print("Comparative Accuracy: ", correct / total)

    # compute worst case accuracy for both zs and comparative
    worst_case_acc_zs = 1
    worst_case_acc_comp = 1

    for i in range(len(class_labels)):
        i_inds = (all_labels == i)
        zs_group_acc = (all_labels[i_inds] == all_zs_predictions[i_inds]).sum() / len(all_labels[i_inds])
        comp_group_acc = (all_labels[i_inds] == all_predictions[i_inds]).sum() / len(all_labels[i_inds])

        print("zs", zs_group_acc)
        print("comp", comp_group_acc)

        if zs_group_acc < worst_case_acc_zs:
            worst_case_acc_zs = zs_group_acc
        if comp_group_acc < worst_case_acc_comp:
            worst_case_acc_comp = comp_group_acc

    print("Worst Case ZS Accuracy: ", worst_case_acc_zs)
    print("Worst Case Comparative Accuracy: ", worst_case_acc_comp)

def run_comparative_best(dataset, openclip=True, checkpoint=None, weight_avg=False, weight_alpha=None):
    '''
    Function to run comparatives with using comparative leveraging best class and 3 worst classes
    '''

    # load model
    model, preprocess = load_clip_model("ViT-L/14", openclip=openclip, checkpoint=checkpoint, weight_avg=weight_avg, weight_alpha=weight_alpha)
    _, test_loader = get_dataloaders(dataset, preprocess, batch_size=32)

    class_labels = get_class_labels(dataset)
    prompts = [f"This is a photo of a {label}" for label in class_labels]

    # get comparatives
    # if checkpoint is None:
        # comparative_prompts = get_comparatives_best(dataset, ft=False)
    # else:
        # comparative_prompts = get_comparatives_best(dataset)
    comparative_prompts = None

    # encode prompts
    if openclip:

        tokenizer = open_clip.get_tokenizer("ViT-L-14")
        prompts_token = tokenizer(prompts).to(device)

    else:
        prompts_token = clip.tokenize(prompts).to(device)

    text_features = model.encode_text(prompts_token)
    text_features /= text_features.norm(dim=-1, keepdim=True) # normalize tensors

    # loop through and encode comparatives
    comparative_features = []

    for _, comparative in comparative_prompts:

        if openclip:
            comparative_token = tokenizer(comparative).to(device)
        else:
            comparative_token = clip.tokenize(comparative).to(device)

        comparative_features.append(model.encode_text(comparative_token))

    comparative_features = torch.cat(comparative_features, dim=0)
    comparative_features /= comparative_features.norm(dim=-1, keepdim=True) # normalize tensors

    # comparative_prompts given format of (best_class_index, worst_class_index), str(difference between best - worst)
    # replace prompt of worst_class_index with prompt of best_class_index - difference between best - worst    
    for j, ((best_ind, worst_ind), _) in enumerate(comparative_prompts):
        text_features[worst_ind] = text_features[best_ind] - comparative_features[j]

    correct = 0
    total = 0
    confusion_matrix = torch.zeros((len(prompts), len(prompts)))

    all_labels = []
    all_predictions = []

    for _, (images, labels) in tqdm(enumerate(test_loader)):
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            probs = (image_features @ text_features.T).softmax(dim=-1).cpu().numpy()
            predicted_labels = probs.argmax(axis=1)

            correct += (predicted_labels == labels.cpu().numpy()).sum()
            total += len(labels)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted_labels)

            # add to confusion matrix
            for i in range(len(labels)):
                confusion_matrix[labels[i]][predicted_labels[i]] += 1


    # convert predictions and labels to numpy
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    print("Comparative Accuracy: ", correct / total)
    if len(class_labels) < 11:
         print(confusion_matrix)


def run_extended(dataset, checkpoint=None, weight_avg=False, weight_alpha=None):
    '''
    Function to run CLIP models with extended class descriptions
    '''

    # load model
    model, preprocess = load_clip_model("ViT-L/14", openclip=True, checkpoint=checkpoint, weight_avg=weight_avg, weight_alpha=weight_alpha) 
    train_loader, test_loader = get_dataloaders(dataset, preprocess, batch_size=32)

    class_labels = get_class_labels(dataset)
    extended_class_labels = get_extended_class_labels(dataset)

    prompts = []
    for i in range(len(class_labels)):
        prompts.append(f"This is a photo of a {class_labels[i]}. {extended_class_labels[i]}")

    # encode prompts
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    prompts_token = tokenizer(prompts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(prompts_token)
        text_features /= text_features.norm(dim=-1, keepdim=True) # normalize tensors

    correct = 0
    total = 0

    # check confusion matrix
    confusion_matrix = torch.zeros((len(prompts), len(prompts)))

    for _, (images, labels) in tqdm(enumerate(test_loader)):
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            probs = (image_features @ text_features.T).softmax(dim=-1).cpu().numpy()
            predicted_labels = probs.argmax(axis=1)
            correct += (predicted_labels == labels.cpu().numpy()).sum()
            total += len(labels)

            # add to confusion matrix
            for i in range(len(labels)):
                confusion_matrix[labels[i]][predicted_labels[i]] += 1
    
    print("Prompts Zeroshot Accuracy: ", correct / total)
    if len(extended_class_labels) < 11:
       print(confusion_matrix)

    cm_copy = confusion_matrix.clone()
    for i in range(len(class_labels)):
        cm_copy[i][i] = 0
    print("Top 3 Confusions")
    for i in range(3):
        max_val = cm_copy.max()
        max_index = torch.argmax(cm_copy)
        row = max_index // len(class_labels)
        col = max_index % len(class_labels)
        print("c1", class_labels[row], "c2", class_labels[col], "inds", row, col)
        cm_copy[row][col] = 0
    
def run_difference_classification(dataset, checkpoint=None, weight_avg=False, weight_alpha=None):

    # load model
    model, preprocess = load_clip_model("ViT-L/14", openclip=True, checkpoint=checkpoint, weight_avg=weight_avg, weight_alpha=weight_alpha) 

    # accuracies 
    accs = []

    for seed in range(5):
        test_loader = get_difference_dataloaders(dataset, preprocess, seed=seed, batch_size=32)

        correct = 0
        total = 0

        # loop through difference test loader
        for _, (images1, images2, difference_tokens, _, _) in tqdm(enumerate(test_loader)):

            with torch.no_grad():
                images1 = images1.to(device)
                images2 = images2.to(device)
                difference_tokens = difference_tokens.to(device)


                if dataset == "cub":
                    image_features1 = images1
                    image_features2 = images2
                else:
                    image_features1 = model.encode_image(images1)
                    image_features2 = model.encode_image(images2)

                image_features1 /= image_features1.norm(dim=-1, keepdim=True)
                image_features2 /= image_features2.norm(dim=-1, keepdim=True)

                difference_features = model.encode_text(difference_tokens)
                difference_features /= difference_features.norm(dim=-1, keepdim=True)

                # get difference between image features
                difference_image_features = image_features1 - image_features2
                difference_image_features /= difference_image_features.norm(dim=-1, keepdim=True)

                # get cosine similarity between difference and difference_features
                cosine_sim_pos_full = (difference_features @ difference_image_features.T).cpu().numpy()

                # grab only along diagonal
                cosine_sim_pos = cosine_sim_pos_full[np.diag_indices(cosine_sim_pos_full.shape[0])]

                # model is correct if cosine_sim_pos > cosine_sim_neg
                correct += (cosine_sim_pos > 0).sum()
                total += len(cosine_sim_pos)

        print("Difference Classification Accuracy: ", correct / total)
        accs = accs + [correct / total]
    
    print("Average Accuracy: ", np.mean(accs))
    print("Standard Error: ", np.std(accs) / np.sqrt(len(accs)))

    return np.mean(accs), np.std(accs) / np.sqrt(len(accs))


def run_comparative_subset(dataset, openclip=True, checkpoint=None, alpha=0.8, weight_avg=False, weight_alpha=None, subset=None):
    # load model
    model, preprocess = load_clip_model("ViT-L/14", openclip=openclip, checkpoint=checkpoint, weight_avg=weight_avg, weight_alpha=weight_alpha) 
    _, test_loader = get_dataloaders(dataset, preprocess, batch_size=32)

    class_labels = get_class_labels(dataset)
    prompts = [f"This is a photo of a {label}" for label in class_labels]

    # get comparatives
    if checkpoint is not None:
        comparative_prompts = get_comparatives(dataset, ft=True) 
    else:
        comparative_prompts = get_comparatives(dataset)
        
    # encode prompts
    if openclip:
        tokenizer = open_clip.get_tokenizer("ViT-L-14")
        prompts_token = tokenizer(prompts).to(device)
    else:
        prompts_token = clip.tokenize(prompts).to(device)

    text_features = model.encode_text(prompts_token)
    text_features /= text_features.norm(dim=-1, keepdim=True) # normalize tensors

    # loop through and encode comparatives
    comparative_features = []
    for _, comparative in comparative_prompts:
        if openclip:
            comparative_token = tokenizer(comparative).to(device)
        else:
            comparative_token = clip.tokenize(comparative).to(device)
        comparative_features.append(model.encode_text(comparative_token))
    comparative_features = torch.cat(comparative_features, dim=0)
    comparative_features /= comparative_features.norm(dim=-1, keepdim=True) # normalize tensors

    # average comparative - other class with original class embedding
    sum_comparative_features = torch.zeros_like(text_features)
    class_counts = np.zeros(len(class_labels))

    # keep track of updated class inds
    updated_class_inds = set()
    for index, (pair, _) in enumerate(comparative_prompts):
        sum_comparative_features[pair[1]] += text_features[pair[0]] - comparative_features[index]
        class_counts[pair[1]] += 1
        updated_class_inds.add(pair[0])
        updated_class_inds.add(pair[1])

    # add one to class_counts at 0 indices so that divide by 0 error does not happen
    class_counts[class_counts == 0] = 1
    updated_class_inds = sorted(list(updated_class_inds)) # index is subset class, value is original class

    # average  
    sum_comparative_features /= torch.tensor(class_counts).unsqueeze(-1).to(device)
    comp_text_features = (alpha) * text_features + (1 - alpha) * sum_comparative_features
    comp_text_features /= comp_text_features.norm(dim=-1, keepdim=True) # normalize tensors

    # get standard zs performance
    zs_text_features = text_features / text_features.norm(dim=-1, keepdim=True) # normalize tensors

    # only evaluate on subset of classes from updated_class_inds
    subset_text_features = comp_text_features[updated_class_inds]
    subset_zs_text_features = zs_text_features[updated_class_inds]

    correct = 0
    total = 0

    # check confusion matrix
    all_labels = []
    all_predictions = []
    all_zs_predictions = []

    for _, (images, labels) in tqdm(enumerate(test_loader), total=len(test_loader)):

        # filter for labels in updated_class_inds
        mask = torch.tensor([label in updated_class_inds for label in labels], dtype=torch.bool)
        images = images[mask].to(device)
        labels = labels[mask].to(device)

        if len(images) == 0:
            continue

        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            probs = (image_features @ subset_text_features.T).softmax(dim=-1).cpu().numpy()
            predicted_labels = probs.argmax(axis=1)

            zs_probs = (image_features @ subset_zs_text_features.T).softmax(dim=-1).cpu().numpy()
            zs_predicted_labels = zs_probs.argmax(axis=1)

            correct += (predicted_labels == labels.cpu().numpy()).sum()
            total += len(labels)
            # convert labels to numpy and mapped to subset indices
            labs = labels.cpu().numpy()
            conv_labels = np.array([updated_class_inds.index(lab) for lab in labs])
            all_labels.extend(conv_labels)
            all_predictions.extend(predicted_labels)
            all_zs_predictions.extend(zs_predicted_labels)
    
    # convert predictions and labels to numpy
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_zs_predictions = np.array(all_zs_predictions)
    
    print("Subset ZS Accuracy: ", (all_labels == all_zs_predictions).sum() / len(all_labels))
    print("Subset Comparative Accuracy: ", (all_labels == all_predictions).sum() / len(all_labels))

if __name__ == "__main__":

    # setup argparse
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="zs", help="zs, appended, avg, subtract")
    parser.add_argument("--alpha", type=float, default=0.9, help="alpha value for subtracted embeddings")
    parser.add_argument("--dataset", type=str, default="cifar10", help="cifar10, cifar100")
    parser.add_argument("--llm", type=str, default="flan-t5", help="chatgpt, flan-t5, stability")
    parser.add_argument("--openclip", action="store_true", help="use openclip instead of clip")
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint to load")
    parser.add_argument("--weight_avg", action="store_true", help="weight average embeddings")
    parser.add_argument("--weight_alpha", type=float, default=0.5, help="weight average embeddings amount")
    args = parser.parse_args()

    args.openclip = True # in case i forgot :)

    # set all random seeds
    torch.manual_seed(0)
    np.random.seed(0)

    if args.mode == "zs":
        run_zero_shot(args.dataset, openclip=args.openclip, checkpoint=args.checkpoint, weight_avg=args.weight_avg, weight_alpha=args.weight_alpha)
    elif args.mode == "comp":
        run_comparative(args.dataset, openclip=args.openclip, checkpoint=args.checkpoint, alpha=args.alpha, weight_avg=args.weight_avg, weight_alpha=args.weight_alpha)
    elif args.mode == "extended":
        run_extended(args.dataset, checkpoint=args.checkpoint, weight_avg = args.weight_avg, weight_alpha=args.weight_alpha)
    elif args.mode == "diff":
        run_difference_classification(args.dataset, checkpoint=args.checkpoint, weight_avg=args.weight_avg, weight_alpha=args.weight_alpha)
    elif args.mode == "comp_subset":
        run_comparative_subset(args.dataset, openclip=args.openclip, checkpoint=args.checkpoint, alpha=args.alpha, weight_avg=args.weight_avg, weight_alpha=args.weight_alpha)
    else:
        print("Not implemented yet")