import torch
import torch.utils.data as data
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.datasets import EuroSAT
from torchvision.transforms import transforms
from torchvision.datasets import SUN397, Flowers102, INaturalist, \
    Food101, Places365, StanfordCars, STL10, PCAM, OxfordIIITPet
import torchvision
from PIL import Image

import open_clip

from torchvision.datasets import ImageFolder
from datasets import load_dataset

from tqdm import tqdm

import pickle 
import sys
from typing import Tuple
import os
import certifi
import numpy as np

os.environ['SSL_CERT_FILE'] = certifi.where()

def get_class_labels(dataset_name):

    if dataset_name == "cifar10":
        return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    elif dataset_name == "cifar100":
        cifar100_dataset = load_dataset("cifar100")
        return cifar100_dataset["train"].features["fine_label"].names

    elif dataset_name == "eurosat":
        return [
            'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 
            'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 
            'River', 'SeaLake'
        ]

    elif dataset_name == "awa2":
        test_dataset = AwA2("./data/Animals_with_Attributes2/", preprocess=None, split="test")
        return test_dataset.classes

    elif dataset_name == "sun":
        sun397_dataset = SUN397(root='./data/sun/', download=False)
        return sun397_dataset.classes

    elif dataset_name == "cub":
        
        # empty transform
        _, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="datacomp_xl_s13b_b90k")
        dataset = CUBDataset(preprocess=preprocess)
        return dataset.classes
    
    elif dataset_name == "flowers":
        # read classes from file
        with open("./data/flowers_class_labels.txt", "r") as f:
            class_labels = f.readlines()
        return [s.strip() for s in class_labels]

class PairDataset(data.Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

        # generate pairs of unique data
        self.pairs = []

        # set random seeds
        torch.manual_seed(0)

        # randomly sampling 1000 indices
        rand_inds = torch.randperm(len(self.dataset))
        max_ind = min(len(self.dataset), 100)
        rand_inds = rand_inds[:max_ind]

        # sort random indices
        rand_inds = rand_inds.sort()[0]

        for i in range(len(rand_inds)):
            for j in range(i + 1, len(rand_inds)):
                self.pairs.append((rand_inds[i], rand_inds[j]))

    def __getitem__(self, index):
        # get pair index
        pair_index = self.pairs[index]
        # return pair of images
        return self.dataset[pair_index[0]][0], self.dataset[pair_index[1]][0], index

    def __len__(self):
        return len(self.pairs)

# define function to return dataloaders for a dataset
def get_dataloaders(name, preprocess, batch_size=128, num_shots=None):
    print('Loading dataset: {}'.format(name))

    if name == "cifar10":
        train_dataset = CIFAR10(root='./data/cifar10', train=True, download=True, transform=preprocess)
        test_dataset = CIFAR10(root='./data/cifar10', train=False, download=True, transform=preprocess)
    
    elif name == "cifar100":
        train_dataset = CIFAR100(root='./data/cifar100', train=True, download=True, transform=preprocess)
        test_dataset = CIFAR100(root='./data/cifar100', train=False, download=True, transform=preprocess)
    
    elif name == "awa2":
        train_dataset = AwA2("./data/Animals_with_Attributes2/", preprocess=preprocess, split="train")
        test_dataset = AwA2("./data/Animals_with_Attributes2/", preprocess=preprocess, split="test")

    elif name == "eurosat":
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        train_test_dataset = EuroSAT(root='./data/', download=True, transform=preprocess)
        print(len(train_test_dataset))

        # setting random seed
        torch.manual_seed(0)

        #split into train and test, randomly sampling 90% of examples
        rand_indices = torch.randperm(len(train_test_dataset))
        train_dataset = torch.utils.data.Subset(train_test_dataset, rand_indices[:int(len(train_test_dataset)*0.9)])
        test_dataset = torch.utils.data.Subset(train_test_dataset, rand_indices[int(len(train_test_dataset)*0.9):])

    elif name == "sun":
        train_test_dataset = SUN397(root='./data/sun/', download=False, transform=preprocess)
        print(len(train_test_dataset))

        # setting random seed
        torch.manual_seed(0)

        #split into train and test, randomly sampling 90% of examples
        rand_indices = torch.randperm(len(train_test_dataset))
        train_dataset = torch.utils.data.Subset(train_test_dataset, rand_indices[:int(len(train_test_dataset)*0.9)])
        test_dataset = torch.utils.data.Subset(train_test_dataset, rand_indices[int(len(train_test_dataset)*0.9):])

    elif name == "cub":
        train_dataset = CUBDataset(preprocess=preprocess, split="train")
        test_dataset = CUBDataset(preprocess=preprocess, split="test")
        
    elif name == "flowers":
        train_dataset = Flowers102(root='./data/Flowers102/', split='train', download=True, transform=preprocess)
        test_dataset = Flowers102(root='./data/Flowers102/', split='test', download=True, transform=preprocess)

    if num_shots is not None:    
        # take num_shots examples per class in training data
        train_indices = []

        classes = get_class_labels(name)
        num_classes = len(classes)
        num_total = num_classes * num_shots

        # randomly select num_total samples
        rand_inds = torch.randperm(len(train_dataset))
        rand_inds = rand_inds[:num_total]
        train_dataset = torch.utils.data.Subset(train_dataset, rand_inds)

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    return train_loader, test_loader    

## USED FOR COMPARATIVE Adaptation (and to evaluate comparative prompting)
def get_comparative_dataloaders(name, preprocess, batch_size=32):

    print('Loading dataset: {}'.format(name))

    if name == "cifar10":
        train_dataset = CIFAR10(root='./data/cifar10', train=True, download=False, transform=preprocess)
        test_dataset = CIFAR10(root='./data/cifar10', train=False, download=False, transform=preprocess)
    
    elif name == "cifar100":
        train_dataset = CIFAR100(root='./data/datasets/cifar100', train=True, download=True, transform=preprocess)
        test_dataset = CIFAR100(root='./data/datasets/cifar100', train=False, download=True, transform=preprocess)
    
    elif name == "eurosat":
        train_test_dataset = EuroSAT(root='./data/', download=False, transform=preprocess)
        print(len(train_test_dataset))

        # setting random seed
        torch.manual_seed(0)

        #split into train and test, randomly sampling 90% of examples
        rand_indices = torch.randperm(len(train_test_dataset))
        train_dataset = torch.utils.data.Subset(train_test_dataset, rand_indices[:int(len(train_test_dataset)*0.9)])
        test_dataset = torch.utils.data.Subset(train_test_dataset, rand_indices[int(len(train_test_dataset)*0.9):])

    elif name == "sun":
        train_test_dataset = SUN397(root='./data/sun/', download=False, transform=preprocess)
        print(len(train_test_dataset))

        # setting random seed
        torch.manual_seed(0)

        #split into train and test, randomly sampling 90% of examples
        rand_indices = torch.randperm(len(train_test_dataset))
        train_dataset = torch.utils.data.Subset(train_test_dataset, rand_indices[:int(len(train_test_dataset)*0.9)])
        test_dataset = torch.utils.data.Subset(train_test_dataset, rand_indices[int(len(train_test_dataset)*0.9):])

    elif name == "cub":
        train_dataset = CUBDataset(preprocess=preprocess, split="train")
        test_dataset = CUBDataset(preprocess=preprocess, split="test")

        # get indices of data points for train and test set that correspond to first 10 classes
        train_indices = []
        test_indices = []
        for i in range(len(train_dataset)):
            if train_dataset[i][1] < 10:
                train_indices.append(i)
        for i in range(len(test_dataset)):
            if test_dataset[i][1] < 10:
                test_indices.append(i)

        # get subsets of train and test set that correspond to first 10 classes
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        test_dataset = torch.utils.data.Subset(test_dataset, test_indices)

    elif name == "flowers":
        train_dataset = Flowers102(root='./data/Flowers102/', split='train', download=False, transform=preprocess)
        test_dataset = Flowers102(root='./data/Flowers102/', split='test', download=False, transform=preprocess)

    # convert dataset into dataset with pairs of images
    pair_train_dataset = PairDataset(train_dataset)
    pair_test_dataset = PairDataset(test_dataset)

    train_loader = data.DataLoader(pair_train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = data.DataLoader(pair_test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    return train_loader, test_loader   

def get_difference_dataloaders(name, preprocess, seed=0, batch_size=32):

    if name == "awa2":
        # train_dataset = AwA2DifferenceClassificationDataset("/scratch/shared/public_datasets/Animals_with_Attributes2/", preprocess=preprocess, split="train")
        test_dataset = AwA2DifferenceClassificationDataset("./data/datasets/Animals_with_Attributes2/", seed=seed, preprocess=preprocess, split="test")

    elif name == "awa2_size":
        test_dataset = AwA2SizeClassificationDataset("./data/datasets/Animals_with_Attributes2/", preprocess=preprocess, split="test")

    elif name == "cub":
        # train_dataset = CUBDifferenceClassificationDataset("/scratch/shared/public_datasets/CUB_200_2011/", preprocess=preprocess, split="train")
        test_dataset = CUBDifferenceClassificationTokens("./CLIP_prefix_caption/data/birds/ViT-L_14_test.pkl", seed=seed)

    elif name == "cifar100_size":
        test_dataset = CIFAR100SizeClassificationDataset(path="./data/datasets/cifar100", preprocess=preprocess, seed=seed, split="test")

    elif name == "flowers_color":
        test_dataset = FlowersColorClassificationDataset(path="./data/datasets/Flowers102/", preprocess=preprocess, seed=seed, split="test")

    # train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    # return train_loader, test_loader
    return test_loader

def get_coco_dataloader(batch_size=32):

    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    dataset = COCODataset(tokenizer, split="train")
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    return dataloader

def get_coco_rewritten_dataloader(batch_size=32):

    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    dataset = COCODatasetRewritten(tokenizer, split="train")
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    return dataloader



class COCODataset(data.Dataset):

    def __init__(self, tokenizer, split="train"):

        coco_dict = pickle.load(open("./CLIP_prefix_caption/data/coco/CLIP_ViT-L_14_train.pkl", "rb"))
        self.embeddings = coco_dict["clip_embedding"]
        self.captions = [coco_dict["captions"][i]["caption"] for i in range(len(coco_dict["captions"]))]
        self.tokenizer = tokenizer
        self.captions = [tokenizer(caption).squeeze(0) for caption in self.captions]

    def __getitem__(self, index):
        return self.embeddings[index], self.captions[index]

    def __len__(self):
        return len(self.embeddings)

class COCODatasetRewritten(data.Dataset):

    def __init__(self, tokenizer, split="train"):

        coco_dict = pickle.load(open("./data/CLIP_ViT-L_14_train.pkl", "rb"))

        # load rewrite file
        rewrite_dict = pickle.load(open("./llm_rewrite/coco_1000_13b_filtered.pkl", "rb"))
        indices = rewrite_dict["indices"]
        captions = rewrite_dict["rewritten_captions"]
        self.tokenizer = tokenizer

        self.embeddings = coco_dict["clip_embedding"][indices]
        self.captions = [tokenizer(caption).squeeze(0) for caption in captions]

    def __getitem__(self, index):
        return self.embeddings[index], self.captions[index]

    def __len__(self):
        return len(self.embeddings)

class AwA2(data.Dataset):
        
        '''
        AwA2 dataset
        '''
    
        def __init__(self, path, preprocess, split="test"):
    
            # load the AwA2 dataset from path -> structure is a torch image folder dataset
            self.dataset = ImageFolder(root=path + split, transform=preprocess)
            print("Dataset length", len(self.dataset))

            self.attributes_list = []
            # read from file predicates.txt -> names of predicates 
            with open("./data/Animals_with_Attributes2/predicates.txt") as f:
                for line in f:
                    self.attributes_list.append(line.split("\t")[-1].strip())

            # print("Attributes", self.attributes_list)
            
            self.attributes = []
            # load binary matrix corresponding to presence of an attribute
            with open("./data/Animals_with_Attributes2/predicate-matrix-binary.txt") as f:
                self.attributes = [[int(x) for x in line.split(" ")] for line in f]
            # convert to numpy
            self.attributes = np.array(self.attributes)
            # print("Attributes", self.attributes.shape)

            # get name of classes
            self.classes = []
            with open("./data/Animals_with_Attributes2/" + split  + "classes.txt") as f:
                for line in f:
                    self.classes.append(line.strip())
            
            # sort classes by alphabetical order
            self.classes.sort()
            print(self.classes)

        def __getitem__(self, index):
            # get image and label
            image, label = self.dataset[index]
            # get attributes
            attributes = self.attributes[label]

            # convert label to index in class_indices
            # label = self.class_indices.index(label)
            return image, label
        
        def __len__(self):
            return len(self.dataset)

class AwA2SizeClassificationDataset(data.Dataset):
    '''
    Difference based classification task on only size in AwA2
    '''

    def __init__(self, path, preprocess, split="test"):

        # load the AwA2 dataset from path
        self.awa2_dataset = AwA2(path, preprocess, split=split)

        # convert into pairs of data
        # get random set of 100 datapoints
        torch.manual_seed(0)
        rand_inds = torch.randperm(len(self.awa2_dataset))
        rand_inds = rand_inds[:100]

        # get pairs of data
        self.indices = []
        self.comparatives = []
        
        comp_string_big = "The first image contains an animal that is large, while the second image contains an animal that is small."
        comp_string_small = "The first image contains an animal is small, while the second image contains an animal that is large."
        
        tokenizer = open_clip.get_tokenizer("ViT-L-14")
        comp_tokens_big = tokenizer(comp_string_big).squeeze(0)
        comp_tokens_small = tokenizer(comp_string_small).squeeze(0)

        for i in tqdm(range(len(rand_inds)), total=len(rand_inds)):

            for j in range(i + 1, len(rand_inds)):

                # check i and j to not be of the same class
                class_i = self.awa2_dataset[rand_inds[i]][1]
                class_j = self.awa2_dataset[rand_inds[j]][1]

                # print("Class i", class_i, "Class j", class_j)

                # get attribute vectors of i and j
                attributes_i = self.awa2_dataset.attributes[class_i]
                attributes_j = self.awa2_dataset.attributes[class_j]

                # check if there is a difference in size (axis 15 is big)
                if attributes_i[15] != attributes_j[15]:

                    self.indices.append((rand_inds[i], rand_inds[j]))
                    
                    if attributes_i[15] == 1: # first image big
                        self.comparatives.append(comp_tokens_big)
                    else:
                        # self.indices.append((rand_inds[j], rand_inds[i]))
                        self.comparatives.append(comp_tokens_small)

        print("Generated pairs", len(self.indices))

    def __getitem__(self, index):
            
            image1, class1 = self.awa2_dataset.__getitem__(self.indices[index][0])
            image2, class2 = self.awa2_dataset.__getitem__(self.indices[index][1])
    
            # get comparative
            comp = self.comparatives[index]
    
            return image1, image2, comp, class1, class2
    
    def __len__(self):
        return len(self.indices)


class AwA2DifferenceClassificationDataset(data.Dataset):
    
    '''
    Difference based classification task on the AwA2 dataset
    '''

    def __init__(self, path, preprocess, seed=0, split="test"):

        # load the AwA2 dataset from path
        self.awa2_dataset = AwA2(path, preprocess, split=split)

        # convert into pairs of data
        # get random set of 100 datapoints
        torch.manual_seed(seed)
        rand_inds = torch.randperm(len(self.awa2_dataset))
        rand_inds = rand_inds[:100]

        # get pairs of data
        self.indices = []
        for i in tqdm(range(len(rand_inds)), total=len(rand_inds)):
            for j in range(i + 1, len(rand_inds)):

                # check i and j to not be of the same class
                class_i = self.awa2_dataset[rand_inds[i]][1]
                class_j = self.awa2_dataset[rand_inds[j]][1]

                # print("Class i", class_i, "Class j", class_j)

                if class_i != class_j:
                    self.indices.append((rand_inds[i], rand_inds[j]))

        print("Generated pairs", len(self.indices))
        

        # get attributes
        self.attributes = self.awa2_dataset.attributes
        self.attributes_names = self.awa2_dataset.attributes_list

        # get openclip tokenizer
        tokenizer = open_clip.get_tokenizer("ViT-L-14")

        # have dictionary to store results of generated strings (maps: class1 -> {class2 -> tokens})
        self.string_token_dict = {i: {} for i in range(len(self.awa2_dataset.classes))}

        # loop through classes
        for i in tqdm(range(len(self.awa2_dataset.classes)), total=len(self.awa2_dataset.classes)):
            for j in range(len(self.awa2_dataset.classes)):

                # get attributes
                attributes1 = self.attributes[i]
                attributes2 = self.attributes[j]

                # generate string
                diff_string = self.gen_attribute_diff_string(attributes1, attributes2)

                # tokenize
                tokens = tokenizer(diff_string).squeeze(0)
                self.string_token_dict[i][j] = tokens

                # print(tokens.shape)

        print("Finished tokenizing differences")

    def gen_attribute_diff_string(self, attributes1, attributes2):

        '''
        Function to generate a string that correponds to the difference in attributes
        
        attributes1 - a boolean array for attributes contained in image1
        attributes2 - a boolean array for attributes contained in image2

        returns - a string that corresponds to the difference in attributes
        '''

        # get indices of attributes that are true
        indices1 = []
        for i in range(len(attributes1)):
            if attributes1[i] == 1:
                indices1.append(i)

        indices2 = []
        for i in range(len(attributes2)):
            if attributes2[i] == 1:
                indices2.append(i)
        
        # get attributes that are different
        diff_indices1 = []
        for i in indices1:
            if i not in indices2:
                diff_indices1.append(i)

        diff_indices2 = []
        for i in indices2:
            if i not in indices1:
                diff_indices2.append(i)

        # set of attributes to filter out
        to_filter= [
            34, # smelly (olfactory attribute, not visual)
            38, # tunnels (behavioral attribute)
            40, # fast (describes speed, not a visual attribute)
            41, # slow (describes speed, not a visual attribute)
            51, # agility (describes physical ability, not visual)
            52, # fish (diet, not visual)
            53, # meat (diet, not visual)
            54, # plankton (diet, not visual)
            56, # insects (diet, not visual)
            57, # forager (diet/behavioral)
            58, # grazer (diet/behavioral)
            59, # hunter (diet/behavioral)
            60, # scavenger (diet/behavioral)
            61, # skimmer (diet/behavioral)
            62, # stalker (diet/behavioral)
            83, # solitary (social behavior, not a visual attribute of individual animals)
        ]

        # # filter out attributes
        diff_indices1 = [i for i in diff_indices1 if i not in to_filter]
        diff_indices2 = [i for i in diff_indices2 if i not in to_filter]

        # get attribute names
        diff_attributes1 = []
        for i in diff_indices1:
            diff_attributes1.append(self.attributes_names[i])

        diff_attributes2 = []
        for i in diff_indices2:
            diff_attributes2.append(self.attributes_names[i])

        # generate string
        diff_string = "The first image has attributes of {}, while the second image has attributes of {}".format(", ".join(diff_attributes1), ", ".join(diff_attributes2))
        return diff_string

    def __getitem__(self, index):
        
        image1, class1 = self.awa2_dataset.__getitem__(self.indices[index][0])
        image2, class2 = self.awa2_dataset.__getitem__(self.indices[index][1])

        # get difference in attributes
        diff_tokens = self.string_token_dict[class1][class2]
        return image1, image2, diff_tokens, class1, class2

    def __len__(self):
        return len(self.indices)

class CUBDifferenceClassificationTokens(data.Dataset):

    '''
    Load a dataset for birds comparatives pairs using the CLIP tokenizer rather than for a caption model
    '''

    def __len__(self) -> int:
        return len(self.pair_inds)


    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        
        ind1, ind2 = self.pair_inds[item]
        
        # tokens1 = self.captions_tokens[ind1]
        # tokens2 = self.captions_tokens[ind2]
        
        image_embed1 = self.prefixes[ind1]
        image_embed2 = self.prefixes[ind2]

        label1 = self.labels[ind1]
        label2 = self.labels[ind2]

        # return tokenized comparative
        comp_tokens = self.comparatives_tokens[item]
        # return tokens1, prefix1, tokens2, prefix2, comp_tokens 

        return image_embed1, image_embed2, comp_tokens, label1, label2

    def __init__(self, data_path: str, seed=0, comparatives_path="llm_diffs/birds_200_test_filtered.pkl", sub=True):
        
        comp_dict = pickle.load(open(comparatives_path, "rb"))
        
        if sub: # used for downstream db classification
            # randomly select 5000 pairs
            torch.manual_seed(seed)
            rand_inds = torch.randperm(len(comp_dict["indices"]))
            rand_inds = rand_inds[:5000]
        
            self.pair_inds = np.array(comp_dict["indices"])[rand_inds]
            self.comparatives = np.array(comp_dict["llm_diffs"])[rand_inds]
        
        else:
            self.pair_inds = np.array(comp_dict["indices"])
            self.comparatives = np.array(comp_dict["llm_diffs"])

        self.tokenizer = open_clip.get_tokenizer("ViT-L-14")
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)

        print("Data size is %0d" % len(all_data["clip_embedding"]))
        sys.stdout.flush()
        self.prefixes = all_data["clip_embedding"]
        captions_raw = all_data["captions"]
        
        if "labels" not in all_data: # on train so set labels to be a list of all -1's
            self.labels = [-1 for _ in range(len(captions_raw))]
        else:
            self.labels = all_data["labels"]

        # turn comparatives into tokens
        self.comparatives_tokens = []
        for comp in tqdm(self.comparatives):
            comp_tokens = self.tokenizer(comp)[0]
            self.comparatives_tokens.append(comp_tokens)

class CIFAR100SizeClassificationDataset(data.Dataset):

    def __init__(self, path, preprocess, seed=0, split="test"):

        # load the cifar dataset from path
        cifar100_dataset = CIFAR100(root=path, train=False, download=False, transform=preprocess)

        large_classes = [15, 42, 43, 97] # 97 - wolf, 43 - lion, 42 - leopard, 15 - camel
        small_classes = [36, 65, 74, 80] # 36 - hamster, 65 - rabbit, 74 - shrew, 80 - squirrel

        # filter dataset to only contain images with classes in large_classes or small_classes
        indices = []
        for i in range(len(cifar100_dataset)):
            if cifar100_dataset[i][1] in large_classes or cifar100_dataset[i][1] in small_classes:
                indices.append(i)

        self.dataset = torch.utils.data.Subset(cifar100_dataset, indices)

        # randomly select 100 indices
        torch.manual_seed(seed)
        rand_inds = torch.randperm(len(self.dataset))
        rand_inds = rand_inds[:100]
        
        # get pairs of data
        self.indices = []
        self.comparatives = []
        
        comp_string_big = "The first image contains a larger animal, whiel the second contains a smaller animal."
        comp_string_small = "The first image contains a smaller animal, while the second contains a larger animal."

        tokenizer = open_clip.get_tokenizer("ViT-L-14")

        comp_tokens_big = tokenizer(comp_string_big).squeeze(0)
        comp_tokens_small = tokenizer(comp_string_small).squeeze(0)

        for i in tqdm(range(len(rand_inds)), total=len(rand_inds)):

            for j in range(i + 1, len(rand_inds)):

                # check i and j to not be of the same class
                class_i = self.dataset[rand_inds[i]][1]
                class_j = self.dataset[rand_inds[j]][1]

                # check if there is a difference in size (axis 15 is big)
                if class_i != class_j:
                    # check if class_i and class_j are contained in different lists of large_classes or small_classes
                    if (class_i in large_classes and class_j in small_classes) or (class_i in small_classes and class_j in large_classes):
                        self.indices.append((rand_inds[i], rand_inds[j]))
                        
                        if class_i in large_classes:
                            self.comparatives.append(comp_tokens_big)
                        else:
                            self.comparatives.append(comp_tokens_small)

        print("Generated pairs", len(self.indices))

    def __getitem__(self, index):
                
        image1, class1 = self.dataset.__getitem__(self.indices[index][0])
        image2, class2 = self.dataset.__getitem__(self.indices[index][1])

        # get comparative
        comp = self.comparatives[index]

        return image1, image2, comp, class1, class2
    
    def __len__(self):
        return len(self.indices)

class FlowersColorClassificationDataset(data.Dataset):

    def __init__(self, path, preprocess, seed=0, split="test"):

        # load the flowers dataset from path
        flowers_dataset = Flowers102(root=path, split=split, download=False, transform=preprocess)

        blue_classes = [2, 7] # 2 - bluebells, 7 - blue poppy
        yellow_classes = [4, 6, 10, 19] # 4 - sunflowers, 6 - goldenrod, 10 - yellow iris, 19 - daffodil
        
        # filter dataset to only contain images with classes in blue or yellow classes
        indices = []
        for i in range(len(flowers_dataset)):
            if flowers_dataset[i][1] in blue_classes or flowers_dataset[i][1] in yellow_classes:
                indices.append(i)

        self.dataset = torch.utils.data.Subset(flowers_dataset, indices)

        # randomly select 100 indices
        torch.manual_seed(seed)
        rand_inds = torch.randperm(len(self.dataset))
        rand_inds = rand_inds[:100]
        
        # get pairs of data
        self.indices = []
        self.comparatives = []

        comp_string_blue = "The first flower is blue, while the second is yellow."
        comp_string_yellow = "The first flower is yellow, while the second is blue."


        tokenizer = open_clip.get_tokenizer("ViT-L-14")

        comp_tokens_blue = tokenizer(comp_string_blue).squeeze(0)
        comp_tokens_yellow = tokenizer(comp_string_yellow).squeeze(0)

        for i in tqdm(range(len(rand_inds)), total=len(rand_inds)):

            for j in range(i + 1, len(rand_inds)):

                # check i and j to not be of the same class
                class_i = self.dataset[rand_inds[i]][1]
                class_j = self.dataset[rand_inds[j]][1]

                # check if there is a difference in color
                if class_i != class_j:
                    # check if class_i and class_j are contained in different lists of blue_classes or yellow_classes
                    if (class_i in blue_classes and class_j in yellow_classes) or (class_i in yellow_classes and class_j in blue_classes):
                        self.indices.append((rand_inds[i], rand_inds[j]))
                        
                        if class_i in blue_classes:
                            self.comparatives.append(comp_tokens_blue)
                        else:
                            self.comparatives.append(comp_tokens_yellow)

        print("Generated pairs", len(self.indices))

    def __getitem__(self, index):
                    
            image1, class1 = self.dataset.__getitem__(self.indices[index][0])
            image2, class2 = self.dataset.__getitem__(self.indices[index][1])
    
            # get comparative
            comp = self.comparatives[index]
    
            return image1, image2, comp, class1, class2
    
    def __len__(self):
        return len(self.indices)


class CUBDataset(data.Dataset):

    def __init__(self, preprocess, split="train"):

        # load the CUB dataset from path
        path = "./data/CC6204-Hackaton-Cub-Dataset/data"

        # load classses from text file
        with open(f"{path}/classes.txt") as f:
            self.classes = [line.strip().split(" ")[1][4:] for line in f]

        # convert cub dataset to correct format
        data, labels = [], []
        captions = []

        train_inds = []
        test_inds = []
        with open(f"{path}/train_test_split.txt") as f:
            for line in f:
                ind, is_train = line.strip().split(" ")
                if is_train == "1":
                    train_inds.append(ind)
                else:
                    test_inds.append(ind)
        
        if split == "train":
            inds = train_inds
        else:
            inds = test_inds
        
        # loop through images file and get correct paths
        with open(f"{path}/images.txt") as f:
            for line in f:
                ind, img_path = line.strip().split(" ")
                if ind in inds:
                    image = Image.open(f"{path}/images/{img_path}")
                    data.append(preprocess(image))
                    labels.append(int(img_path.split(".")[0]) - 1)
        
                    with open(f"{path}/text/" + img_path[:-4] + ".txt") as f:
                        captions.append(f.read().strip())

        self.images = data
        self.labels = labels
        self.images = torch.stack(self.images)
        self.captions = captions

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.images)
