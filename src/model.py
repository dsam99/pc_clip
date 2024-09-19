import torch
from PIL import Image
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import clip
import open_clip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_clip_model(model_name, openclip=True, checkpoint=None, weight_avg=False, weight_alpha=None):

    '''
    Load CLIP model from HuggingFace
    :param model_name: model name
    :return: model, preprocessor
    '''

    if openclip:

        if checkpoint is None:
            model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained="datacomp_xl_s13b_b90k")
            model = model.to(device)

        if checkpoint is not None:
            model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=False)
            print("Loading checkpoint: " + str(checkpoint))
            checkpoint = torch.load(checkpoint)
            model.load_state_dict(checkpoint)

            if weight_avg:

                # averaging weights
                print("averaging weights")

                model2, _, _ = open_clip.create_model_and_transforms(model_name, pretrained="datacomp_xl_s13b_b90k")
                # average models in weight space
                model_dict = model.state_dict()
                model2_dict = model2.state_dict()

                if weight_alpha is not None:
                    for k in model_dict.keys():
                        model_dict[k] = (weight_alpha * model_dict[k] + (1 - weight_alpha) * model2_dict[k])
                else:
                    for k in model_dict.keys():
                        model_dict[k] = (model_dict[k] + model2_dict[k]) / 2
                
                model.load_state_dict(model_dict)

            model = model.to(device)

    else:
        model, preprocess = clip.load(model_name, device=device)
    return model, preprocess

if __name__ == "__main__":
    
    # load model
    model, preprocess = load_clip_model("ViT-L/14") 

