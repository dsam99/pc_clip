import torch
import numpy as np
import pickle

if __name__ == "__main__":
    # load llm differences
    llm_dict = pickle.load(open("llm_diffs/coco_diffs_1000.pkl", "rb"))
    llm_diffs = llm_dict["llm_diffs"]
    indices = llm_dict["indices"]

    new_diffs = []
    new_indices = []


    skip_count = 0
    early_count = 0

    for i in range(len(llm_diffs)):
        if llm_diffs[i].startswith("\n\n\n"):
            skip_count += 1 
            continue
        
        if llm_diffs[i].startswith(" \n\n\n"):
            skip_count += 1 
            continue

        if llm_diffs[i].startswith("# 1000"):
            skip_count += 1
            continue
        if llm_diffs[i].startswith("#include"): # filtering out code responses
            skip_count += 1
            continue
        
        if "#define" in llm_diffs[i]:
            skip_count += 1
            continue

        if "0000000000000000000000000000000000000000000000000000000000" in llm_diffs[i]: # filtering out case with just a weird number
            skip_count += 1
            continue
        if "100% of the time" in llm_diffs[i]:
            skip_count += 1
            continue

        if "gatsbyjs" in llm_diffs[i]: # filtering out weird generation
            skip_count += 1
            continue
        
        if "The following are the steps to create a successful landing page:" in llm_diffs[i]: # filtering out weird generation
            skip_count += 1
            continue

        if "The following are the steps to create a successful business plan for a small business" in llm_diffs[i]: # filtering out weird generation
            skip_count += 1
            continue

        if "ѐs." in llm_diffs[i]: # filtering out weird generation
            skip_count += 1
            continue

        if "ѐs," in llm_diffs[i]: # filtering out weird generation
            skip_count += 1
            continue

        if "%3D%3D%3D%3D%3D%3D%3D%" in llm_diffs[i]: # filtering out weird generation
            skip_count += 1
            continue

        if "birdtalking about the weather" in llm_diffs[i]: # filtering out weird generation
            skip_count += 1
            continue

        if "The cat is standing on the car seat, looking out the window." in llm_diffs[i]:
            skip_count += 1
            continue

        if "$\}\}%" in llm_diffs[i]:
            skip_count += 1
            continue

        if "Description of the image:" in llm_diffs[i]:
            skip_count += 1
            continue

        if "Description:" in llm_diffs[i]:
            skip_count += 1
            continue

        if "www.sciencedirect.com" in llm_diffs[i]:
            skip_count += 1
            continue

        if ".. .. .." in llm_diffs[i]:
            skip_count += 1
            continue

        if "\" \" \"" in llm_diffs[i]:
            skip_count += 1
            continue

        if "thththth" in llm_diffs[i]:
            skip_count += 1
            continue

        if "?????????" in llm_diffs[i]:
            skip_count += 1
            continue

        if "t=t=t=t=" in llm_diffs[i]:
            skip_count += 1
            continue

        if "{{{" in llm_diffs[i]:
            skip_count += 1
            continue

        if "aaaaaaa" in llm_diffs[i]:
            skip_count += 1
            continue

        if "t:t:t:" in llm_diffs[i]:
            skip_count += 1
            continue

        if "\"\"\"" in llm_diffs[i]:
            skip_count += 1
            continue

        if "front-end-d" in llm_diffs[i]:
            skip_count += 1
            continue

        if "lamalamalama" in llm_diffs[i]:
            skip_count += 1
            continue

        if "tgthtgthtgth" in llm_diffs[i]:
            skip_count += 1
            continue

        if "t.t.t.t" in llm_diffs[i]:
            skip_count += 1
            continue

        if "t1t1t1" in llm_diffs[i]:
            skip_count += 1
            continue

        if "t=A:t=A" in llm_diffs[i]:
            skip_count += 1
            continue

        if "</p>" in llm_diffs[i]:
            skip_count += 1
            continue

        if "aiaiai" in llm_diffs[i]:
            skip_count += 1
            continue

        if ". . . . " in llm_diffs[i]:
            skip_count += 1
            continue

        if "t-t-t-t" in llm_diffs[i]:
            skip_count += 1
            continue

        if "!!!!!" in llm_diffs[i]:
            skip_count += 1
            continue

        if "tthth" in llm_diffs[i]:
            skip_count += 1
            continue

        if "1000000" in llm_diffs[i]:
            skip_count += 1
            continue

        if "thumbthumb" in llm_diffs[i]:
            skip_count += 1
            continue
        if "****" in llm_diffs[i]:
            skip_count += 1
            continue

        if "11111" in llm_diffs[i]:
            skip_count += 1
            continue

        if "-----" in llm_diffs[i]:
            skip_count += 1
            continue

        if "<br>" in llm_diffs[i]:
            skip_count += 1
            continue

        if "0:00:00" in llm_diffs[i]:
            skip_count += 1
            continue

        if "55555" in llm_diffs[i]:
            skip_count += 1
            continue

        if "thttht" in llm_diffs[i]:
            skip_count += 1
            continue

        if "t,t,t," in llm_diffs[i]:
            skip_count += 1
            continue

        if "tone:tone" in llm_diffs[i]:
            skip_count += 1
            continue

        if "släktet" in llm_diffs[i]:
            skip_count += 1
            continue

        if "*" in llm_diffs[i]:
            skip_count += 1
            continue
        if "mwm" in llm_diffs[i]:
            skip_count += 1
            continue

        if "I's a bird, I's a bird," in llm_diffs[i]:
            skip_count += 1
            continue
        if "t_t_t_" in llm_diffs[i]:
            skip_count += 1
            continue
        
        # check if non-ascci characters are present
        if not all(ord(char) < 128 for char in llm_diffs[i]):
            skip_count += 1
            continue

        curr_string = llm_diffs[i].strip()
        if "Q:" in curr_string: # check if new question is starting and filter out
            # find index of Q:
            q_index = llm_diffs[i].find("Q:")
            # new_diffs.append(llm_diffs[i][:q_index].strip())
            # new_indices.append(indices[i])
            curr_string = llm_diffs[i][:q_index].strip()
            early_count += 1


        ##### adding for v2
        if "I hope" in curr_string: # check if generic content is starting and remove
            q_index = llm_diffs[i].find("I hope")
            curr_string = llm_diffs[i][:q_index].strip()
            early_count += 1

        if "#2:" in curr_string:
            # find index of #2:
            start_index = llm_diffs[i].find("#2:")
            curr_string = llm_diffs[i][:start_index].strip()
            early_count += 1

        if "Note" in curr_string: # check if note is starting and filter out
            # find index of Note:
            note_index = llm_diffs[i].find("Note")
            curr_string = llm_diffs[i][:note_index].strip()
            early_count += 1

        new_diffs.append(curr_string)
        new_indices.append(indices[i])

    print("total skips", skip_count)
    print("early", early_count)
    
    # go through and remove additional hashtags
    for i, x in enumerate(new_diffs):
        to_add = x
        to_add = to_add.replace("#1:", "")
        to_add = to_add.replace("A: ", "")
        new_diffs[i] = to_add.replace("#", "").strip()

    # replace newline with space
    for i, x in enumerate(new_diffs):
        to_add = x
        to_add = to_add.replace("\n", "").strip()
        new_diffs[i] = to_add.replace("\t", " ").strip()

    # remove repeated spaces
    for i, x in enumerate(new_diffs):
        to_add = x
        new_diffs[i] = to_add.replace("  ", " ").strip()

    # if sentence starts with ". ", then remove
    for i, x in enumerate(new_diffs):
        to_add = x
        if to_add.startswith(". "):
            new_diffs[i] = to_add[2:].strip()

    filtered_new_diffs = []
    filtered_new_indices = []
    # remove instances that have a length shorter than 60
    for i, x in enumerate(new_diffs):
        if len(x) < 60:
            continue
        else:
            filtered_new_diffs.append(x)
            filtered_new_indices.append(new_indices[i])

    # save responses
    save_dict = {}
    save_dict["indices"] = filtered_new_indices
    save_dict["llm_diffs"] = filtered_new_diffs

    pickle.dump(save_dict, open("llm_diffs/coco_diffs_1000_filtered.pkl", "wb"))
    # write llm diffs to a text file
    with open("llm_diffs/coco_diffs_1000_filtered_v2.txt", "w") as f:
        for i, x in enumerate(filtered_new_diffs):
            f.write(x + "\n")