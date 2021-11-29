import json
import cv2
import numpy as np
import sys
import os
import tqdm

CONST_FOLDER = "Data"

def loadImage(path, x0, xf, y0, yf):
    #print(path)
    image = cv2.imread(path, 0)
    return image[y0:yf, x0:xf]

def loadData(path):
    print("Loading data...")
    X = []
    Y = []
    for file in os.listdir("Data/CTCMalaga/JSON"):
        print(f"Data/CTCMalaga/{file}")
        with open(f"Data/CTCMalaga/{file}") as jsonfile:
            print("Data/CTCMalaga/IMG/"+os.path.splitext(file)[0])
            data = json.load(jsonfile)
            for page in data["pages"]:
                for region in page["regions"]:
                    if region['type'] == 'staff' and "symbols" in region:
                        symbol_sequence = []
                        for s in region["symbols"]:
                            try:
                                symbol_sequence.append((s["agnostic_symbol_type"] + ":" + s["position_in_staff"], s["bounding_box"]["fromX"]))
                            except:
                                symbol_sequence.append((s["agnostic_symbol_type"] + ":" + s["position_in_staff"], s["approximateX"]))                        

                        sorted_symbols = sorted(symbol_sequence, key=lambda symbol: symbol[1])
                        sequence = [sym[0] for sym in sorted_symbols]
                        top, left, bottom, right = region["bounding_box"]["fromY"], \
                                                               region["bounding_box"]["fromX"], \
                                                               region["bounding_box"]["toY"], \
                                                               region["bounding_box"]["toX"]

                        selected_region = loadImage("Data/CTCMalaga/IMG/"+os.path.splitext(file)[0], left, right, top, bottom)
                        selected_region_augmented = loadImage("Data/CTCMalaga/IMG/"+os.path.splitext(file)[0], left-25, right+25, top-25, bottom+25)
                        if selected_region is not None:
                            X.append(selected_region)
                            #X.append(selected_region_augmented)
                            #Y.append(sequence)
                            Y.append(sequence)
    
    print("Data Loaded!")
    return np.array(X), np.array(Y)

def loadDataPrimus(path):
    X = []
    Y = []
    limit = 20000
    i = 0
    for folder in tqdm.tqdm(os.listdir(f"{path}")):
       img = cv2.imread(f"{path}/{folder}/{folder}_distorted.jpg", 0)
       X.append(img)
       with open(f"{path}/{folder}/{folder}.agnostic") as agnosticfile:
           Y.append(agnosticfile.readline().strip().split("\t"))
       
       if i > limit:
           break
       
       i+=1
    
    return np.array(X), np.array(Y)
