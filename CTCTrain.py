from data_load import loadData, loadDataPrimus

from model_templates.CRNNCTC import get_CRNN_CTC_model
from model_templates.CNNTRFCTC import get_CNNTransformer_CTC_model
from model_templates.ViTCTC import get_vit_model

import argparse
import numpy as np
from sklearn.utils import shuffle
import itertools
from utils import check_and_retrieveVocabulary, data_preparation_CTC
import cv2
import os
import tensorflow as tf
import random

import sys

os.environ["CUDA_VISIBLE_DEVICES"]="0"

config = tf.compat.v1.ConfigProto(gpu_options = 
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

fixed_height = 64

def parse_arguments():
    parser = argparse.ArgumentParser(description="Program arguments to work")
    parser.add_argument('--data_path', type=str, help="Corpus to be processed")
    parser.add_argument('--model_name', type=str, help="Model name")

    args = parser.parse_args()
    return args

def levenshtein(a,b):
    "Computes the Levenshtein distance between a and b."
    n, m = len(a), len(b)

    if n > m:
        a,b = b,a
        n,m = m,n

    current = range(n+1)
    for i in range(1,m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1,n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]

def getCTCValidationData(model, X, Y, i2w):
    acc_ed_ser = 0
    acc_len_ser = 0

    randomindex = random.randint(0, len(X)-1)

    for i in range(len(X)):
        pred = model.predict(np.expand_dims(np.expand_dims(X[i],axis=0),axis=-1))[0]

        out_best = np.argmax(pred,axis=1)

        # Greedy decoding (TODO Cambiar por la funcion analoga del backend de keras)
        out_best = [k for k, g in itertools.groupby(list(out_best))]
        decoded = []
        for c in out_best:
            if c < len(i2w):  # CTC Blank must be ignored
                decoded.append(i2w[c])

        groundtruth = [i2w[label] for label in Y[i]]

        if(i == randomindex):
            print(f"Prediction - {decoded}")
            print(f"True - {groundtruth}")

        acc_len_ser += len(Y[i])
        acc_ed_ser += levenshtein(decoded, groundtruth)


    ser = 100. * acc_ed_ser / acc_len_ser
    return ser

def CTCTest(model, X, Y, i2w):
    predictions = []
    true = []
    for i in range(len(X)):
       pred = model.predict(np.expand_dims(np.expand_dims(X[i],axis=0),axis=-1))[0]

       out_best = np.argmax(pred,axis=1)

       # Greedy decoding (TODO Cambiar por la funcion analoga del backend de keras)
       out_best = [k for k, g in itertools.groupby(list(out_best))]
       decoded = []
       for c in out_best:
           if c < len(i2w):  # CTC Blank must be ignored
               decoded.append(i2w[c])

       groundtruth = [i2w[label] for label in Y[i]]
       predictions.append(decoded)
       true.append(groundtruth)
    
    return predictions, true

def main():
    args = parse_arguments()
    
    XTrain = []
    YTrain = []

    XTrain, YTrain = loadDataPrimus(args.data_path)

    XTrain, YTrain = shuffle(XTrain, YTrain)

    Y_Encoded = []
    w2i, i2w = check_and_retrieveVocabulary([YTrain], "./vocab", args.model_name)
    
    for i in range(len(XTrain)):
        img = (255. - XTrain[i]) / 255.
        width = int(float(fixed_height * img.shape[1]) / img.shape[0])
        XTrain[i] = cv2.resize(img, (width, fixed_height))
        Y_Encoded.append([w2i[symbol] for symbol in YTrain[i]])         

    YTrain = np.array(Y_Encoded)
    print(XTrain.shape)
    print(YTrain.shape)

    #sys.exit()

    vocabulary_size = len(w2i)

    model_tr = None
    model_pr = None

    if args.model_name == "ViTCTC":
        model_tr, model_pr = get_vit_model(input_shape=(fixed_height,None,1),vocabulary_size=vocabulary_size, patch_size=fixed_height, model_depth=128)

    if args.model_name == "CNNTransformerCTC":
        model_tr, model_pr = get_CNNTransformer_CTC_model(input_shape=(fixed_height,None,1),vocabulary_size=vocabulary_size, 
                                                          transf_nheads=8, transf_depth=128, transf_ffunits=512)
    if args.model_name == "CRNNCTC":
        model_tr, model_pr = get_CNNTransformer_CTC_model(input_shape=(fixed_height,None,1),vocabulary_size=vocabulary_size)

    XValidate = XTrain[:int(len(XTrain)*0.25)]
    YValidate = YTrain[:int(len(YTrain)*0.25)]

    XTrain = XTrain[int(len(XTrain)*0.25):]
    YTrain = YTrain[int(len(YTrain)*0.25):]

    X_train, Y_train, L_train, T_train = data_preparation_CTC(XTrain, YTrain, fixed_height)

    print('Training with ' + str(X_train.shape[0]) + ' samples.')
    
    inputs = {'the_input': X_train,
                 'the_labels': Y_train,
                 'input_length': L_train,
                 'label_length': T_train,
                 }
    
    outputs = {'ctc': np.zeros([len(X_train)])}
    best_ser = 10000
    not_improved = 0

    for super_epoch in range(10000):
       model_tr.fit(inputs,outputs, batch_size = 16, epochs = 5, verbose = 1)
       SER = getCTCValidationData(model_pr, XValidate, YValidate, i2w)
       print(f"EPOCH {super_epoch} | SER {SER}")
       if SER < best_ser:
           print("SER improved - Saving epoch")
           model_pr.save_weights(f"{args.model_name}.h5")
           best_ser = SER
           not_improved = 0
       else:
           not_improved += 1
           if not_improved == 5:
               break
    
    model = model_pr.load_weights(f"{args.model_name}.h5")

    prediction_array, ground_truth = CTCTest(model, XValidate, YValidate, i2w)
    for idx, prediciton in enumerate(prediction_array):
        with open(f"test/img{idx}_results.txt", "w") as output_file_pr:
            output_file_pr.write("Prediction - " + " ".join(prediciton) + "\n")
            output_file_pr.write("True - " + " ".join(ground_truth[idx]) + "\n")
            cv2.imwrite(f"test/img{idx}.jpg", (XValidate[idx] * 255.))

if __name__ == "__main__":
    main()

    

    


