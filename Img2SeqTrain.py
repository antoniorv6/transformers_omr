import tensorflow as tf
import numpy as np
import argparse
import random

from data_load import loadData, loadDataPrimus
from sklearn.utils import shuffle
from model_templates.CNNTRF import get_cnn_transformer
from utils import check_and_retrieveVocabulary
import cv2
import tqdm
from sklearn.model_selection import train_test_split
import sys

config = tf.compat.v1.ConfigProto(gpu_options = 
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

fixed_height = 64
BATCH_SIZE = 16

def parse_arguments():
    parser = argparse.ArgumentParser(description="Program arguments to work")
    parser.add_argument('--data_path', type=str, help="Corpus to be processed")
    parser.add_argument('--model_name', type=str, help="Model name")
    parser.add_argument('--encoding', type=str, help="Encoding type")
    parser.add_argument('--corpus_name', type=str, help="Corpus name")


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

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    returnedseq = seq[:, tf.newaxis, tf.newaxis, :]
    # add extra dimensions to add the padding
    # to the attention logits.
    return returnedseq # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

def create_target_masks(tar):
    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tar.shape[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return combined_mask.numpy()

def batch_confection(X, Y, max_image_width, max_seq_len, w2i):
    #Add padding to the image
    Batch_X = np.zeros(shape=[len(X), fixed_height, max_image_width, 1], dtype=np.float32)
    Batch_Y = np.zeros(shape=(len(Y), max_seq_len + 1))
    Batch_T = np.zeros(shape=(len(Y), max_seq_len + 1))
    
    for i, img in enumerate(X):
        Batch_X[i, 0:img.shape[0], 0:img.shape[1],0] = img
    
    for i, seq in enumerate(Y):
        for j, char in enumerate(seq):
            Batch_Y[i][j] = random.randint(1, len(w2i)-1)
            if j > 0:
                Batch_T[i][j-1]= char
            
        Batch_Y[i][0] = w2i['<sos>']

    return Batch_X, Batch_Y, Batch_T

def batch_generator(X,Y, batch_size, max_image_width, max_seq_len, w2i):
    index = 0
    while True:
        BatchX = X[index:index + batch_size]
        BatchY = Y[index:index + batch_size]

        image_input, decoder_input, ground_truth = batch_confection(BatchX, BatchY,
                                                                    max_image_width, max_seq_len, w2i)
        combined_mask = create_target_masks(decoder_input)

        yield [image_input, decoder_input, combined_mask], ground_truth
        
        index = (index + batch_size) % len(X)
        if index > len(X): index = 0

def predict_sequence(model, image, w2i, i2w, max_seq_len, max_image_width):
    dec_in = [w2i['<sos>']]
    predSequence = ['<sos>']
    #target_padding_mask = create_target_masks(decoder_input)
    for i in range(max_seq_len):
        encoder_input, _, _ = batch_confection([image], [dec_in], max_image_width, max_seq_len, w2i, True)
        target_padding_mask = create_target_masks(np.asarray([dec_in]))
        inputs = [encoder_input, np.asarray([dec_in]), target_padding_mask]
        predictions = model.predict(inputs, batch_size=1)
        pred = predictions[0][-1]
        pred_id = np.argmax(pred)
        predSequence.append(i2w[pred_id])
        dec_in.append(pred_id)
        if i2w[pred_id] == '<eos>':
            break
    
    return predSequence

def validateModel(model, testGen, i2w, numOfBatches, encodingType):
    current_edition_val = 0
    predictionSequence = []
    truesequence = []
    for _ in tqdm.tqdm(range(numOfBatches)):
        inputs, ground_truth = next(testGen)
        predictions = model.predict(inputs, batch_size=BATCH_SIZE)
        for i, prediction in enumerate(predictions):
            raw_sequence = [i2w[char] for char in np.argmax(prediction, axis=1)]
            raw_trueseq = [i2w[char] for char in ground_truth[i]]
            predictionSequence = []
            truesequence = []

            # Turn into split-seq if standard
                

            for char in raw_sequence:
                
                if encodingType == 'standard':
                    predictionSequence += char.split(":")
                else:
                    predictionSequence += [char]
                
                if char == '<eos>':
                    break
            
            for char in raw_trueseq:
                if encodingType == 'standard':
                    truesequence += char.split(":")
                else:
                    truesequence += [char]

                if char == '<eos>':
                    break

            current_edition_val += levenshtein(truesequence, predictionSequence) / len(truesequence)

    print("Prediction: " + str(predictionSequence))
    print("True: " + str(truesequence))

    return current_edition_val

def main():
    args = parse_arguments()
    
    XTrain = []
    YTrain = []

    XTrain, YTrain = loadData(args.data_path, args.encoding)
    for i,sequence in enumerate(YTrain):
        YTrain[i] = ['<sos>'] + sequence + ['<eos>']

    Y_Encoded = []
    w2i, i2w = check_and_retrieveVocabulary([YTrain], f"./vocab/{args.corpus_name}_{args.encoding}", args.model_name)
    
    for i in range(len(XTrain)):
        img = (255. - XTrain[i]) / 255.
        width = int(float(fixed_height * img.shape[1]) / img.shape[0])
        XTrain[i] = cv2.resize(img, (width, fixed_height))
        Y_Encoded.append([w2i[symbol] for symbol in YTrain[i]])        

    YTrain = np.array(Y_Encoded)
    print(XTrain.shape)
    print(YTrain.shape)

    max_image_width = max([img.shape[1] for img in XTrain])
    #max_image_height = max([img.shape[0] for img in XTrain])
    max_length_seq = max([len(w) for w in YTrain])
    print(max_length_seq)
    
    model = None
    if args.model_name == "CNNTransformer":
         model = get_cnn_transformer(conv_filters=[4,8,16,16],
                              num_convs=[1,1,2,2],
                              kernel_sizes= [(5,5),(3,3),(3,3),(3,3)],
                              pool_layers=[(2,2),(2,2),(2,2),(2,2)],
                              image_input_shape=(fixed_height, None, 1),
                              MAX_SEQ_LEN=max_length_seq,
                              VOCAB_SIZE=len(w2i),
                              transformer_encoder_layers=1,
                              transformer_decoder_layers=1,
                              ff_depth=128,
                              num_heads=8,
                              POS_ENCODING_INPUT=max_image_width,
                              POS_ENCODING_TARGET=512)

    
    XTrain, XVal, YTrain, YVal = train_test_split(XTrain, YTrain, test_size=0.5, random_state=0)
    XVal, XTest, YVal, YTest = train_test_split(XVal, YVal, test_size=0.5, random_state=0)


    batch_gen = batch_generator(XTrain, YTrain, BATCH_SIZE, max_image_width, max_length_seq, w2i)
    val_gen = batch_generator(XVal, YVal, BATCH_SIZE, max_image_width, max_length_seq, w2i)
    test_gen = batch_generator(XTest, YTest, BATCH_SIZE, max_image_width, max_length_seq, w2i)

    best_ser = 10000
    for SUPER_EPOCH in range(10000):
        model.fit(batch_gen, steps_per_epoch=len(XTrain)//BATCH_SIZE, epochs=20, verbose=1)
        edition_val_train = validateModel(model, batch_gen, i2w, len(XTrain)//BATCH_SIZE, args.encoding)
        edition_val = validateModel(model, val_gen, i2w, len(XVal)//BATCH_SIZE, args.encoding)
        edition_test = validateModel(model, test_gen, i2w, len(XTest)//BATCH_SIZE, args.encoding)
        SER_TRAIN = (100. * edition_val_train) / len(XTrain)
        SER = (100. * edition_val) / len(XVal)
        SER_TEST = (100. * edition_test) / len(XTest)

        print(f'| Epoch {(SUPER_EPOCH + 1)} | Train SER: {SER_TRAIN} | Validation SER: {SER} | Test SER: {SER_TEST}')
        
        if SER < best_ser:
            print("SER improved, saving model")
            model.save_weights(f"model{args.model_name}_{args.corpus_name}_{args.encoding}.h5")
            best_ser = SER
            tries = 0
        elif SUPER_EPOCH > 15:
            print(f"SER not improved, remaining attempts: {5 - tries}")
            tries += 1
            if tries > 5:
                break

        print(f"SUPER EPOCH {SUPER_EPOCH+1} | SER VALIDATION {SER} | SER TEST {SER_TEST}")



if __name__ == "__main__":
    main()