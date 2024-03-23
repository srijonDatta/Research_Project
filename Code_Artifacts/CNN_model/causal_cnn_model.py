import sys, os, random
import numpy as np
import keras
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D
from keras.layers.merge import concatenate
from tensorflow.keras import layers
import tensorflow as tf
from sklearn.metrics import accuracy_score


if len(sys.argv) < 4:
    print('Needs 3 arguments - \n'
          '1. Batch size during training\n'
          '2. Batch size during testing\n'
          '3. No. of epochs\n')
    exit(0)

seed_value = 12321
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
np.random.seed(seed_value)

# command line both for train and test
DATADIR_train = '/store/causalIR/train_hist/'   # (1)
DATADIR_test = '/store/causalIR/test_hist/'   # (1)

# A matrix is treated a grayscale image, i.e. am image with num_channels = 1
NUMCHANNELS = 1
# HIDDEN_LAYER_DIM = 16
# Num top docs (Default: 10)
K = 1
# M: bin-size (Default: 30)
M = 120 # depends on max query length
BATCH_SIZE_TRAIN = int(sys.argv[1])   # (7 - depends on the total no. of ret docs)
BATCH_SIZE_TEST = int(sys.argv[2])
EPOCHS = int(sys.argv[3])  # (8)


class InteractionData:
    # Interaction data of query qid with K top docs -
    # each row vector is a histogram of interaction data for a document

    def __init__(self, docid, dataPathBase=DATADIR_train):
        self.docid = docid
        histFile = "{}/{}.hist".format(dataPathBase, self.docid)
        # df = pd.read_csv(histFile, delim_whitespace=True, header=None)
        # self.matrix = df.to_numpy()
        histogram = np.genfromtxt(histFile, delimiter=" ")
        self.matrix = histogram[:, 4:]


class PairedInstance:
    def __init__(self, line):
        l = line.strip().split('\t')
        if len(l) > 2:
            self.doc_a = l[0]
            self.doc_b = l[1]
            self.class_label = int(l[2])
        else:
            self.doc_a = l[0]
            self.doc_b = l[1]

    def __str__(self):
        return "({}, {})".format(self.qid_a, self.qid_b)

    def getKey(self):
        return "{}-{}".format(self.qid_a, self.qid_b)


# Separate instances for training/test sets etc. Load only the id pairs.
# Data is loaded later in batches with a subclass of Keras generator
class PairedInstanceIds:
    '''
    Each line in this file should comprise three tab separated fields
    <id1> <id2> <label (1/0)>
    '''

    def __init__(self, idpairLabelsFile):
        self.data = {}

        with open(idpairLabelsFile) as f:
            content = f.readlines()

        # remove whitespace characters like `\n` at the end of each line
        for x in content:
            instance = PairedInstance(x)
            self.data[instance.getKey()] = instance

allPairs_train = PairedInstanceIds(DATADIR + 'train_input/rel_label.pairs')   # (3)
allPairsList_train = list(allPairs_train.data.values())

allPairs_test = PairedInstanceIds(DATADIR + 'test_input/rel_label.pairs')    # (4)
allPairsList_test = list(allPairs_test.data.values())

print ('{}/{} pairs for training'.format(len(allPairsList_train), len(allPairsList_train)))
print ('{}/{} pairs for testing'.format(len(allPairsList_test), len(allPairsList_test)))

'''
The files need to be residing in the folder data/
Each file is a matrix of values that's read using
'''

class PairCmpDataGeneratorTrain(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, paired_instances_ids, dataFolder=DATADIR_train, batch_size=BATCH_SIZE_TRAIN, dim_rel=(K, M, NUMCHANNELS)):
        'Initialization'
        self.paired_instances_ids = paired_instances_ids
        self.dim = [dim_rel, dim_nonrel]
        self.batch_size = batch_size
        self.dataDir = dataFolder
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.paired_instances_ids) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs = [self.paired_instances_ids[k] for k in indexes]

        # Generate data
        X = self.__data_generation(list_IDs)

        return X

    def on_epoch_end(self):
        'Update indexes after each epoch'
        self.indexes = np.arange(len(self.paired_instances_ids))

    def __data_generation(self, list_IDs):
        'Generates data pairs containing batch_size samples'
        # Initialization
        X = [np.empty((self.batch_size, *self.dim[i])) for i in range(4)]
        Z = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, paired_instance in enumerate(list_IDs):
            a_id = paired_instance.docid_rel
            b_id = paired_instance.docid_nonrel

            # read from the data file and construct the instances
            a_data = InteractionData(a_id, self.dataDir)
            a_data_top = a_data.matrix[0:5, :]

            b_data = InteractionData(b_id, self.dataDir)
            b_data_top = b_data.matrix[0:5, :]

            w, h = a_data.matrix.shape
            w_top, h_top = a_data_top.shape
            a_data_top = a_data_top.reshape(w_top, h_top, 1)

            b_data_top = b_data_top.reshape(w_top, h_top, 1)

            X[0][i,] = a_data_top
            X[2][i,] = b_data_top
            Z[i] = paired_instance.class_label

        return X, Z


class PairCmpDataGeneratorTest(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, paired_instances_ids, dataFolder=DATADIR_idf, batch_size=BATCH_SIZE_TRAIN, dim_rel=(K, M, NUMCHANNELS)):
        'Initialization'
        self.paired_instances_ids = paired_instances_ids
        self.dim = [dim_rel, dim_nonrel]
        self.batch_size = batch_size
        self.dataDir = dataFolder
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.paired_instances_ids) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs = [self.paired_instances_ids[k] for k in indexes]

        # Generate data
        X = self.__data_generation(list_IDs)

        return X

    def on_epoch_end(self):
        'Update indexes after each epoch'
        self.indexes = np.arange(len(self.paired_instances_ids))

    def __data_generation(self, list_IDs):
        'Generates data pairs containing batch_size samples'
        # Initialization
        X = [np.empty((self.batch_size, *self.dim[i])) for i in range(4)]
        Z = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, paired_instance in enumerate(list_IDs):
            a_id = paired_instance.doc_rel
            b_id = paired_instance.doc_nonrel

            # read from the data file and construct the instances
            a_data = InteractionData(a_id, self.dataDir)
            a_data_rel = a_data.matrix[0:5, :]

            b_data = InteractionData(b_id, self.dataDir)
            b_data_rel = b_data.matrix[0:5, :]

            w, h = a_data.matrix.shape
            w_top, h_top = a_data_rel.shape
            a_data_rel = a_data_rel.reshape(w_top, h_top, 1)

            # b_data.matrix = b_data.matrix.reshape(w, h, 1)
            b_data_rel = b_data_rel.reshape(w_top, h_top, 1)

            X[0][i,] = a_data_rel
            X[2][i,] = b_data_rel
            # Z[i] = paired_instance.class_label

        return X


def build_siamese(input_shape_top, input_shape_bottom):
    input_a_top = Input(shape=input_shape_top, dtype='float32')
    input_a_bottom = Input(shape=input_shape_bottom, dtype='float32')

    input_b_top = Input(shape=input_shape_top, dtype='float32')
    input_b_bottom = Input(shape=input_shape_bottom, dtype='float32')

    matrix_encoder_top = Sequential(name='sequence_1')
    matrix_encoder_top.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape_top))
    matrix_encoder_top.add(MaxPooling2D(padding='same'))
    matrix_encoder_top.add(Flatten())
    matrix_encoder_top.add(Dropout(0.2))
    matrix_encoder_top.add(Dense(128, activation='relu'))

    matrix_encoder_bottom = Sequential(name='sequence_2')
    matrix_encoder_bottom.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape_bottom))
    matrix_encoder_bottom.add(MaxPooling2D(padding='same'))
    matrix_encoder_bottom.add(Flatten())
    matrix_encoder_bottom.add(Dropout(0.2))
    matrix_encoder_bottom.add(Dense(128, activation='relu'))

    encoded_a_top = matrix_encoder_top(input_a_top)
    encoded_a_bottom = matrix_encoder_bottom(input_a_bottom)
    merged_vector_a = concatenate([encoded_a_top, encoded_a_bottom], axis=-1, name='concatenate_1')

    encoded_b_top = matrix_encoder_top(input_b_top)
    encoded_b_bottom = matrix_encoder_bottom(input_b_bottom)
    merged_vector_b = concatenate([encoded_b_top, encoded_b_bottom], axis=-1, name='concatenate_2')

    # ==============================

    merged_vector = concatenate([merged_vector_a, merged_vector_b], axis=-1, name='concatenate_final')
    # And add a logistic regression (2 class - sigmoid) on top
    # used for backpropagating from the (pred, true) labels
    predictions = Dense(1, activation='sigmoid')(merged_vector)

    siamese_net = Model([input_a_top, input_a_bottom, input_b_top, input_b_bottom], outputs=predictions)
    return siamese_net

siamese_model = build_siamese((K, M, 1))
siamese_model.compile(loss = keras.losses.BinaryCrossentropy(),
                      optimizer = keras.optimizers.Adam(),
                      metrics=['accuracy'])
siamese_model.summary()

training_generator = PairCmpDataGeneratorTrain(allPairsList_train, dataFolder=DATADIR_idf+'train_input/')
siamese_model.fit_generator(generator=training_generator,
                            use_multiprocessing=True,
                            epochs=EPOCHS,
                            workers=4)

# siamese_model.save_weights('/store/causalIR/foo.weights')
test_generator = PairCmpDataGeneratorTest(allPairsList_test, dataFolder=DATADIR_idf+'test_input/')
predictions = siamese_model.predict(test_generator)  # just to test, will rerank LM-scored docs
# print('predict ::: ', predictions)
# print('predict shape ::: ', predictions.shape)
with open(DATADIR + "12april.test.res", 'w') as outFile:     # (9)
    i = 0
    for entry in test_generator.paired_instances_ids:
        if predictions[i][0] >= 0.45:
            outFile.write(entry.qid_a + '\t' + entry.qid_b + '\t' + str(round(predictions[i][0], 4)) + '\t' + '1\n')
        else:
            outFile.write(entry.qid_a + '\t' + entry.qid_b + '\t' + str(round(predictions[i][0], 4)) + '\t' + '0\n')
        i += 1
outFile.close()

# measure accuracy
gt_file = np.genfromtxt('/store/causalIR/model-aware-qpp/CW-experiments/test_ap.pairs.gt', delimiter='\t')    # (10)
actual = gt_file[:, 2:]

predict_file = np.genfromtxt('/store/causalIR/model-aware-qpp/CW-experiments/unsupervised_res/maxIDF.pairs', delimiter='\t')
predict = predict_file[:, 2:]

score = accuracy_score(actual, predict)
print('Accuracy : ', round(score, 4))