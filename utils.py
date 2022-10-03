import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras.metrics import kullback_leibler_divergence
from tensorflow.keras.preprocessing import image
from TripletDirectoryIterator import TripletDirectoryIterator
from config import get_arguments
import tensorflow as tf
import keras
# import numpy as np
from sklearn.utils.extmath import cartesian
import math
# from scipy.spatial.distance import cosine
import numpy as np

args = get_arguments()

# Traditional Triplet Loss
def triplet_euclidean_loss(y_true, y_pred):
    enc_size = int(K.get_variable_shape(y_pred)[1]/3)
    anchor_encoding = y_pred[:, :enc_size]
    positive_encoding = y_pred[:, enc_size:2 * enc_size]
    negative_encoding = y_pred[:, 2 * enc_size:]
    margin = K.constant(2.0)

    def euclidean_dist(a, e):
        return K.sum(K.square(a - e), axis=-1)  # squared euclidean distance
        # return K.sqrt(K.sum(K.square(a - e), axis=-1))  # original euclidean distance

    pos_dist = euclidean_dist(anchor_encoding, positive_encoding)
    neg_dist = euclidean_dist(anchor_encoding, negative_encoding)
    basic_loss = pos_dist - neg_dist + margin

    return K.mean(K.maximum(basic_loss, 0.0))


# KN Loss
def triplet_euclidean_K_negative_loss(y_true, y_pred):

    # split embedding vector
    # enc_size = int(keras.backend.get_variable_shape(y_pred)[1] / 7)
    enc_size = int(keras.backend.get_variable_shape(y_pred)[1] / 5)
    # enc_size = int(keras.backend.get_variable_shape(y_pred)[1] / 3)

    anchor_encoding = y_pred[:, :enc_size]
    positive_encoding = y_pred[:, enc_size:2 * enc_size]
    # negative_encoding_1 = y_pred[:, 2 * enc_size:]
    negative_encoding_1 = y_pred[:, 2 * enc_size:3 * enc_size]
    negative_encoding_2 = y_pred[:, 3 * enc_size:4 * enc_size]
    negative_encoding_3 = y_pred[:, 4 * enc_size:]
    # negative_encoding_3 = y_pred[:, 4 * enc_size:5 * enc_size]
    # negative_encoding_4 = y_pred[:, 5 * enc_size:6 * enc_size]
    # negative_encoding_5 = y_pred[:, 6 * enc_size:]

    # margin
    margin = K.constant(3.0)

    # euclidean distance
    def euclidean_dist(a, p):
        return K.sum(K.square(a - p), axis=-1)  # squared euclidean distance
            # return K.sqrt(K.sum(K.square(a - e), axis=-1))  # original euclidean distance

    # QN dist
    pos_dist = euclidean_dist(anchor_encoding, positive_encoding)
    neg_dist_1 = euclidean_dist(anchor_encoding, negative_encoding_1)
    neg_dist_2 = euclidean_dist(anchor_encoding, negative_encoding_2)
    neg_dist_3 = euclidean_dist(anchor_encoding, negative_encoding_3)
    # neg_dist_4 = euclidean_dist(anchor_encoding, negative_encoding_4)
    # neg_dist_5 = euclidean_dist(anchor_encoding, negative_encoding_5)

    # PN dist
    pos_neg_dist_1 = euclidean_dist(positive_encoding, negative_encoding_3)
    pos_neg_dist_2 = euclidean_dist(positive_encoding, negative_encoding_3)
    pos_neg_dist_3 = euclidean_dist(positive_encoding, negative_encoding_3)

    # QN dist loss
    basic_loss_1 = K.maximum(pos_dist - neg_dist_1 + margin, 0.0)
    basic_loss_2 = K.maximum(pos_dist - neg_dist_2 + margin, 0.0)
    basic_loss_3 = K.maximum(pos_dist - neg_dist_3 + margin, 0.0)
    # basic_loss_4 = K.maximum(pos_dist - neg_dist_4 + margin, 0.0)
    # basic_loss_5 = K.maximum(pos_dist - neg_dist_5 + margin, 0.0)

    # PN dist loss
    pos_basic_loss_1 = K.maximum(pos_dist - pos_neg_dist_1 + margin, 0.0)
    pos_basic_loss_2 = K.maximum(pos_dist - pos_neg_dist_2 + margin, 0.0)
    pos_basic_loss_3 = K.maximum(pos_dist - pos_neg_dist_3 + margin, 0.0)

    all_dist = [basic_loss_1, basic_loss_2, basic_loss_3, pos_basic_loss_1, pos_basic_loss_2, pos_basic_loss_3]
    total_loss = 0

    for basic_loss in range(len(all_dist)):
        total_loss += K.mean(K.pow(K.exp(all_dist[basic_loss]), 0.5))

    return total_loss


def squash(activations, axis=-1):
    scale = K.sum(K.square(activations), axis, keepdims=True) / \
            (1 + K.sum(K.square(activations), axis, keepdims=True)) / \
            K.sqrt(K.sum(K.square(activations), axis, keepdims=True) + K.epsilon())
    return scale * activations


def decay_lr(lr, rate):
    return lr * rate


def custom_generator_K_negative(it):
    while True:
        pairs_batch, y_batch = it.next()
        # yield ([pairs_batch[0], pairs_batch[1], pairs_batch[2], pairs_batch[3], pairs_batch[4], y_batch[0]],
        #        [y_batch[0], y_batch[0], y_batch[1], y_batch[2], pairs_batch[3], pairs_batch[4]])
        yield ([pairs_batch[0], pairs_batch[1], pairs_batch[2], pairs_batch[3], pairs_batch[4], y_batch[0]],
        [y_batch[0], y_batch[0]])#, y_batch[0], y_batch[1], y_batch[2], y_batch[3], y_batch[4]])
        # yield ([pairs_batch[0], pairs_batch[1], pairs_batch[2], pairs_batch[3], pairs_batch[4], pairs_batch[5], pairs_batch[6], y_batch[0]],
        #        [y_batch[0], y_batch[0]])


def get_iterator(file_path, input_size=256, batch_size=32,
                 shift_fraction=0., h_flip=False, zca_whit=False, rot_range=0.,
                 bright_range=0., shear_range=0., zoom_range=0.):
    data_gen = image.ImageDataGenerator(width_shift_range=shift_fraction,
                                        height_shift_range=shift_fraction,
                                        horizontal_flip=h_flip,
                                        zca_whitening=zca_whit,
                                        rotation_range=rot_range,
                                        brightness_range=bright_range,
                                        shear_range=shear_range,
                                        zoom_range=zoom_range,
                                        rescale=1./255)
    t_iterator = TripletDirectoryIterator(directory=file_path, image_data_generator=data_gen,
                                          batch_size=batch_size, target_size=(input_size, input_size))

    return t_iterator
