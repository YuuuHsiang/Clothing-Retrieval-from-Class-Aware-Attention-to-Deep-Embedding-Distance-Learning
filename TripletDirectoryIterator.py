from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from random import shuffle
import numpy as np
import numpy.random as rng
import os
import re

splitter = re.compile("\s+")


class TripletDirectoryIterator(image.DirectoryIterator):
    def __init__(self, directory, image_data_generator,
                 bounding_boxes: dict = None, landmark_info: dict = None, attr_info: dict = None,
                 num_landmarks=26, num_attrs=463,
                 target_size=(256, 256), color_mode: str = 'rgb',
                 classes=None, class_mode: str = 'categorical',
                 batch_size: int = 32, shuffle: bool = True, seed=None, data_format=None,
                 follow_links: bool = False):
        super().__init__(directory, image_data_generator, target_size, color_mode, classes, class_mode, batch_size,
                         shuffle, seed, data_format, follow_links)
        self.bounding_boxes = bounding_boxes
        self.landmark_info = landmark_info
        self.attr_info = attr_info
        self.num_landmarks = num_landmarks
        self.num_attrs = num_attrs
        self.num_bbox = 4

    def next(self):
        """
        # Returns
            The next batch.
        """

        locations = np.zeros((self.batch_size,) + (self.num_bbox,), dtype=K.floatx())
        landmarks = np.zeros((self.batch_size,) + (self.num_landmarks,), dtype=K.floatx())
        attributes = np.zeros((self.batch_size,) + (self.num_attrs,), dtype=K.floatx())

        # initialize KN for the input image batch
        # K = 5 (range = 7(5 + anchor + positive))
        pairs = [np.zeros((self.batch_size, self.target_size[0], self.target_size[1], 3)) for _ in range(7)]
        batch_y = [np.zeros((self.batch_size, self.num_classes)) for _ in range(7)]

        # K = 3 (range = 5(3 + anchor + positive)))
        # pairs = [np.zeros((self.batch_size, self.target_size[0], self.target_size[1], 3)) for _ in range(5)]
        # batch_y = [np.zeros((self.batch_size, self.num_classes)) for _ in range(5)]

        # Randomly selected images(A, P, KN)
        for i in range(self.batch_size):
            # Pick anchor image
            # print("Anchor image")
            a = '\\'
            idx_1 = rng.randint(0, self.samples)
            anchor_item_idx = str(self.filenames[idx_1]).split(a)[-2]
            pairs[0][i, :, :, :] = self.get_image(idx_1)
            batch_y[0][i, self.classes[idx_1]] = 1
            # print(anchor_item_idx)

            # pick positive and negative samples to anchor image.
            # print("Positive image")
            idx_2 = rng.randint(0, self.samples)
            positive_item_idx = str(self.filenames[idx_2]).split(a)[-2]
            while positive_item_idx != anchor_item_idx:
                idx_2 = rng.randint(0, self.samples)
                positive_item_idx = str(self.filenames[idx_2]).split(a)[-2]

            # print(positive_item_idx)
            pairs[1][i, :, :, :] = self.get_image(idx_2)
            batch_y[1][i, self.classes[idx_2]] = 1

            # print("Negative image 1")
            idx_3 = rng.randint(0, self.samples)
            negative_item_idx_1 = str(self.filenames[idx_3]).split(a)[-2]
            while negative_item_idx_1 == anchor_item_idx:
                idx_3 = rng.randint(0, self.samples)
                negative_item_idx_1 = str(self.filenames[idx_3]).split(a)[-2]

            # print(negative_item_idx)
            pairs[2][i, :, :, :] = self.get_image(idx_3)
            batch_y[2][i, self.classes[idx_3]] = 1

            # print("Negative image 2")
            idx_4 = rng.randint(0, self.samples)
            negative_item_idx_2 = str(self.filenames[idx_4]).split(a)[-2]
            while negative_item_idx_2 == anchor_item_idx or negative_item_idx_2 == negative_item_idx_1:
                idx_4 = rng.randint(0, self.samples)
                negative_item_idx_2 = str(self.filenames[idx_4]).split(a)[-2]

            # print(negative_item_idx)
            pairs[3][i, :, :, :] = self.get_image(idx_4)
            batch_y[3][i, self.classes[idx_4]] = 1

            # print("Negative image 3")
            idx_5 = rng.randint(0, self.samples)
            negative_item_idx_3 = str(self.filenames[idx_5]).split(a)[-2]
            while negative_item_idx_3 == anchor_item_idx or negative_item_idx_3 == negative_item_idx_1 or negative_item_idx_3 == negative_item_idx_2:
                idx_5 = rng.randint(0, self.samples)
                negative_item_idx_3 = str(self.filenames[idx_5]).split(a)[-2]

            # print(negative_item_idx)
            pairs[4][i, :, :, :] = self.get_image(idx_5)
            batch_y[4][i, self.classes[idx_5]] = 1

            # print("Negative image 4")
            # idx_6 = rng.randint(0, self.samples)
            # negative_item_idx_4 = str(self.filenames[idx_3]).split(a)[-2]
            # while negative_item_idx_4 == anchor_item_idx or negative_item_idx_4 == negative_item_idx_3 or \
            #         negative_item_idx_4 == negative_item_idx_2 or negative_item_idx_4 == negative_item_idx_1:
            #     idx_6 = rng.randint(0, self.samples)
            #     negative_item_idx_4 = str(self.filenames[idx_6]).split(a)[-2]
            #
            # # print(negative_item_idx)
            # pairs[5][i, :, :, :] = self.get_image(idx_6)
            # batch_y[5][i, self.classes[idx_6]] = 1
            #
            # # print("Negative image 5")
            # idx_7 = rng.randint(0, self.samples)
            # negative_item_idx_5 = str(self.filenames[idx_7]).split(a)[-2]
            # while negative_item_idx_5 == anchor_item_idx or negative_item_idx_5 == negative_item_idx_4 or negative_item_idx_5 == negative_item_idx_3 or \
            #         negative_item_idx_5 == negative_item_idx_2 or negative_item_idx_5 == negative_item_idx_1:
            #     idx_7 = rng.randint(0, self.samples)
            #     negative_item_idx_5 = str(self.filenames[idx_7]).split(a)[-2]
            #
            # # print(negative_item_idx)
            # pairs[6][i, :, :, :] = self.get_image(idx_7)
            # batch_y[6][i, self.classes[idx_7]] = 1

            if self.bounding_boxes is not None:
                locations[i] = (self.get_bbox(self.filenames[idx_1]),
                                self.get_bbox(self.filenames[idx_2]),
                                self.get_bbox(self.filenames[idx_3]))

            if self.landmark_info is not None:
                landmarks[i] = (self.get_landmark_info(self.filenames[idx_1]),
                                self.get_landmark_info(self.filenames[idx_2]),
                                self.get_landmark_info(self.filenames[idx_3]))

            if self.attr_info is not None:
                attr_info_lst_1 = self.attr_info[self.filenames[idx_1]]
                attr_info_lst_2 = self.attr_info[self.filenames[idx_2]]
                attr_info_lst_3 = self.attr_info[self.filenames[idx_3]]
                attributes[i] = (np.asarray(attr_info_lst_1), np.asarray(attr_info_lst_2), np.asarray(attr_info_lst_3))

        if self.shuffle:
            self.shuffle_batches(batch_y, pairs)

        pairs = np.asarray(pairs)

        # y = [batch_y, locations, landmarks, attributes]
        # statements = [True, self.bounding_boxes is not None,
        #               self.landmark_info is not None, self.attr_info is not None]
        #
        # y = np.asarray([y_ for y_, s in zip(y, statements) if s]).reshape((self.batch_size,))
        return pairs, batch_y

    # defined batches
    @staticmethod
    def shuffle_batches(batch_y, pairs):
        anchor_img = pairs[0]
        anchor_y = batch_y[0]
        positive_img = pairs[1]
        positive_y = batch_y[1]
        negative_img_1 = pairs[2]
        negative_y_1 = batch_y[2]
        negative_img_2 = pairs[3]
        negative_y_2 = batch_y[3]
        negative_img_3 = pairs[4]
        negative_y_3 = batch_y[4]
        # negative_img_4 = pairs[5]
        # negative_y_4 = batch_y[5]
        # negative_img_5 = pairs[6]
        # negative_y_5 = batch_y[6]

        # zip batch
        tmp = list(zip(anchor_img, positive_img, negative_img_1, negative_img_2, negative_img_3, anchor_y, positive_y,
                       negative_y_1, negative_y_2, negative_y_3))
        shuffle(tmp)
        anchor_img, positive_img, negative_img_1, negative_img_2, negative_img_3, anchor_y, positive_y, negative_y_1, negative_y_2, negative_y_3 = zip(
            *tmp)
        # tmp = list(zip(anchor_img, positive_img, negative_img_1, negative_img_2, negative_img_3, negative_img_4, negative_img_5, anchor_y, positive_y, negative_y_1, negative_y_2, negative_y_3, negative_y_4, negative_y_5))
        # shuffle(tmp)
        # anchor_img, positive_img, negative_img_1, negative_img_2, negative_img_3, negative_img_4, negative_img_5, anchor_y, positive_y, negative_y_1, negative_y_2, negative_y_3, negative_y_4, negative_y_5 = zip(*tmp)
        pairs[0] = np.array(anchor_img)
        pairs[1] = np.array(positive_img)
        pairs[2] = np.array(negative_img_1)
        pairs[3] = np.array(negative_img_2)
        pairs[4] = np.array(negative_img_3)
        # pairs[5] = np.array(negative_img_4)
        # pairs[6] = np.array(negative_img_5)
        batch_y[0] = np.array(anchor_y)
        batch_y[1] = np.array(positive_y)
        batch_y[2] = np.array(negative_y_1)
        batch_y[3] = np.array(negative_y_2)
        batch_y[4] = np.array(negative_y_3)
        # batch_y[5] = np.array(negative_y_4)
        # batch_y[6] = np.array(negative_y_5)

    def get_image(self, idx):
        fname = self.filenames[idx]
        # print("Category: " + str(self.classes[idx_2]) + ", Filename: " + str(fname_2) + "\n")
        img = image.load_img(os.path.join(self.directory, fname),
                             grayscale=self.color_mode == 'grayscale',
                             target_size=self.target_size)
        img = image.img_to_array(img, data_format=self.data_format)
        img = self.image_data_generator.random_transform(img)
        img = self.image_data_generator.standardize(img)
        return img

    def get_bbox(self, fname):
        bbox = self.bounding_boxes[fname]
        return np.asarray([bbox['origin']['x'], bbox['origin']['y'], bbox['width'], bbox['height']], dtype=K.floatx())

    def get_landmark_info(self, fname):
        landmark_info = self.landmark_info[fname]
        return np.asarray([landmark_info["clothes_type"], landmark_info["variation_type"],
                           landmark_info['1']['visibility'], landmark_info['1']['x'],
                           landmark_info['1']['y'],
                           landmark_info['2']['visibility'], landmark_info['2']['x'],
                           landmark_info['2']['y'],
                           landmark_info['3']['visibility'], landmark_info['3']['x'],
                           landmark_info['3']['y'],
                           landmark_info['4']['visibility'], landmark_info['4']['x'],
                           landmark_info['4']['y'],
                           landmark_info['5']['visibility'], landmark_info['5']['x'],
                           landmark_info['5']['y'],
                           landmark_info['6']['visibility'], landmark_info['6']['x'],
                           landmark_info['6']['y'],
                           landmark_info['7']['visibility'], landmark_info['7']['x'],
                           landmark_info['7']['y'],
                           landmark_info['8']['visibility'], landmark_info['8']['x'],
                           landmark_info['8']['y']], dtype=K.floatx())
