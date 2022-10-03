from tensorflow.keras import backend as K
from tensorflow.keras import models, layers
from layers import Mask, Length
from blocks import capsule_model, vgg_16_pretrained_tensorflow

class MultiGPUNet(models.Model):
    def __init__(self, ser_model, gpus):
        # pmodel = multi_gpu_model(ser_model, gpus)
        pmodel = 0
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(MultiGPUNet, self).__getattribute__(attrname)


def Clothing_Retrieval_with_Class_Aware_Attention_and_K_negative(input_shape, args):
    x = layers.Input(shape=input_shape)

    # Capsule network
    caps_model = models.Model(x, capsule_model(x, args))
    caps_model.summary()

    # Input(Anchor、Positive、Negative)
    x1 = layers.Input(shape=input_shape)
    x2 = layers.Input(shape=input_shape)
    x3 = layers.Input(shape=input_shape)
    x4 = layers.Input(shape=input_shape)
    x5 = layers.Input(shape=input_shape)
    # x6 = layers.Input(shape=input_shape)
    # x7 = layers.Input(shape=input_shape)

    # Capsule encoding
    anchor_encoding = caps_model(x1)
    positive_encoding = caps_model(x2)
    negative_encoding_1 = caps_model(x3)
    negative_encoding_2 = caps_model(x4)
    negative_encoding_3 = caps_model(x5)
    # negative_encoding_4 = caps_model(x6)
    # negative_encoding_5 = caps_model(x7)

    # Vgg16 model
    vgg16 = vgg_16_pretrained_tensorflow(args)
    vgg16.summary()

    # Vgg16 encoding
    vgg16_anchor_encoding = vgg16(x1)
    vgg16_positive_encoding = vgg16(x2)
    vgg16_negative_encoding_1 = vgg16(x3)
    vgg16_negative_encoding_2 = vgg16(x4)
    vgg16_negative_encoding_3 = vgg16(x5)

    # Capsule l2 norm output
    # shape: (None, NUM_CLASS, DIM_CAPSULE)
    l2_norm = layers.Lambda(lambda enc: K.l2_normalize(enc, axis=-1) + K.epsilon())
    l2_anchor_encoding = l2_norm(anchor_encoding)
    l2_positive_encoding = l2_norm(positive_encoding)
    l2_negative_encoding_1 = l2_norm(negative_encoding_1)
    l2_negative_encoding_2 = l2_norm(negative_encoding_2)
    l2_negative_encoding_3 = l2_norm(negative_encoding_3)
    # l2_negative_encoding_4 = l2_norm(negative_encoding_4)
    # l2_negative_encoding_5 = l2_norm(negative_encoding_5)

    y1 = layers.Input(shape=(args.num_class,))

    # Flatten capsule features
    masked_anchor_encoding = Mask(name="anchor_mask")([l2_anchor_encoding, y1])
    masked_positive_encoding = Mask(name="positive_mask")([l2_positive_encoding, y1])
    masked_negative_encoding_1 = Mask(name="negative_mask_1")([l2_negative_encoding_1, y1])
    masked_negative_encoding_2 = Mask(name="negative_mask_2")([l2_negative_encoding_2, y1])
    masked_negative_encoding_3 = Mask(name="negative_mask_3")([l2_negative_encoding_3, y1])
    # masked_negative_encoding_4 = Mask(name="negative_mask_4")([l2_negative_encoding_4, y1])
    # masked_negative_encoding_5 = Mask(name="negative_mask_5")([l2_negative_encoding_5, y1])

    # Concate two branch
    out1 = layers.Concatenate()([masked_anchor_encoding, vgg16_anchor_encoding])
    out2 = layers.Concatenate()([masked_positive_encoding, vgg16_positive_encoding])
    out3 = layers.Concatenate()([masked_negative_encoding_1, vgg16_negative_encoding_1])
    out4 = layers.Concatenate()([masked_negative_encoding_2, vgg16_negative_encoding_2])
    out5 = layers.Concatenate()([masked_negative_encoding_3, vgg16_negative_encoding_3])
    # out6 = layers.Concatenate()([masked_negative_encoding_4, vgg16_l2_negative_encoding_4])
    # out7 = layers.Concatenate()([masked_negative_encoding_5, vgg16_l2_negative_encoding_5])

    # Output embedding
    out = layers.Concatenate()([out1, out2, out3, out4, out5])
    # out = layers.Concatenate()([out1, out2, out3, out4, out5, out6, out7])

    # train model
    model = models.Model(inputs=[x1, x2, x3, x4, x5, y1],
                         outputs=[out, out])

    # test model
    eval_model = models.Model(inputs=[x1, y1], outputs=out1)

    return model, eval_model