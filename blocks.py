from tensorflow.keras import layers, models
from tensorflow.keras.layers import Permute, Multiply, MultiHeadAttention
from utils import squash
from layers import CapsNet, VisionTransformer, PatchEncoder, Patches, \
    GroupNormalization
from tensorflow.keras import utils
from tensorflow.keras.backend import image_data_format, dot, l2_normalize, epsilon
import tensorflow as tf
from config import get_arguments

args = get_arguments()
l2_norm = layers.Lambda(lambda enc: l2_normalize(enc, axis=-1) + epsilon())

# spatial attention down sampleing
def dilated_spatial_down_conv(tensor, numclass):
    channel_axis = 1 if image_data_format() == "channels_first" else -1
    filters = tensor.shape[channel_axis]

    out = layers.Conv2D(filters, kernel_size=(3, 3), strides=(2, 2), padding='same')(tensor)
    out = layers.BatchNormalization()(out)
    # out = layers.ReLU()(out)
    out1 = out
    out = layers.SeparableConv2D(numclass, kernel_size=(3, 3), padding='same', dilation_rate=2)(out)
    out = layers.Softmax()(out)
    out = layers.Conv2D(1, kernel_size=(1, 1), padding='same')(out)
    out = layers.Activation('sigmoid')(out)

    out = layers.Multiply()([out1, out])

    return out

# spatial attention
def dilated_spatial_conv(tensor, numclass):
    channel_axis = 1 if image_data_format() == "channels_first" else -1
    filters = tensor.shape[channel_axis]


    out = layers.Conv2D(filters, kernel_size=(3, 3), padding='same')(tensor)
    out = layers.BatchNormalization()(out)
    out = layers.Activation('relu')(out)
    out = layers.Conv2D(numclass, kernel_size=(3, 3), dilation_rate=2, padding='same')(out)
    out = layers.BatchNormalization()(out)
    out = layers.Softmax()(out)
    out = layers.Conv2D(1, kernel_size=(3, 3), padding='same')(out)
    out = layers.BatchNormalization()(out)
    out = layers.Activation('sigmoid')(out)

    out = layers.Multiply()([tensor, out])

    return out

def vision_transformer_block(inputs, patch_size, input_size, projection_dim, transformer_layers, mlp_head_units):
    # patch_size = 16  # Size of the patches to be extract from the input images
    num_patches = (input_size // patch_size) ** 2
    # projection_dim = 128
    num_heads = 4
    transformer_units = [
        projection_dim * 2,
        projection_dim,
    ]  # Size of the transformer layers

    # transformer_layers = 4
    # mlp_head_units = [1024, 256]  # Size of the dense layers of the final classifier

    def mlp(x, hidden_units, dropout_rate):
        for units in hidden_units:
            # x = layers.Dense(units, activation=tf.nn.gelu)(x)
            x = layers.Dense(units, activation=tf.nn.relu)(x)
            x = layers.Dropout(dropout_rate)(x)
        return x

    # inputs = layers.Input(shape=input_shape)
    # Create patches.
    patches = Patches(patch_size)(inputs)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # x1 = GroupNormalization(groups=16, axis=1)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # x3 = GroupNormalization(groups=16, axis=1)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    # representation = GroupNormalization(groups=16, axis=1)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.3)(representation)

    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.3)
    # features = layers.Activation('relu')(features)
    # Classify outputs.
    # logits = layers.Dense(num_classes)(features)

    # Create the Keras model.
    # model = keras.Model(inputs=inputs, outputs=logits)
    return features


def class_down_attention(tensor, num_class):
    channel_axis = 1 if image_data_format() == "channels_first" else -1
    filters = tensor.shape[channel_axis]

    out_shape = (1, 1, num_class)

    out = layers.Conv2D(num_class, kernel_size=(1, 1), padding='same')(tensor)
    out = layers.BatchNormalization()(out)
    out = layers.Softmax()(out)
    out = layers.GlobalAveragePooling2D()(out)

    # out = layers.Conv2D(filters, kernel_size=(1, 1))(out)
    out = layers.Reshape(out_shape)(out)
    out = layers.Conv2D(filters, kernel_size=(1, 1), padding='same')(out)

    out = Multiply()([out, tensor])
    out = layers.Conv2D(filters, kernel_size=(3, 3), strides=(2, 2), padding='same')(out)

    return out

# Channel attention
def class_attention(tensor, num_class):
    channel_axis = 1 if image_data_format() == "channels_first" else -1
    filters = tensor.shape[channel_axis]

    # filters = Weight_Standardization(filters)
    # num_class = Weight_Standardization(num_class)

    out_shape = (1, 1, num_class)

    out = layers.Conv2D(num_class, kernel_size=(3, 3), padding='same')(
        tensor)
    out = layers.BatchNormalization()(out)
    # out = GroupNormalization(out)
    out = layers.Softmax()(out)
    out = layers.GlobalAveragePooling2D()(out)
    # l2_norm(out)
    # out = layers.Conv2D(filters, kernel_size=(1, 1))(out)
    out = layers.Reshape(out_shape)(out)
    out = layers.Conv2D(filters, kernel_size=(1, 1), padding='same')(out)
    out = layers.BatchNormalization()(out)
    # out = GroupNormalization(groups=16, axis=channel_axis)(out)

    out = Multiply()([out, tensor])

    return out

# AEM
def se_block(tensor, reduce_ratio=16):
    init = tensor
    channel_axis = 1 if image_data_format() == "channels_first" else -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = layers.GlobalAveragePooling2D()(init)
    se = layers.Reshape(se_shape)(se)
    se = layers.Dense(filters // reduce_ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(
        se)  # FC1
    se = layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)  # FC2

    if image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    return se


def multiply(init, se):
    x = Multiply()([init, se])  # Scale
    return x


def Weight_Standardization(kernel):
    kernel_mean = tf.math.reduce_mean(kernel, axis=[0, 1, 2], keepdims=True, name='kernel_mean')
    kernel = kernel - kernel_mean
    kernel_std = tf.keras.backend.std(kernel, axis=[0, 1, 2], keepdims=True)
    kernel = kernel / (kernel_std + 1e-5)


def conv_bn_block(inputs, filters, k_size, stride, padding, name):
    channel_axis = 1 if image_data_format() == "channels_first" else -1
    out = layers.Conv2D(filters=filters, kernel_size=k_size, strides=stride, padding=padding, name=name)(inputs)
    out = layers.BatchNormalization()(out)
    # out = GroupNormalization(groups=16, axis=channel_axis)(out)
    out = layers.ReLU()(out)
    return out



# Residual block with AEM
def residual_block(y, nb_channels, _strides=(2, 2), _project_shortcut=False):
    shortcut = y
    # group_size = 16
    # channel_axis = 1 if image_data_format() == "channels_first" else -1

    # down-sampling is performed with a stride of 2
    # y = layers.SeparableConv2D(nb_channels, (3, 3), strides=_strides, padding='same')(y)
    y = layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same', use_bias=False)(y)
    # y = GroupNormalization(groups=group_size, axis=channel_axis)(y)
    y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)

    se = se_block(y)
    y = multiply(y, se)
    # identity shortcuts used directly when the input and output are of the same dimensions
    if _project_shortcut or _strides != (1, 1):
        shortcut = layers.SeparableConv2D(nb_channels, kernel_size=(1, 1), strides=_strides, padding='same',
                                          use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    y = layers.add([shortcut, y])
    y = layers.LeakyReLU()(y)

    return y


def residual_attention_block(y, nb_channels, args, _strides=(2, 2)):

    # down-sampling is performed with a stride of 2
    y = layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)
    y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)

    # attention
    ca_out = class_attention(y, args.num_class)
    sa_out = dilated_spatial_conv(y, args.num_class)

    # Add
    y = layers.Add()([ca_out, sa_out])

    return y


def primary_capsule(inputs, dim_capsule, name):
    inputs = layers.Reshape(target_shape=[-1, dim_capsule], name=name + '_reshape')(inputs)
    return layers.Lambda(squash, name=name + '_squash')(inputs)


def capsule_model(inputs, args):

    out = conv_bn_block(inputs, filters=64, k_size=3, stride=1, padding="same", name="conv_block_1")

    # Version 1
    # out = residual_block(out, nb_channels=64, _project_shortcut=True)
    # out_64_down = dilated_spatial_down_conv(out, args.num_class)
    # out = residual_block(out, nb_channels=64, _project_shortcut=True)
    # out_64 = dilated_spatial_conv(out, args.num_class)
    # out = layers.Add()([out_64_down, out_64])
    # # carafe_out = carafe(out, 64, 2, 3, 3)
    # # out = layers.SpatialDropout2D(rate=0.2)(out)
    # out = class_attention(out, args.num_class)
    #
    # out = residual_block(out, nb_channels=128, _project_shortcut=True)
    # out_128_down = dilated_spatial_down_conv(out, args.num_class)
    # out = residual_block(out, nb_channels=128, _project_shortcut=True)
    # out_128 = dilated_spatial_conv(out, args.num_class)
    # out = layers.Add()([out_128_down, out_128])
    # out = class_attention(out, args.num_class)
    #
    # out = residual_block(out, nb_channels=256, _project_shortcut=True)
    # out_256_down = dilated_spatial_down_conv(out, args.num_class)
    # out = residual_block(out, nb_channels=256, _project_shortcut=True)
    # out_256 = dilated_spatial_conv(out, args.num_class)
    # out = layers.Add()([out_256_down, out_256])
    # out = layers.SpatialDropout2D(rate=0.2)(out)
    # out = class_attention(out, args.num_class)

    # Version 2
    # out = residual_block(out, nb_channels=64, _project_shortcut=True)
    # out_64_down = dilated_spatial_down_conv(out, args.num_class)
    # out = residual_block(out, nb_channels=64, _project_shortcut=True)
    # out_64 = dilated_spatial_conv(out, args.num_class)
    # out = layers.Add()([out_64_down, out_64])
    # out = class_attention(out, args.num_class)
    #
    # out = residual_block(out, nb_channels=128, _project_shortcut=True)
    # out_128_down = dilated_spatial_down_conv(out, args.num_class)
    # out = residual_block(out, nb_channels=128, _project_shortcut=True)
    # out_128 = dilated_spatial_conv(out, args.num_class)
    # out = layers.Add()([out_128_down, out_128])
    # out = class_attention(out, args.num_class)
    #
    # out = residual_block(out, nb_channels=256, _project_shortcut=True)
    # out_256_down = dilated_spatial_down_conv(out, args.num_class)
    # out = residual_block(out, nb_channels=256, _project_shortcut=True)
    # out = layers.Add()([out_256_down, out])
    # out = class_attention(out, args.num_class)

    # ViT Attention Version 3
    # out = conv_bn_block(out, filters=64, k_size=3, stride=2, padding="same", name="conv_block_2")
    out = residual_block(out, nb_channels=64, _project_shortcut=True)
    # out = residual_attention_block(out, nb_channels=128, _project_shortcut=True)
    out1 = class_attention(out, args.num_class)
    out1 = dilated_spatial_conv(out1, args.num_class)
    v3_out1 = vision_transformer_block(out, patch_size=16, input_size=128, projection_dim=128, transformer_layers=6,
                                       mlp_head_units=[512, 256])
    v3_out2 = vision_transformer_block(out1, patch_size=8, input_size=128, projection_dim=64, transformer_layers=4,
                                       mlp_head_units=[512, 256])
    out = layers.Add()([v3_out1, v3_out2])
    cap_out1 = primary_capsule(out, dim_capsule=16, name="primarycaps", args=args)
    out = CapsNet(num_capsule=args.num_class, dim_capsule=args.dim_capsule, routings=3, name="CapsNet")(
        cap_out1)

    # Version 2 + multi-capsule
    # out = residual_block(out, nb_channels=64, _project_shortcut=True)
    # out_64_sa = dilated_spatial_conv(out, args.num_class)
    # out_64_ca = class_attention(out, args.num_class)
    # out = layers.Multiply()([out_64_sa, out_64_ca])
    #
    # # caps out
    # caps_out1 = primary_capsule(out, dim_capsule=16, name="primarycaps_1", args=args)
    # caps_out1 = CapsNet(num_capsule=args.num_class, dim_capsule=args.dim_capsule, routings=3,
    #                         name="CapsNet_1")(caps_out1)
    #
    # out = residual_block(out, nb_channels=128, _project_shortcut=True)
    # out_128_sa = dilated_spatial_conv(out, args.num_class)
    # out_128_ca = class_attention(out, args.num_class)
    # out = layers.Multiply()([out_128_sa, out_128_ca])
    #
    # caps_out2 = primary_capsule(out, dim_capsule=16, name="primarycaps_2", args=args)
    # caps_out2 = CapsNet(num_capsule=args.num_class, dim_capsule=args.dim_capsule, routings=3,
    #                         name="CapsNet_2")(caps_out2)
    #
    # out1 = layers.Add()([caps_out1, caps_out2])
    #
    # out = residual_block(out, nb_channels=256, _project_shortcut=True)
    # out_256_sa = dilated_spatial_conv(out, args.num_class)
    # out_256_ca = class_attention(out, args.num_class)
    # out = layers.Multiply()([out_256_sa, out_256_ca])
    #
    # caps_out3 = primary_capsule(out, dim_capsule=16, name="primarycaps_3", args=args)
    # caps_out3 = CapsNet(num_capsule=args.num_class, dim_capsule=args.dim_capsule, routings=3,
    #                         name="CapsNet_3")(caps_out3)
    #
    # out = layers.Add()([out1, caps_out3])

    # out = primary_capsule(out, dim_capsule=16, name="primarycaps", args=args)
    # out = CapsNet(num_capsule=args.num_class, dim_capsule=args.dim_capsule, routings=3, name="CapsNet")(
    #     out)

    return out

# group normalization
def vgg_16_pretrained(args, weights_path=True):
    TF_WEIGHTS_PATH_NO_TOP = (
        'https://github.com/fchollet/deep-learning-models/'
        'releases/download/v0.1/'
        'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

    img_input = layers.Input(shape=(256, 256, 3))

    channel_axis = 1 if image_data_format() == "channels_first" else -1
    group_size = 16

    # Block 1
    x = layers.Conv2D(64, (3, 3),
                      padding='same',
                      name='block1_conv1',
                      # kernel_initializer='he_nornal',
                      kernel_regularizer=Weight_Standardization)(img_input)
    x = GroupNormalization(groups=group_size, axis=channel_axis)(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, (3, 3),
                      padding='same',
                      name='block1_conv2',
                      kernel_regularizer=Weight_Standardization)(x)
    x = GroupNormalization(groups=group_size, axis=channel_axis)(x)
    x = layers.Activation('relu')(x)

    s1_GAP = layers.GlobalAveragePooling2D()(x)
    s1 = l2_norm(s1_GAP)

    s1_residual = residual_attention_block(x, nb_channels=128, args=args)

    y = layers.Conv2D(64, (1, 1),
                      padding='same',
                      strides=(2, 2),
                      name='downsampling_1',
                      kernel_regularizer=Weight_Standardization)(x)
    x = GroupNormalization(groups=group_size, axis=channel_axis)(y)
    x = layers.Activation('relu')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3),
                      padding='same',
                      name='block2_conv1',
                      kernel_regularizer=Weight_Standardization)(x)
    x = GroupNormalization(groups=group_size, axis=channel_axis)(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, (3, 3),
                      padding='same',
                      name='block2_conv2',
                      kernel_regularizer=Weight_Standardization)(x)
    # x = layers.BatchNormalization()(x)
    x = GroupNormalization(groups=group_size, axis=channel_axis)(x)
    x = layers.Activation('relu')(x)

    s2_conv = layers.Add()([s1_residual, x])
    s2_GAP = layers.GlobalAveragePooling2D()(s2_conv)
    s2 = l2_norm(s2_GAP)

    s2_residual = residual_attention_block(x, nb_channels=256, args=args)

    y = layers.Conv2D(128, (1, 1),
                      padding='same',
                      strides=(2, 2),
                      name='downsampling_2',
                      kernel_regularizer=Weight_Standardization)(x)
    x = GroupNormalization(groups=group_size, axis=channel_axis)(y)
    x = layers.Activation('relu')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3),
                      padding='same',
                      name='block3_conv1',
                      kernel_regularizer=Weight_Standardization)(x)
    x = GroupNormalization(groups=group_size, axis=channel_axis)(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, (3, 3),
                      padding='same',
                      name='block3_conv2',
                      kernel_regularizer=Weight_Standardization)(x)
    x = GroupNormalization(groups=group_size, axis=channel_axis)(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, (3, 3),
                      padding='same',
                      name='block3_conv3',
                      kernel_regularizer=Weight_Standardization)(x)
    x = GroupNormalization(groups=group_size, axis=channel_axis)(x)
    x = layers.Activation('relu')(x)

    s3_conv = layers.Add()([s2_residual, x])
    s3_GAP = layers.GlobalAveragePooling2D()(s3_conv)
    s3 = l2_norm(s3_GAP)

    s3_residual = residual_attention_block(x, nb_channels=512, args=args)

    y = layers.Conv2D(256, (1, 1),
                      padding='same',
                      strides=(2, 2),
                      name='downsampling_3',
                      kernel_regularizer=Weight_Standardization)(x)
    x = GroupNormalization(groups=group_size, axis=channel_axis)(y)
    x = layers.Activation('relu')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3),
                      padding='same',
                      name='block4_conv1',
                      kernel_regularizer=Weight_Standardization)(x)
    x = GroupNormalization(groups=group_size, axis=channel_axis)(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, (3, 3),
                      padding='same',
                      name='block4_conv2',
                      kernel_regularizer=Weight_Standardization)(x)
    # x = layers.BatchNormalization()(x)
    x = GroupNormalization(groups=group_size, axis=channel_axis)(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, (3, 3),
                      padding='same',
                      name='block4_conv3',
                      kernel_regularizer=Weight_Standardization)(x)
    # x = layers.BatchNormalization()(x)
    x = GroupNormalization(groups=group_size, axis=channel_axis)(x)
    x = layers.Activation('relu')(x)

    s4_conv = layers.Add()([s3_residual, x])
    s4_GAP = layers.GlobalAveragePooling1D()(s4_conv)
    s4 = l2_norm(s4_GAP)

    # Block 5
    # x = layers.Conv2D(512, (3, 3),
    #                   padding='same',
    #                   name='block5_conv1')(y)
    # x = layers.BatchNormalization()(x)
    # x = layers.Activation('relu')(x)
    # x = layers.Conv2D(512, (3, 3),
    #                   padding='same',
    #                   name='block5_conv2')(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.Activation('relu')(x)
    # x = layers.Conv2D(512, (3, 3),
    #                   padding='same',
    #                   name='block5_conv3')(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.Activation('relu')(x)
    # s5 = class_attention(x, args.num_class)
    # s5 = layers.GlobalAveragePooling2D()(s5)
    # s5 = layers.GlobalAveragePooling2D()(x)
    # s5 = l2_norm(s5)
    # x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = layers.Concatenate()([s1, s2, s3, s4])

    # Create model.
    model = models.Model(img_input, x, name='vgg16')

    # pre-train weight
    if weights_path:
        weights_path = utils.get_file(
            'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
            TF_WEIGHTS_PATH_NO_TOP,
            cache_subdir='models',
            file_hash='6d6bbae143d832006294945121d1f1fc')
        model.load_weights(weights_path, by_name=True)

    return model

# Version 5 lower branch
def vgg_16_pretrained_tensorflow(args, weights_path=True):
    TF_WEIGHTS_PATH_NO_TOP = (
        'https://github.com/fchollet/deep-learning-models/'
        'releases/download/v0.1/'
        'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

    img_input = layers.Input(shape=(256, 256, 3))

    # Block 1
    x = layers.Conv2D(64, (3, 3),
                      padding='same',
                      name='block1_conv1')(img_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, (3, 3),
                      padding='same',
                      name='block1_conv2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    s1_GAP = layers.GlobalAveragePooling2D()(x)
    s1 = l2_norm(s1_GAP)

    s1_residual = residual_attention_block(x, nb_channels=128, args=args)

    # s1_conv = layers.Conv2D(128, (3, 3),
    #                         padding='same',
    #                         strides=(2, 2),
    #                         name='r_downsampling_1')(x)
    #
    #
    # s1_conv_b = layers.BatchNormalization()(s1_conv)
    # s1_conv_b = layers.Activation('relu')(s1_conv_b)
    # s1_conv_c = class_attention(s1_conv_b, args.num_class)
    # s1_conv_d = dilated_spatial_conv(s1_conv_b, args.num_class)
    # s1_conv = layers.Add()([s1_conv_c, s1_conv_d])

    # downsampling
    y = layers.Conv2D(64, (3, 3),
                      padding='same',
                      strides=(2, 2),
                      name='downsampling_1')(x)
    x = layers.BatchNormalization()(y)
    x = layers.Activation('relu')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3),
                      padding='same',
                      name='block2_conv1')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, (3, 3),
                      padding='same',
                      name='block2_conv2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    s2_out = layers.Add()([s1_residual, x])
    s2_GAP = layers.GlobalAveragePooling2D()(s2_out)
    s2 = l2_norm(s2_GAP)

    s2_residual = residual_attention_block(x, nb_channels=256, args=args)

    # downsampling
    y = layers.Conv2D(128, (3, 3),
                      padding='same',
                      strides=(2, 2),
                      name='downsampling_2')(x)
    x = layers.BatchNormalization()(y)
    x = layers.Activation('relu')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3),
                      padding='same',
                      name='block3_conv1')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, (3, 3),
                      padding='same',
                      name='block3_conv2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, (3, 3),
                      padding='same',
                      name='block3_conv3')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    s3_out = layers.Add()([s2_residual, x])
    s3_GAP = layers.GlobalAveragePooling2D()(s3_out)
    s3 = l2_norm(s3_GAP)

    s3_residual = residual_attention_block(x, nb_channels=512, args=args)

    # downsampling
    y = layers.Conv2D(256, (3, 3),
                      padding='same',
                      strides=(2, 2),
                      name='downsampling_3')(x)
    x = layers.BatchNormalization()(y)
    x = layers.Activation('relu')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3),
                      padding='same',
                      name='block4_conv1')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, (3, 3),
                      padding='same',
                      name='block4_conv2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(512, (3, 3),
                      padding='same',
                      name='block4_conv3')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    s4_out = layers.Add()([s3_residual, x])
    s4_GAP = layers.GlobalAveragePooling2D()(s4_out)
    s4 = l2_norm(s4_GAP)

    # Output embedding
    x = layers.Concatenate()([s1, s2, s3, s4])

    # Create model.
    model = models.Model(img_input, x, name='vgg16')

    # load weight
    if weights_path:
        weights_path = utils.get_file(
            'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
            TF_WEIGHTS_PATH_NO_TOP,
            cache_subdir='models',
            file_hash='6d6bbae143d832006294945121d1f1fc')
        model.load_weights(weights_path, by_name=True)

    return model
