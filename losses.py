from keras import backend as K
from keras.models import Model
from keras.applications.vgg16 import VGG16
import numpy as np
from keras.layers.merge import _Merge


# Note the image_shape must be multiple of patch_shape
image_shape = (256, 256, 3)
BATCH_SIZE = 16

def perceptual_loss(y_true, y_pred):
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=image_shape)
    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    loss_model.trainable = False
    return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))


def wasserstein_loss(y_true, y_pred):
    #Calculates the Wasserstein loss for a sample batch.
    return K.mean(y_true * y_pred)

# 计算一批“平均”样本的梯度惩罚损失。
def gradient_penalty_loss(y_true, y_pred, averaged_samples,
                          gradient_penalty_weight):
    
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)
    
    # 在改进的WGAN中，1 - Lipschitz约束通过向损失函数添加一个术语来强制执行，如果梯度范数远离1，则该惩罚网络。
    # 但是，无法在输入空间的所有点评估此函数。
    # 本文中使用的折衷方案是在实际样本和生成样本之间的线上选择随机点，并检查这些点的梯度。
    # 请注意，它是渐变w.r.t.输入平均样本，而不是鉴别器的权重，我们正在惩罚！
    #
    # 为了评估梯度，我们必须首先通生成器得到输出并评估损失。然后我们得到判别器的梯度w.r.t.输入平均样本。
    # 然后可以针对该梯度计算l2范数和罚分。
    #
    # 请注意，此损失失函数需要原始的平均样本作为输入，但Keras仅支持将y_true和y_pred传递给loss函数。
    # 为了解决这个问题，我们使用averaged_samples参数创建函数的partial（），并将其用于模型训练。“”
    # ＃首先得到渐变：
    # ＃assume： -  y_pred有维度（batch_size，1）
    # ＃ -  averaged_samples有维度（batch_size，nbr_features）
    # # graveients后面有维度（batch_size，nbr_features），基本上
    # # nbr_features维度梯度向量列表

    

class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this
    outputs a random point on the line between each pair of input points.

    Inheriting from _Merge is a little messy but it was the quickest solution I could
    think of. Improvements appreciated."""

    def _merge_function(self,inputs):
        weights = K.random_uniform((BATCH_SIZE, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])