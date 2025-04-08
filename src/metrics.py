import tensorflow as tf
import lpips
import torch

def MSE(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Returns a vector of Mean Squared Error of two (batch_size, height, width, channels) tensors
    Where each value in the vector is the MSE of the corresponding image in the batch
    """
    return tf.reduce_mean(tf.square(y_true - y_pred), axis=(1, 2, 3))

def MSEimages(y_true, y_pred):
    """
    Returns a vector of Mean Squared Error of two (batch_size, height, width, channels) tensors
    Where each value in the vector is an image of the MSE of the corresponding image in the batch
    """
    return tf.square(y_true - y_pred)

def PSNR(y_true, y_pred):
    """
    Returns a vector of Peak Signal to Noise Ratio of two (batch_size, height, width, channels) tensors
    Where each value in the vector is the PSNR of the corresponding image in the batch
    """
    return tf.image.psnr(y_true, y_pred, max_val=2.0)

def SSIM(y_true, y_pred):
    """
    Returns a vector of Structural Similarity Index of two (batch_size, height, width, channels) tensors
    Where each value in the vector is the SSIM of the corresponding image in the batch
    """
    return tf.image.ssim(y_true, y_pred, max_val=2.0)

def LPIPS(y_true, y_pred):
    """
    Returns a vector of Learned Perceptual Image Patch Similarity of two (batch_size, height, width, channels) tensors
    Where each value in the vector is the LPIPS of the corresponding image in the batch
    """
    lpips_model = lpips.LPIPS(net='alex')
    y_true = tf.image.resize(y_true, (256, 256), method='nearest')
    y_pred = tf.image.resize(y_pred, (256, 256), method='nearest')
    y_true = tf.image.convert_image_dtype(y_true, tf.uint8)
    y_pred = tf.image.convert_image_dtype(y_pred, tf.uint8)
    y_true = tf.image.convert_image_dtype(y_true, tf.float32)
    y_pred = tf.image.convert_image_dtype(y_pred, tf.float32)
    y_true = tf.transpose(y_true, perm=[0, 3, 1, 2])
    y_pred = tf.transpose(y_pred, perm=[0, 3, 1, 2])
    y_true = torch.tensor(y_true.numpy())
    y_pred = torch.tensor(y_pred.numpy())
    return lpips_model.forward(y_true, y_pred).detach().numpy()