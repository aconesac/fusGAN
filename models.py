import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.config import *
from src.metrics import MSE, PSNR, SSIM, LPIPS
import time

class GAN():
    def __init__(self, generator: tf.keras.Model = None, discriminator: tf.keras.Model = None, 
                 generator_optimizer: tf.keras.optimizers.Optimizer = None, 
                 discriminator_optimizer: tf.keras.optimizers.Optimizer = None, 
                 loss_object: tf.keras.losses.Loss = None) -> None:
        
        self.generator = generator
        self.discriminator = discriminator
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.loss_object = loss_object
        self.generator_loss = tf.keras.metrics.Mean(name='generator_loss')
        self.l1_loss = tf.keras.metrics.Mean(name='l1_loss')
        self.discriminator_loss = tf.keras.metrics.Mean(name='discriminator_loss')

    def generator_loss_fn(self, disc_generated_output: tf.Tensor, gen_output, target: tf.Tensor):
        gan_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        total_gen_loss = gan_loss + (LAMBDA * l1_loss)
        return total_gen_loss, gan_loss, l1_loss

    def discriminator_loss_fn(self, disc_real_output: tf.Tensor, disc_generated_output: tf.Tensor):
        real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = self.loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss

    @tf.function
    def train_step(self, input_image: tf.Tensor, target: tf.Tensor):
        # print(target.shape, input_image.shape)
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)

            disc_real_output = self.discriminator([input_image, target], training=True)
            disc_generated_output = self.discriminator([input_image, gen_output], training=True)

            total_loss, gen_loss, l1_loss  = self.generator_loss_fn(disc_generated_output, gen_output, target)
            disc_loss = self.discriminator_loss_fn(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_variables))

        self.generator_loss(gen_loss)
        self.discriminator_loss(disc_loss)
        self.l1_loss(l1_loss)
        
    def fit(self, train_dataset, epochs: int):
        history = {'generator_loss': [], 'discriminator_loss': [], 'l1_loss': []}
        with tqdm(total=epochs) as pbar:
            for epoch in range(epochs):
                with tqdm(total=len(train_dataset), desc=f'Epoch {epoch+1}/{epochs}', position=0, leave=True) as pbar2:
                    for input_image, target in train_dataset:
                        losses = self.train_step(input_image, target)
                        [history[key].append(loss) for key, loss in zip(history.keys(), [self.generator_loss.result().numpy(), self.discriminator_loss.result().numpy(), self.l1_loss.result().numpy()])]
                        pbar2.set_postfix(generator_loss=self.generator_loss.result().numpy(), discriminator_loss=self.discriminator_loss.result().numpy(), l1_loss=self.l1_loss.result().numpy())
                        pbar2.update(1)
                        
                    # print(f'Epoch {epoch+1}, Generator Loss: {self.generator_loss.result()}, Discriminator Loss: {self.discriminator_loss.result()}')
                    self.generator_loss.reset_state()
                    self.discriminator_loss.reset_state()
                    pbar.update(1)
                    
                self.generate_images(input_image, target)   
                    
        return history
                    
    def evaluate(self, test_dataset):
        start_time = time.time()
        for input_batch, target_batch in test_dataset:
            generated_images = self.generator(input_batch, training=False)
            discriminator_pred = self.discriminator([input_batch, generated_images], training=False)
            discriminator_real = self.discriminator([input_batch, target_batch], training=False)
            break
        end_time = time.time()
        print(f'Evaluation time: {end_time - start_time}')
        
        mse = MSE(target_batch, generated_images)
        psnr = PSNR(target_batch, generated_images)
        ssim = SSIM(target_batch, generated_images)
        lpips = LPIPS(target_batch, generated_images)
        
        return mse, psnr, ssim, lpips, end_time - start_time
    
    def predict(self, input_batch):
        return self.generator(input_batch, training=False)
                
    def generate_images(self, test_input: tf.Tensor, tar: tf.Tensor):
        prediction = self.generator(test_input, training=True)
        # plt.figure(figsize=(15,15))

        display_list = [tf.math.reduce_sum(test_input[0], axis=-1), tar[0], prediction[0]]
        title = ['Input Image', 'Ground Truth', 'Predicted Image']

        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.title(title[i])
            # Getting the pixel values between [0, 1] to plot it.
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')
        plt.savefig('out/output.png')
        
    def save_model(self, generator_path: str, discriminator_path: str):
        self.generator.save(generator_path)
        if discriminator_path is not None:
            self.discriminator.save(discriminator_path)
        
    def load_model(self, generator_path: str, discriminator_path: str):
        self.generator = tf.keras.models.load_model(generator_path)
        self.discriminator = tf.keras.models.load_model(discriminator_path)
        
    def generate_image(self, test_input: tf.Tensor) -> tf.Tensor:
        return self.generator(test_input, training=False)
    
class Model():
    def __init__(self) -> None:
        pass
    
    def downsample(self, filters, size: int, apply_batchnorm = True):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                    kernel_initializer=initializer, use_bias=False))

        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())

        result.add(tf.keras.layers.LeakyReLU())

        return result

    def upsample(self, filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False))

        result.add(tf.keras.layers.BatchNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))

        result.add(tf.keras.layers.ReLU())

        return result
    
    def build_model(self):
        pass
    
    def save_model(self, path: str):
        pass
    
class Discriminator(Model):
    def __init__(self, input_channels: int) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.model = self.build_model()
        self.model_num_params = self.model.count_params()

    def build_model(self) -> tf.keras.Model:
        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=[128, 128, self.input_channels], name='input_image')
        tar = tf.keras.layers.Input(shape=[128, 128, 1], name='target_image')

        x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 128, 128, input_channels + 1)

        down1 = self.downsample(64, 4, False)(x)  # (batch_size, 64, 64, 64)
        down2 = self.downsample(128, 4)(down1)  # (batch_size, 32, 32, 128)
        down3 = self.downsample(256, 4)(down2)  # (batch_size, 16, 16, 256)
        down4 = self.downsample(256, 4)(down3)  # (batch_size, 8, 8, 512)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down4)  # (batch_size, 10, 10, 512)
        conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                      kernel_initializer=initializer,
                                      use_bias=False)(zero_pad1)  # (batch_size, 7, 7, 512)

        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 9, 9, 512)

        last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                      kernel_initializer=initializer)(zero_pad2)  # (batch_size, 6, 6, 1)

        return tf.keras.Model(inputs=[inp, tar], outputs=last)

    def save_model(self, path: str) -> None:
        self.model.save(path)

class Generator(Model):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.model = self.build_model()
        self.model_num_params = self.model.count_params()

    def build_model(self) -> tf.keras.Model:
        inputs = tf.keras.layers.Input(shape=[128, 128, self.input_channels])

        down_stack = [
            self.downsample(128, 4, apply_batchnorm=False),  # (batch_size, 64, 64, 128)
            self.downsample(256, 4),  # (batch_size, 32, 32, 256)
            self.downsample(512, 4),  # (batch_size, 16, 16, 512)
            self.downsample(512, 4),  # (batch_size, 8, 8, 512)
            self.downsample(512, 4),  # (batch_size, 4, 4, 512)
        ]

        up_stack = [
            self.upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
            self.upsample(512, 4, apply_dropout=True),  # (batch_size, 16, 16, 1024)
            self.upsample(256, 4),  # (batch_size, 32, 32, 512)
            self.upsample(128, 4),  # (batch_size, 64, 64, 256)
            self.upsample(64, 4),   # (batch_size, 128, 128, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(self.output_channels, 4,
                                             strides=2,
                                             padding='same',
                                             kernel_initializer=initializer,
                                             activation='tanh',dtype=tf.float32)  # (batch_size, 64, 64, 3)
        x = inputs

        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)

    def save_model(self, path: str) -> None:
        self.model.save(path)
        
class BigGenerator(Model):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.model = self.build_model()
        self.model_num_params = self.model.count_params()

    def build_model(self) -> tf.keras.Model:
        inputs = tf.keras.layers.Input(shape=[128, 128, self.input_channels])

        down_stack = [
            self.downsample(128, 4, apply_batchnorm=False),  # (batch_size, 64, 64, 128)
            self.downsample(256, 4),  # (batch_size, 32, 32, 256)
            self.downsample(512, 4),  # (batch_size, 16, 16, 512)
            self.downsample(1024, 4),  # (batch_size, 8, 8, 1024)
            self.downsample(2048, 4),  # (batch_size, 4, 4, 1024)
        ]
        
        up_stack = [
            self.upsample(1024, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
            self.upsample(512, 4, apply_dropout=True),  # (batch_size, 16, 16, 1024)
            self.upsample(256, 4),  # (batch_size, 32, 32, 512)
            self.upsample(128, 4),  # (batch_size, 64, 64, 256)
            self.upsample(64, 4),   # (batch_size, 128, 128, 128)
        ]
        
        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(self.output_channels, 4,
                                             strides=2,
                                             padding='same',
                                             kernel_initializer=initializer,
                                             activation='tanh',dtype=tf.float32)
        x = inputs
        
        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)
            
        skips = reversed(skips[:-1])
        
        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])
            
        x = last(x)
        
        return tf.keras.Model(inputs=inputs, outputs=x)
    
    def save_model(self, path: str) -> None:
        self.model.save(path)
        
class HugeGenerator(Model):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.model = self.build_model()
        self.model_num_params = self.model.count_params()

    def build_model(self) -> tf.keras.Model:
        inputs = tf.keras.layers.Input(shape=[128, 128, self.input_channels])

        down_stack = [
            self.downsample(128, 2, apply_batchnorm=False),  # (batch_size, 64, 64, 128)
            self.downsample(256, 2),  # (batch_size, 32, 32, 256)
            self.downsample(512, 2),  # (batch_size, 16, 16, 512)
            self.downsample(1024, 2),  # (batch_size, 8, 8, 1024)
            self.downsample(2048, 2),  # (batch_size, 4, 4, 1024)
            self.downsample(4096, 2),  # (batch_size, 2, 2, 1024)
            self.downsample(8192, 2),  # (batch_size, 1, 1, 1024)
        ]
        
        up_stack = [
            self.upsample(4096, 2, apply_dropout=True),  # (batch_size, 8, 8, 1024)
            self.upsample(2048, 2, apply_dropout=True),  # (batch_size, 16, 16, 1024)
            self.upsample(1024, 2),  # (batch_size, 32, 32, 512)
            self.upsample(512, 2),  # (batch_size, 64, 64, 256)
            self.upsample(256, 2),   # (batch_size, 128, 128, 128)
        ]
        
        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(self.output_channels, 2,
                                             strides=2,
                                             padding='same',
                                             kernel_initializer=initializer,
                                             activation='tanh',dtype=tf.float32)
        x = inputs
        
        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)
            
        skips = reversed(skips[:-1])
        
        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])
            
        x = last(x)
        
        return tf.keras.Model(inputs=inputs, outputs=x)
            
            
            