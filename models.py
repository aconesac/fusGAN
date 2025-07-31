import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.config import *
from src.metrics import MSE, PSNR, SSIM, LPIPS
import time
import os

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
        self.generated_accuracy = tf.keras.metrics.Mean(name='generated_accuracy')
        self.real_accuracy = tf.keras.metrics.Mean(name='real_accuracy')

    def generator_loss_fn(self, disc_generated_output: tf.Tensor, gen_output, target: tf.Tensor):
        gan_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output) # Generator wants discriminator to think generated images are real
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output)) # L1 loss for image reconstruction
        total_gen_loss = gan_loss + (LAMBDA * l1_loss)
        return total_gen_loss, gan_loss, l1_loss

    def discriminator_loss_fn(self, disc_real_output: tf.Tensor, disc_generated_output: tf.Tensor):
        real_loss = self.loss_object(tf.ones_like(disc_real_output) * 0.9, disc_real_output)
        generated_loss = self.loss_object(tf.zeros_like(disc_generated_output) + 0.1, disc_generated_output)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss
    
    def discriminator_accuracy_fn(self, disc_real_output: tf.Tensor, disc_generated_output: tf.Tensor):
        real_predictions = tf.cast(disc_real_output > 0.5, tf.float32)
        generated_predictions = tf.cast(disc_generated_output < 0.5, tf.float32)
        real_accuracy = tf.reduce_mean(real_predictions)
        generated_accuracy = tf.reduce_mean(generated_predictions)
        return real_accuracy, generated_accuracy

    @tf.function
    def discriminator_train_step(self, input_image: tf.Tensor, target: tf.Tensor):
        """Train only the discriminator"""
        with tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=False)  # Don't train generator here
            
            disc_real_output = self.discriminator([input_image, target], training=True)
            disc_generated_output = self.discriminator([input_image, gen_output], training=True)
            
            disc_loss = self.discriminator_loss_fn(disc_real_output, disc_generated_output)
        
        discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_variables))
        
        real_accuracy, generated_accuracy = self.discriminator_accuracy_fn(disc_real_output, disc_generated_output)
        
        return disc_loss, real_accuracy, generated_accuracy, disc_real_output, disc_generated_output
    
    @tf.function
    def generator_train_step(self, input_image: tf.Tensor, target: tf.Tensor):
        """Train only the generator"""
        with tf.GradientTape() as gen_tape:
            gen_output = self.generator(input_image, training=True)
            
            disc_generated_output = self.discriminator([input_image, gen_output], training=False)  # Don't train discriminator here
            
            total_loss, gen_loss, l1_loss = self.generator_loss_fn(disc_generated_output, gen_output, target)
        
        generator_gradients = gen_tape.gradient(total_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        
        return gen_loss, l1_loss, total_loss
    
    @tf.function
    def train_step(self, input_image: tf.Tensor, target: tf.Tensor, train_discriminator: bool = True, train_generator: bool = True):
        """
        Combined training step with optional discriminator and generator training
        
        Args:
            input_image: Input tensor
            target: Target tensor  
            train_discriminator: Whether to update discriminator weights
            train_generator: Whether to update generator weights
        """
        gen_loss = tf.constant(0.0)
        disc_loss = tf.constant(0.0)  
        l1_loss = tf.constant(0.0)
        real_accuracy = tf.constant(0.0)
        generated_accuracy = tf.constant(0.0)
        
        # Train discriminator if requested
        if train_discriminator:
            disc_loss, real_accuracy, generated_accuracy, disc_real_output, disc_generated_output = self.discriminator_train_step(input_image, target)
            self.discriminator_loss(disc_loss)
            self.real_accuracy(real_accuracy)
            self.generated_accuracy(generated_accuracy)
        
        # Train generator if requested  
        if train_generator:
            gen_loss, l1_loss, total_loss = self.generator_train_step(input_image, target)
            self.generator_loss(gen_loss)
            self.l1_loss(l1_loss)
        
        return gen_loss, disc_loss, l1_loss, real_accuracy, generated_accuracy
    
    def fit(self, train_dataset, epochs: int, disc_steps: int = 1, gen_steps: int = 1):
        """
        Alternative training method with block-based alternating pattern
        
        Args:
            train_dataset: Training dataset
            epochs: Number of epochs to train
            disc_steps: Number of consecutive discriminator training steps
            gen_steps: Number of consecutive generator training steps
        """
        history = {'generator_loss': [], 'discriminator_loss': [], 'l1_loss': [], 
                  'real_acc': [], 'gen_acc': []}
        
        with tqdm(total=epochs) as pbar:
            for epoch in range(epochs):
                with tqdm(total=len(train_dataset), desc=f'Epoch {epoch+1}/{epochs}', position=0, leave=True) as pbar2:
                    batch_iter = iter(train_dataset)
                    
                    try:
                        while True:
                            # Train discriminator for disc_steps
                            for _ in range(disc_steps):
                                input_image, target = next(batch_iter)
                                gen_loss, disc_loss, l1_loss, real_accuracy, generated_accuracy = self.train_step(
                                    input_image, target, train_discriminator=True, train_generator=False)
                                
                                pbar2.set_postfix(
                                    gen_loss=self.generator_loss.result().numpy(),
                                    disc_loss=self.discriminator_loss.result().numpy(),
                                    l1_loss=self.l1_loss.result().numpy(),
                                    gen_acc=self.generated_accuracy.result().numpy(),
                                    real_acc=self.real_accuracy.result().numpy(),
                                    mode="D"
                                )
                                pbar2.update(1)
                            
                            # Train generator for gen_steps
                            for _ in range(gen_steps):
                                input_image, target = next(batch_iter)
                                gen_loss, disc_loss, l1_loss, real_accuracy, generated_accuracy = self.train_step(
                                    input_image, target, train_discriminator=False, train_generator=True)
                                
                                pbar2.set_postfix(
                                    gen_loss=self.generator_loss.result().numpy(),
                                    disc_loss=self.discriminator_loss.result().numpy(),
                                    l1_loss=self.l1_loss.result().numpy(),
                                    gen_acc=self.generated_accuracy.result().numpy(),
                                    real_acc=self.real_accuracy.result().numpy(),
                                    mode="G"
                                )
                                pbar2.update(1)
                                
                    except StopIteration:
                        # End of dataset
                        pass

                    # Update history with the latest losses and accuracies 
                    [history[key].append(loss) for key, loss in zip(history.keys(), [self.generator_loss.result(),self.discriminator_loss.result(), self.l1_loss.result(),self.real_accuracy.result(), self.generated_accuracy.result()])]
                    
                    self.generator_loss.reset_state()
                    self.discriminator_loss.reset_state()
                    self.l1_loss.reset_state()
                    self.real_accuracy.reset_state()
                    self.generated_accuracy.reset_state()
                    pbar.update(1)
                    
                self.generate_images(input_image, target, output_path="out")   
                    
        return history
                    
    def evaluate(self, test_dataset):
        start_time = time.time()
        with tqdm(total=len(test_dataset), desc='Evaluating', position=0, leave=True) as pbar:
            for input_batch, target_batch in test_dataset:
                generated_images = self.generator(input_batch, training=False)
                discriminator_pred = self.discriminator([input_batch, generated_images], training=False)
                discriminator_real = self.discriminator([input_batch, target_batch], training=False)
                pbar.update(1)
        end_time = time.time()
        print(f'Evaluation time: {end_time - start_time}')
        
        print("\nCalculating metrics...")
        mse = MSE(target_batch, generated_images)
        psnr = PSNR(target_batch, generated_images)
        ssim = SSIM(target_batch, generated_images)
        lpips = LPIPS(target_batch, generated_images)
        # lpips = tf.zeros((len(target_batch), 1))
                
        return mse, psnr, ssim, lpips, end_time - start_time
    
    def predict(self, input_batch):
        return self.generator(input_batch, training=False)
                
    def generate_images(self, test_input: tf.Tensor, tar: tf.Tensor, output_path: str = None, index: int = None, metrics: list = None):
        """
        Generates and saves images from the test input and target images.
        """
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
        if index is not None:
            output_path = os.path.join(output_path, f'generated_image_{index}.png')
        else:
            output_path = 'generated_image.png'
        
        # Getting the prediction from the generator
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
        plt.savefig(output_path)
        plt.close()
        
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
        
class SmallDiscriminator(Model):
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
        
        down1 = self.downsample(32, 4, False)(x)  # (batch_size, 64, 64, 32)
        down2 = self.downsample(64, 4)(down1)  # (batch_size, 32, 32, 64)
        down3 = self.downsample(128, 4)(down2)  # (batch_size, 16, 16, 128)
        down4 = self.downsample(128, 4)(down3)  # (batch_size, 8, 8, 128)
        
        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down4)  # (batch_size, 10, 10, 128)
        conv = tf.keras.layers.Conv2D(128, 4, strides=1,
                                      kernel_initializer=initializer,
                                      use_bias=False)(zero_pad1)  # (batch_size, 7, 7, 128) 
        
        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
        
        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
        
        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 9, 9, 128)

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
            
            
            