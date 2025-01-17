import tensorflow as tf
from models import GAN, Generator, Discriminator
from src.dataset import MatDataset
from src.parser import parse_arguments
import os

# args = parse_arguments()
# paths = [args.ct_data_path, args.mask_data_path, args.output_path]

train_filenames = [os.path.join('data/train', f) for f in os.listdir('data/train') if f.endswith('.mat')]

train_dataset = MatDataset(train_filenames).dataset

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

generator = Generator(2,1).model
discriminator = Discriminator(2).model

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

gan = GAN(generator, discriminator, generator_optimizer, discriminator_optimizer, loss_object)

gan.fit(train_dataset, 150)

gan.save_model('models/generator.keras', 'models/discriminator.keras')