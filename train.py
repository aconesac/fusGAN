import tensorflow as tf
from models import GAN, Generator, Discriminator
from src.dataset import MatDataset
from src.parser import parse_arguments
import os

# Parse command line arguments
# args = parse_arguments()
# paths = [args.ct_data_path, args.mask_data_path, args.output_path]

# Load data
train_filenames = [os.path.join('data/train', f) for f in os.listdir('data/train') if f.endswith('.mat')]
train_dataset = MatDataset(train_filenames).dataset

# Define optimizers
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# Check if models exist for retraining
if os.path.exists('SavedModels/generator.keras') and os.path.exists('SavedModels/discriminator.keras'):
    print("Loading existing models for retraining...")
    generator = tf.keras.models.load_model('SavedModels/generator.keras')
    discriminator = tf.keras.models.load_model('SavedModels/discriminator.keras')
else:
    print("Creating new models...")
    generator = Generator(2, 1).model
    discriminator = Discriminator(2).model

# Define loss function
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Initialize GAN
gan = GAN(generator, discriminator, generator_optimizer, discriminator_optimizer, loss_object)

# Train the model
epochs = 150
gan.fit(train_dataset, epochs)

# Save the models
gan.save_model('SavedModels/generatorRetrain.keras', 'SavedModels/discriminatorRetrain.keras')