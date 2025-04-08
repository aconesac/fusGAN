import tensorflow as tf
import os
from models import GAN
from src.dataset import MatDataset
from src.config import *
from .AdvancedCheckpointConverter import convert_gan_checkpoints

# Configuration
CHECKPOINT_GEN_PATH = 'checkpoints/generator'    # Path to generator checkpoint
CHECKPOINT_DISC_PATH = 'checkpoints/discriminator'  # Path to discriminator checkpoint
KERAS_MODELS_DIR = 'models'   # Directory to save .keras models
TRAIN_DATA_DIR = 'data/train'  # Directory containing training data
EPOCHS = 50  # Number of additional epochs for retraining

def main():
    # Step 1: Convert checkpoints to .keras format
    print("Converting checkpoints to .keras format...")
    generator, discriminator = convert_gan_checkpoints(
        CHECKPOINT_GEN_PATH, 
        CHECKPOINT_DISC_PATH, 
        KERAS_MODELS_DIR
    )
    
    # Step 2: Prepare dataset
    print("Preparing dataset...")
    train_filenames = [os.path.join(TRAIN_DATA_DIR, f) for f in os.listdir(TRAIN_DATA_DIR) 
                      if f.endswith('.mat')]
    train_dataset = MatDataset(train_filenames).dataset
    
    # Step 3: Set up optimizers
    generator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE, beta_1=0.5)
    
    # Step 4: Define loss function
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    # Step 5: Create GAN with loaded models
    gan = GAN(generator, discriminator, generator_optimizer, discriminator_optimizer, loss_object)
    
    # Step 6: Continue training
    print(f"Starting retraining for {EPOCHS} epochs...")
    gan.fit(train_dataset, EPOCHS)
    
    # Step 7: Save the retrained models
    retrained_gen_path = os.path.join(KERAS_MODELS_DIR, 'generator_retrained.keras')
    retrained_disc_path = os.path.join(KERAS_MODELS_DIR, 'discriminator_retrained.keras')
    
    gan.save_model(retrained_gen_path, retrained_disc_path)
    print(f"Retraining complete. Models saved to {retrained_gen_path} and {retrained_disc_path}")

if __name__ == "__main__":
    main()
