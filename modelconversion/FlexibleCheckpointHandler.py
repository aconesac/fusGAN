import tensorflow as tf
import os
import glob
import sys

# add the path to the src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import Generator, Discriminator, GAN
from src.dataset import MatDataset

def find_latest_checkpoint(checkpoint_dir):
    """
    Find the latest checkpoint in a directory
    
    Args:
        checkpoint_dir (str): Directory containing checkpoint files
        
    Returns:
        str: Prefix of the latest checkpoint
    """
    # Try to read the checkpoint file
    checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint')
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                first_line = f.readline()
                # Extract checkpoint name from line like: model_checkpoint_path: "ckpt-6"
                if 'model_checkpoint_path' in first_line:
                    parts = first_line.split('"')
                    if len(parts) >= 2:
                        return parts[1]
        except:
            pass
        
    # If reading the checkpoint file failed, fall back to globbing
    print(f"Failed to read checkpoint file: {checkpoint_file}. Attempting to find checkpoints using glob.")
    # Check if the checkpoint file exists
    if not os.path.exists(checkpoint_file):
        print(f"Checkpoint file not found: {checkpoint_file}")
    
    # If checkpoint file parsing failed, find the highest numbered checkpoint
    checkpoints = glob.glob(os.path.join(checkpoint_dir, '*.index'))
    print(checkpoints)
    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
    
    # Extract checkpoint numbers and find the highest
    checkpoint_numbers = []
    for checkpoint in checkpoints:
        base = os.path.basename(checkpoint)
        # Handle patterns like 'ckpt-5.index' or 'model-1000.index'
        parts = base.split('-')
        if len(parts) >= 2:
            try:
                num = int(parts[1].split('.')[0])
                checkpoint_numbers.append((num, parts[0] + '-' + parts[1].split('.')[0]))
            except ValueError:
                continue
    
    if not checkpoint_numbers:
        return None
    
    # Return the highest numbered checkpoint prefix
    return sorted(checkpoint_numbers, reverse=True)[0][1]

def convertCheckpoints2Keras(checkpoint_dir, save_dir):
    """
    Identify checkpoint type, convert to .keras, and retrain
    
    Args:
        checkpoint_dir (str): Directory containing checkpoint files
        save_dir (str): Directory to save .keras models
        train_data_dir (str): Directory containing training data
        epochs (int): Number of epochs for retraining
        
    Returns:
        GAN: Trained GAN model
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Step 1: Find latest checkpoint
    checkpoint_prefix = find_latest_checkpoint(checkpoint_dir)
    if not checkpoint_prefix:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    
    print(f"Found latest checkpoint: {checkpoint_prefix}")
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_prefix)
    
    # Step 2: Initialize models
    generator = Generator(2, 1).model
    discriminator = Discriminator(2).model
    
    # Step 3: Create optimizers
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    
    # Step 4: Try different checkpoint structures
    # First try: Combined checkpoint (with both generator and discriminator)
    try:
        print("Attempting to load as combined checkpoint...")
        checkpoint = tf.train.Checkpoint(
            generator=generator,
            discriminator=discriminator,
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer
        )
        status = checkpoint.restore(checkpoint_path)
        status.expect_partial()
        print("Successfully loaded combined checkpoint!")
        checkpoint_type = "combined"
    except:
        # Second try: Separate checkpoints for each model
        try:
            print("Attempting to load as separate generator checkpoint...")
            gen_checkpoint = tf.train.Checkpoint(model=generator)
            gen_status = gen_checkpoint.restore(checkpoint_path)
            gen_status.expect_partial()
            
            print("Attempting to load as separate discriminator checkpoint...")
            disc_checkpoint = tf.train.Checkpoint(model=discriminator)
            disc_checkpoint_path = checkpoint_path.replace("generator", "discriminator")
            disc_status = disc_checkpoint.restore(disc_checkpoint_path)
            disc_status.expect_partial()
            
            print("Successfully loaded separate checkpoints!")
            checkpoint_type = "separate"
        except:
            raise ValueError(f"Failed to load checkpoint from {checkpoint_path}")
    
    # Step 5: Save as .keras format
    generator_path = os.path.join(save_dir, 'generator.keras')
    discriminator_path = os.path.join(save_dir, 'discriminator.keras')
    
    generator.save(generator_path)
    discriminator.save(discriminator_path)
    
    print(f"Models saved to .keras format in {save_dir}")

# Example usage
if __name__ == "__main__":
    # Default paths - adjust these to match your setup
    checkpoint_dir = 'SavedModels/checkpoints_baseline_bigdata_dataaug2_low_disc_lr'
    save_dir = 'SavedModels'
    train_data_dir = 'data/train'
    
    # Convert and retrain
    gan = convertCheckpoints2Keras(
        checkpoint_dir, 
        save_dir
    )