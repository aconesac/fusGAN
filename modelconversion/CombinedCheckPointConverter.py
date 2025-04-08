import tensorflow as tf
import os
from models import Generator, Discriminator, GAN

def convert_combined_checkpoint(checkpoint_dir, checkpoint_prefix, save_dir, 
                               gen_input_channels=2, gen_output_channels=1, disc_input_channels=2):
    """
    Convert a combined checkpoint containing both generator and discriminator to .keras format
    
    Args:
        checkpoint_dir (str): Directory containing the checkpoint files
        checkpoint_prefix (str): Prefix of the checkpoint files (e.g., 'ckpt-6')
        save_dir (str): Directory to save the .keras format models
        gen_input_channels (int): Number of input channels for generator
        gen_output_channels (int): Number of output channels for generator
        disc_input_channels (int): Number of input channels for discriminator
        
    Returns:
        tuple: (generator, discriminator) loaded models
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize models with the same architecture as during training
    generator = Generator(gen_input_channels, gen_output_channels).model
    discriminator = Discriminator(disc_input_channels).model
    
    # Create optimizers (needed for checkpoint structure)
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    
    # Create a checkpoint that combines both models (matching the structure of your saved checkpoint)
    checkpoint = tf.train.Checkpoint(
        generator=generator,
        discriminator=discriminator,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer
    )
    
    # Full path to checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_prefix)
    
    # Restore from checkpoint
    status = checkpoint.restore(checkpoint_path)
    
    # Use expect_partial() to suppress warnings about optimizer variables
    status.expect_partial()
    
    print(f"Checkpoint restored from {checkpoint_path}")
    
    # Save models in .keras format
    generator_path = os.path.join(save_dir, 'generator.keras')
    discriminator_path = os.path.join(save_dir, 'discriminator.keras')
    
    generator.save(generator_path)
    discriminator.save(discriminator_path)
    
    print(f"Generator saved to {generator_path}")
    print(f"Discriminator saved to {discriminator_path}")
    
    return generator, discriminator

# Example usage
if __name__ == "__main__":
    # Default paths - adjust these to match your setup
    checkpoint_dir = 'checkpoints'
    checkpoint_prefix = 'ckpt-6'  # Use your latest checkpoint
    save_dir = 'models'
    
    # Convert the checkpoint
    generator, discriminator = convert_combined_checkpoint(
        checkpoint_dir, 
        checkpoint_prefix, 
        save_dir
    )