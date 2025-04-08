import tensorflow as tf
import os
import glob
from models import Generator, Discriminator

def identify_checkpoint_format(checkpoint_path):
    """
    Identify the format of a TensorFlow checkpoint
    
    Args:
        checkpoint_path (str): Path to the checkpoint or directory
        
    Returns:
        tuple: (format_type, path)
            format_type: 'v1', 'v2', 'saved_model', or 'unknown'
            path: Actual path to load from
    """
    # Check if it's a directory
    if os.path.isdir(checkpoint_path):
        # Check for SavedModel format
        if os.path.exists(os.path.join(checkpoint_path, 'saved_model.pb')):
            return 'saved_model', checkpoint_path
        
        # Look for checkpoint file
        if os.path.exists(os.path.join(checkpoint_path, 'checkpoint')):
            return 'v2', tf.train.latest_checkpoint(checkpoint_path)
        
        # Look for .meta files (v1 checkpoint)
        meta_files = glob.glob(os.path.join(checkpoint_path, '*.meta'))
        if meta_files:
            return 'v1', meta_files[0].replace('.meta', '')
    else:
        # Check for direct checkpoint files
        if checkpoint_path.endswith('.h5') or checkpoint_path.endswith('.hdf5'):
            return 'h5', checkpoint_path
        
        if os.path.exists(checkpoint_path + '.index'):
            return 'v2', checkpoint_path
            
        if os.path.exists(checkpoint_path + '.meta'):
            return 'v1', checkpoint_path
    
    return 'unknown', None

def load_weights_from_checkpoint(model, checkpoint_path, format_type=None):
    """
    Load weights from a checkpoint into a model
    
    Args:
        model: TensorFlow model to load weights into
        checkpoint_path (str): Path to the checkpoint
        format_type (str, optional): Format type ('v1', 'v2', 'saved_model', 'h5')
        
    Returns:
        model: Model with loaded weights
    """
    if format_type is None:
        format_type, path = identify_checkpoint_format(checkpoint_path)
        if format_type == 'unknown':
            raise ValueError(f"Could not identify checkpoint format for {checkpoint_path}")
        checkpoint_path = path
    
    print(f"Loading checkpoint using format: {format_type}")
    
    if format_type == 'saved_model':
        # Load from SavedModel format
        loaded_model = tf.keras.models.load_model(checkpoint_path)
        model.set_weights(loaded_model.get_weights())
    
    elif format_type == 'h5':
        # Load from h5 format
        model.load_weights(checkpoint_path)
    
    elif format_type == 'v1' or format_type == 'v2':
        # Load from checkpoint format
        checkpoint = tf.train.Checkpoint(model=model)
        status = checkpoint.restore(checkpoint_path)
        status.expect_partial()
    
    return model

def convert_gan_checkpoints(generator_path, discriminator_path, save_dir, 
                           gen_input_channels=2, gen_output_channels=1, disc_input_channels=2):
    """
    Convert GAN checkpoints to .keras format
    
    Args:
        generator_path (str): Path to generator checkpoint
        discriminator_path (str): Path to discriminator checkpoint
        save_dir (str): Directory to save .keras models
        gen_input_channels (int): Number of input channels for generator
        gen_output_channels (int): Number of output channels for generator
        disc_input_channels (int): Number of input channels for discriminator
        
    Returns:
        tuple: (generator, discriminator) loaded models
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize models with correct architecture
    generator_model = Generator(gen_input_channels, gen_output_channels).model
    discriminator_model = Discriminator(disc_input_channels).model
    
    # Load weights if paths are provided
    if generator_path:
        gen_format, gen_path = identify_checkpoint_format(generator_path)
        if gen_format != 'unknown':
            generator_model = load_weights_from_checkpoint(generator_model, gen_path, gen_format)
            generator_model.save(os.path.join(save_dir, 'generator.keras'))
            print(f"Generator saved to {os.path.join(save_dir, 'generator.keras')}")
        else:
            print(f"Could not identify generator checkpoint format for {generator_path}")
    
    if discriminator_path:
        disc_format, disc_path = identify_checkpoint_format(discriminator_path)
        if disc_format != 'unknown':
            discriminator_model = load_weights_from_checkpoint(discriminator_model, disc_path, disc_format)
            discriminator_model.save(os.path.join(save_dir, 'discriminator.keras'))
            print(f"Discriminator saved to {os.path.join(save_dir, 'discriminator.keras')}")
        else:
            print(f"Could not identify discriminator checkpoint format for {discriminator_path}")
    
    return generator_model, discriminator_model

# Example usage
if __name__ == "__main__":
    pass
    # Example with different paths
    # convert_gan_checkpoints(
    #     'checkpoints/generator',        # Directory or specific checkpoint
    #     'checkpoints/discriminator',    # Directory or specific checkpoint
    #     'models'                        # Output directory
    # )
