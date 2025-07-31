import tensorflow as tf
from models import GAN, Generator, Discriminator
from src.dataset import MatDataset
from src.parser import parse_arguments
import os
import src.config as config
import json
from datetime import datetime
import platform

# Parse command line arguments
# args = parse_arguments()
# paths = [args.ct_data_path, args.mask_data_path, args.output_path]

# Load data
train_filenames = [os.path.join('data/train', f) for f in os.listdir('data/train') if f.endswith('.mat')]
train_dataset = MatDataset(train_filenames).dataset

# Store dataset info for training record
dataset_info = {
    'train_files_count': len(train_filenames),
    'train_files': train_filenames[:10],  # Store first 10 filenames as sample
    'data_directory': 'data/train'
}

# Define optimizers
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-5, beta_1=0.5)

# Check if models exist for retraining
model_retraining = False
retrain = 'no'  # Set to 'yes' to retrain existing models, 'no' to create new ones
if os.path.exists('SavedModels/generator.keras') and os.path.exists('SavedModels/discriminator.keras') and retrain == 'yes':
    print("Loading existing models for retraining...")
    generator = tf.keras.models.load_model('SavedModels/generator.keras')
    discriminator = tf.keras.models.load_model('SavedModels/discriminator.keras')
    model_retraining = True
else:
    print("Creating new models...")
    generator = Generator(2, 1).model
    discriminator = Discriminator(2).model

# Define loss function
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Initialize GAN
gan = GAN(generator, discriminator, generator_optimizer, discriminator_optimizer, loss_object)

# Train the model
epochs = 200
training_start_time = datetime.now()
print(f"Training started at: {training_start_time}")

history = gan.fit(train_dataset, epochs, discriminator_ratio=1, generator_ratio=3)

training_end_time = datetime.now()
training_duration = training_end_time - training_start_time
print(f"Training completed at: {training_end_time}")
print(f"Total training duration: {training_duration}")

# Save the models
gan.save_model('SavedModels/generator200ep.keras', 'SavedModels/discriminator200ep.keras')

# Save training history
with open(f'SavedModels/training_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
    json.dump({k: [float(x) for x in v] if isinstance(v, list) else float(v) 
           for k, v in history.items()}, f, indent=4)

# Plot training history
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history['generator_loss'], label='Generator Loss')
plt.plot(history['discriminator_loss'], label='Discriminator Loss')
plt.plot(history['l1_loss'], label='L1 Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history['real_acc'], label='Real Accuracy')
plt.plot(history['gen_acc'], label='Generated Accuracy')
plt.title('Discriminator Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(f'SavedModels/training_history_plot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')

# Save to a json file all the important information about the training
training_info = {
    # Training metadata
    'training_id': f"training_{training_start_time.strftime('%Y%m%d_%H%M%S')}",
    'training_start_time': training_start_time.isoformat(),
    'training_end_time': training_end_time.isoformat(),
    'training_duration_seconds': training_duration.total_seconds(),
    'training_duration_str': str(training_duration),
    
    # Model information
    'model_retraining': model_retraining,
    'generator_model': 'SavedModels/generatorRetrain.keras',
    'discriminator_model': 'SavedModels/discriminatorRetrain.keras',
    'generator_params': generator.count_params(),
    'discriminator_params': discriminator.count_params(),
    
    # Training parameters
    'epochs': epochs,
    'generator_optimizer': str(generator_optimizer.get_config()),
    'discriminator_optimizer': str(discriminator_optimizer.get_config()),
    'loss_object': str(loss_object.get_config()),
    
    # Dataset information
    'dataset_info': dataset_info,
    
    # Configuration from config.py
    'config': {
        'BATCH_SIZE': config.BATCH_SIZE,
        'SHUFFLE': config.SHUFFLE,
        'AUGMENT': config.AUGMENT,
        'RESIZE': config.RESIZE,
        'RESHAPE': config.RESHAPE,
        'TRAIN_EPOCHS': config.TRAIN_EPOCHS,
        'LEARNING_RATE': config.LEARNING_RATE,
        'LAMBDA': config.LAMBDA,
        'NORMALIZATION': config.NORMALIZATION,
        'BUFFER_SIZE': config.BUFFER_SIZE
    },
    
    # System information
    'system_info': {
        'tensorflow_version': tf.__version__,
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'gpu_available': tf.config.list_physical_devices('GPU'),
        'gpu_count': len(tf.config.list_physical_devices('GPU'))
    },
    
    # File paths for reproducibility
    'paths': {
        'train_data_dir': 'data/train',
        'saved_models_dir': 'SavedModels',
        'training_script': 'train.py'
    }
}

# Save training info to JSON file
training_info_filename = f"SavedModels/training_info_{training_start_time.strftime('%Y%m%d_%H%M%S')}.json"
with open(training_info_filename, 'w') as f:
    json.dump(training_info, f, indent=4, default=str)

# Also save a copy as the latest training info
with open('SavedModels/training_info_latest.json', 'w') as f:
    json.dump(training_info, f, indent=4, default=str)

print(f"Training info saved to: {training_info_filename}")
print(f"Latest training info saved to: SavedModels/training_info_latest.json")