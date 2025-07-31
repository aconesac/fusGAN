import os
os.environ['MPLBACKEND'] = 'Agg'  # Set matplotlib backend before any imports

import tensorflow as tf
from models import GAN, Generator, Discriminator, SmallDiscriminator
from src.dataset import MatDataset
from src.parser import parse_arguments
from src.statistics import print_detailed_statistics, create_test_results_dict
from src.visualization import plot_training_history, create_metrics_visualization
from src.logging_utils import create_training_info, save_training_info, save_training_history
import numpy as np
from datetime import datetime

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
discriminator_optimizer = tf.keras.optimizers.Adam(0.2e-4, beta_1=0.5)

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
    discriminator = SmallDiscriminator(2).model

# Define loss function
# loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
loss_object = tf.keras.losses.MeanSquaredError()

# Initialize GAN
gan = GAN(generator, discriminator, generator_optimizer, discriminator_optimizer, loss_object)

# Train the model
epochs = 1
training_start_time = datetime.now()
print(f"Training started at: {training_start_time}")

disc_ratio = 1  # Number of discriminator training steps per cycle
gen_ratio = 3  # Number of generator training steps per cycle
history = gan.fit(train_dataset, epochs, disc_steps=disc_ratio, gen_steps=gen_ratio)

training_end_time = datetime.now()
training_duration = training_end_time - training_start_time
print(f"Training completed at: {training_end_time}")
print(f"Total training duration: {training_duration}")

# Evaluate the model on test dataset
test_filenames = [os.path.join('data/test', f) for f in os.listdir('data/test') if f.endswith('.mat')]

print(f"\nNumber of test files: {len(test_filenames)}\n")
if len(test_filenames) == 0:
    raise ValueError("No test files found in the specified directory.")

test_dataset = MatDataset(test_filenames, batch_size=len(test_filenames)).dataset
mse, psnr, ssim, lpips, time = gan.evaluate(test_dataset)

# Convert to numpy arrays for statistics calculation
mse_values = mse.numpy()
psnr_values = psnr.numpy()
ssim_values = ssim.numpy()
lpips_values = lpips

# Print comprehensive statistics
print_detailed_statistics(mse_values, psnr_values, ssim_values, lpips_values, time)

# Create test results dictionary for logging
test_results = create_test_results_dict(test_filenames, mse_values, psnr_values, ssim_values, lpips_values, time)

# Save the models
gan.save_model('SavedModels/generatortrial.keras', 'SavedModels/discriminatortrial.keras')

# Save training history
save_training_history(history)

# Create and save visualizations
plot_training_history(history)
create_metrics_visualization(mse_values, psnr_values, ssim_values, lpips_values)

# Create and save comprehensive training information
training_info = create_training_info(
    training_start_time, training_end_time, training_duration,
    model_retraining, generator, discriminator, epochs,
    disc_ratio, gen_ratio, generator_optimizer, discriminator_optimizer,
    loss_object, dataset_info, history, test_results
)

save_training_info(training_info, training_start_time)