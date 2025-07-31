import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging (1=INFO, 2=WARNING, 3=ERROR)
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logging (1=INFO, 2=WARNING, 3=ERROR)

from models import GAN, Generator, Discriminator
from src.dataset import MatDataset
import os
import pandas as pd
from tqdm import tqdm

test_filenames = [os.path.join('data/test', f) for f in os.listdir('data/test') if f.endswith('.mat')]

print(f"\nNumber of test files: {len(test_filenames)}\n")
# Ensure that the test dataset is not empty
if len(test_filenames) == 0:
    raise ValueError("No test files found in the specified directory.")

test_dataset = MatDataset(test_filenames, batch_size=len(test_filenames)).dataset

gan = GAN()

gan.load_model('SavedModels/generatorMSElossuplrDisc.keras','SavedModels/discriminatorMSElossuplrDisc.keras')

mse, psnr, ssim, lpips, time = gan.evaluate(test_dataset)

# calculate the average of the metrics
mse_mean = tf.reduce_mean(mse).numpy()
psnr_mean = tf.reduce_mean(psnr).numpy()
ssim_mean = tf.reduce_mean(ssim).numpy()
lpips_mean = tf.reduce_mean(lpips).numpy()

data = {
    'MSE': mse_mean,
    'PSNR': psnr_mean,
    'SSIM': ssim_mean,
    'LPIPS': lpips_mean,
    'Time': time,
}
evaluations_df = pd.DataFrame(data=data, index=[0])
print(evaluations_df)

resultsPath = 'results/upDisc'
if not os.path.exists(resultsPath):
    os.makedirs(resultsPath, exist_ok=True)
evaluations_df.to_csv(resultsPath + '/evaluation_results.csv', index=False)

# Generate images
output_path = resultsPath + '/generated_images'
os.makedirs(output_path, exist_ok=True)
with tqdm(total=len(test_filenames), desc="Generating images") as pbar:
    for i, (mse_val, psnr_val, ssim_val, lpips_val) in enumerate(zip(mse, psnr, ssim, lpips)):
        test_input, tar = next(iter(test_dataset.take(1)))
        gan.generate_images(test_input, tar, output_path=output_path, index=i, metrics=[mse_val.numpy(), psnr_val.numpy(), ssim_val.numpy(), lpips_val])
        pbar.update(1)