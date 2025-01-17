import tensorflow as tf
from models import GAN, Generator, Discriminator
from src.dataset import MatDataset
import os
import pandas as pd

test_filenames = [os.path.join('data/test', f) for f in os.listdir('data/test') if f.endswith('.mat')]

test_dataset = MatDataset(test_filenames, batch_size=len(test_filenames)).dataset

gan = GAN()

gan.load_model('models/generator.keras', 'models/discriminator.keras')

mse, psnr, ssim, lpips, time = gan.evaluate(test_dataset)

evaluations = pd.DataFrame
({
    'MSE': [mse],
    'PSNR': [psnr],
    'SSIM': [ssim],
    'LPIPS': [lpips],
    'Time': [time]
})

evaluations.to_csv('results/evaluation_results.csv', index=False)