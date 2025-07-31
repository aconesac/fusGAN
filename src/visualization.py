"""
Visualization utilities for training history and results.
"""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


def plot_training_history(history, save_path=None):
    """
    Plot training history including losses and accuracies.
    
    Args:
        history: Dictionary containing training history
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(history['generator_loss'], label='Generator Loss')
    plt.plot(history['discriminator_loss'], label='Discriminator Loss')
    plt.plot(history['l1_loss'], label='L1 Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(history['real_acc'], label='Real Accuracy')
    plt.plot(history['gen_acc'], label='Generated Accuracy')
    plt.title('Discriminator Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training history plot saved to: {save_path}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = f'SavedModels/training_history_plot_{timestamp}.png'
        plt.savefig(default_path)
        print(f"Training history plot saved to: {default_path}")
    
    plt.close()  # Close the figure to free memory


def create_metrics_visualization(mse_values, psnr_values, ssim_values, lpips_values, save_path=None):
    """
    Create visualizations for test metrics distributions.
    
    Args:
        mse_values: numpy array of MSE values
        psnr_values: numpy array of PSNR values
        ssim_values: numpy array of SSIM values
        lpips_values: numpy array of LPIPS values
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    metrics = [
        ('MSE', mse_values, axes[0, 0]),
        ('PSNR', psnr_values, axes[0, 1]),
        ('SSIM', ssim_values, axes[1, 0]),
        ('LPIPS', lpips_values, axes[1, 1])
    ]
    
    for metric_name, values, ax in metrics:
        ax.hist(values, bins=20, alpha=0.7, edgecolor='black')
        ax.set_title(f'{metric_name} Distribution')
        ax.set_xlabel(metric_name)
        ax.set_ylabel('Frequency')
        ax.axvline(values.mean(), color='red', linestyle='--', label=f'Mean: {values.mean():.4f}')
        ax.axvline(np.median(values), color='green', linestyle='--', label=f'Median: {np.median(values):.4f}')
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Metrics distribution plot saved to: {save_path}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = f'SavedModels/metrics_distribution_{timestamp}.png'
        plt.savefig(default_path)
        print(f"Metrics distribution plot saved to: {default_path}")
    
    plt.close()  # Close the figure to free memory
