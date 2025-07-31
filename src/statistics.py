"""
Statistics utilities for model evaluation and analysis.
"""
import numpy as np
import pandas as pd


def calculate_comprehensive_stats(mse_values, psnr_values, ssim_values, lpips_values, evaluation_time):
    """
    Calculate comprehensive statistics for evaluation metrics.
    
    Args:
        mse_values: numpy array of MSE values
        psnr_values: numpy array of PSNR values
        ssim_values: numpy array of SSIM values
        lpips_values: numpy array of LPIPS values
        evaluation_time: float, time taken for evaluation
        
    Returns:
        dict: Dictionary containing all statistics
    """
    def calculate_metric_stats(values, metric_name):
        """Calculate statistics for a single metric."""
        return {
            f'{metric_name}_mean': float(np.mean(values)),
            f'{metric_name}_std': float(np.std(values)),
            f'{metric_name}_min': float(np.min(values)),
            f'{metric_name}_max': float(np.max(values)),
            f'{metric_name}_median': float(np.median(values)),
            f'{metric_name}_q25': float(np.percentile(values, 25)),
            f'{metric_name}_q75': float(np.percentile(values, 75)),
        }
    
    # Calculate statistics for all metrics
    stats = {}
    stats.update(calculate_metric_stats(mse_values, 'MSE'))
    stats.update(calculate_metric_stats(psnr_values, 'PSNR'))
    stats.update(calculate_metric_stats(ssim_values, 'SSIM'))
    stats.update(calculate_metric_stats(lpips_values, 'LPIPS'))
    stats['Time'] = evaluation_time
    
    return stats


def print_detailed_statistics(mse_values, psnr_values, ssim_values, lpips_values, evaluation_time):
    """
    Print detailed statistics in a formatted way.
    
    Args:
        mse_values: numpy array of MSE values
        psnr_values: numpy array of PSNR values
        ssim_values: numpy array of SSIM values
        lpips_values: numpy array of LPIPS values
        evaluation_time: float, time taken for evaluation
    """
    # Calculate and display comprehensive statistics
    test_stats = calculate_comprehensive_stats(mse_values, psnr_values, ssim_values, lpips_values, evaluation_time)
    
    # Create comprehensive evaluation DataFrame
    evaluations_df = pd.DataFrame([test_stats])
    print("\nTest Set Evaluation Statistics:")
    print("=" * 50)
    print(evaluations_df.round(4))

    # Print detailed statistics for each metric
    print("\n" + "="*60)
    print("DETAILED TEST STATISTICS")
    print("="*60)

    metrics = [('MSE', mse_values), ('PSNR', psnr_values), ('SSIM', ssim_values), ('LPIPS', lpips_values)]
    for metric_name, values in metrics:
        print(f"\n{metric_name}:")
        print(f"  Mean: {np.mean(values):.6f}")
        print(f"  Std:  {np.std(values):.6f}")
        print(f"  Min:  {np.min(values):.6f}")
        print(f"  Max:  {np.max(values):.6f}")
        print(f"  Median: {np.median(values):.6f}")
        print(f"  Q25:  {np.percentile(values, 25):.6f}")
        print(f"  Q75:  {np.percentile(values, 75):.6f}")
        print(f"  Range: {np.max(values) - np.min(values):.6f}")
        print(f"  CV:   {(np.std(values) / np.mean(values) * 100):.2f}%")

    print(f"\nEvaluation Time: {evaluation_time:.4f} seconds")
    print("="*60)


def create_test_results_dict(test_filenames, mse_values, psnr_values, ssim_values, lpips_values, evaluation_time):
    """
    Create a comprehensive test results dictionary for JSON logging.
    
    Args:
        test_filenames: list of test file names
        mse_values: numpy array of MSE values
        psnr_values: numpy array of PSNR values
        ssim_values: numpy array of SSIM values
        lpips_values: numpy array of LPIPS values
        evaluation_time: float, time taken for evaluation
        
    Returns:
        dict: Dictionary containing test results for JSON export
    """
    return {
        'test_files_count': len(test_filenames),
        'evaluation_time': float(evaluation_time),
        # Summary statistics
        'mse_mean': float(np.mean(mse_values)),
        'mse_std': float(np.std(mse_values)),
        'mse_min': float(np.min(mse_values)),
        'mse_max': float(np.max(mse_values)),
        'mse_median': float(np.median(mse_values)),
        'mse_q25': float(np.percentile(mse_values, 25)),
        'mse_q75': float(np.percentile(mse_values, 75)),
        
        'psnr_mean': float(np.mean(psnr_values)),
        'psnr_std': float(np.std(psnr_values)),
        'psnr_min': float(np.min(psnr_values)),
        'psnr_max': float(np.max(psnr_values)),
        'psnr_median': float(np.median(psnr_values)),
        'psnr_q25': float(np.percentile(psnr_values, 25)),
        'psnr_q75': float(np.percentile(psnr_values, 75)),
        
        'ssim_mean': float(np.mean(ssim_values)),
        'ssim_std': float(np.std(ssim_values)),
        'ssim_min': float(np.min(ssim_values)),
        'ssim_max': float(np.max(ssim_values)),
        'ssim_median': float(np.median(ssim_values)),
        'ssim_q25': float(np.percentile(ssim_values, 25)),
        'ssim_q75': float(np.percentile(ssim_values, 75)),
        
        'lpips_mean': float(np.mean(lpips_values)),
        'lpips_std': float(np.std(lpips_values)),
        'lpips_min': float(np.min(lpips_values)),
        'lpips_max': float(np.max(lpips_values)),
        'lpips_median': float(np.median(lpips_values)),
        'lpips_q25': float(np.percentile(lpips_values, 25)),
        'lpips_q75': float(np.percentile(lpips_values, 75)),
        
        # All individual values for detailed analysis
        'mse_all_values': [float(x) for x in mse_values],
        'psnr_all_values': [float(x) for x in psnr_values],
        'ssim_all_values': [float(x) for x in ssim_values],
        'lpips_all_values': [float(x) for x in lpips_values]
    }
