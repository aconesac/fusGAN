"""
Logging and data persistence utilities for training information.
"""
import json
import platform
import tensorflow as tf
from datetime import datetime
import src.config as config


def create_training_info(training_start_time, training_end_time, training_duration, 
                        model_retraining, generator, discriminator, epochs, 
                        disc_ratio, gen_ratio, generator_optimizer, discriminator_optimizer,
                        loss_object, dataset_info, history, test_results):
    """
    Create comprehensive training information dictionary.
    
    Args:
        training_start_time: datetime object
        training_end_time: datetime object
        training_duration: timedelta object
        model_retraining: bool
        generator: keras model
        discriminator: keras model
        epochs: int
        disc_ratio: int
        gen_ratio: int
        generator_optimizer: keras optimizer
        discriminator_optimizer: keras optimizer
        loss_object: keras loss function
        dataset_info: dict
        history: dict
        test_results: dict
        
    Returns:
        dict: Comprehensive training information
    """
    return {
        # Training metadata
        'training_id': f"training_{training_start_time.strftime('%Y%m%d_%H%M%S')}",
        'training_start_time': training_start_time.isoformat(),
        'training_end_time': training_end_time.isoformat(),
        'training_duration_seconds': training_duration.total_seconds(),
        'training_duration_str': str(training_duration),
        
        # Model information
        'model_retraining': model_retraining,
        'generator_model': 'SavedModels/generatorMSElossuplrDisc.keras',
        'discriminator_model': 'SavedModels/discriminatorMSElossuplrDisc.keras',
        'generator_params': generator.count_params(),
        'discriminator_params': discriminator.count_params(),
        
        # Training parameters
        'epochs': epochs,
        'discriminator_ratio': disc_ratio,
        'generator_ratio': gen_ratio,
        'training_method': 'batch_level_ratios',
        'generator_optimizer': str(generator_optimizer.get_config()),
        'discriminator_optimizer': str(discriminator_optimizer.get_config()),
        'loss_object': str(loss_object.get_config()),
        
        # Dataset information
        'dataset_info': dataset_info,
        
        # Training history summary
        'training_results': {
            'final_generator_loss': float(history['generator_loss'][-1]) if history['generator_loss'] else None,
            'final_discriminator_loss': float(history['discriminator_loss'][-1]) if history['discriminator_loss'] else None,
            'final_l1_loss': float(history['l1_loss'][-1]) if history['l1_loss'] else None,
            'final_real_accuracy': float(history['real_acc'][-1]) if history['real_acc'] else None,
            'final_generated_accuracy': float(history['gen_acc'][-1]) if history['gen_acc'] else None,
            'training_stable': True if len(history['generator_loss']) == epochs else False
        },
        
        # Test set evaluation results
        'test_results': test_results,
        
        # Configuration from config.py
        'config': {
            'BATCH_SIZE': config.BATCH_SIZE,
            'SHUFFLE': config.SHUFFLE,
            'AUGMENT': config.AUGMENT,
            'RESIZE': config.RESIZE,
            'RESHAPE': config.RESHAPE,
            'TRAIN_EPOCHS': epochs,
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


def save_training_info(training_info, training_start_time):
    """
    Save training information to JSON files.
    
    Args:
        training_info: dict containing training information
        training_start_time: datetime object
    """
    # Save training info to timestamped JSON file
    training_info_filename = f"SavedModels/training_info_{training_start_time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(training_info_filename, 'w') as f:
        json.dump(training_info, f, indent=4, default=str)

    # Also save a copy as the latest training info
    with open('SavedModels/training_info_latest.json', 'w') as f:
        json.dump(training_info, f, indent=4, default=str)

    print(f"Training info saved to: {training_info_filename}")
    print(f"Latest training info saved to: SavedModels/training_info_latest.json")


def save_training_history(history):
    """
    Save training history to JSON file.
    
    Args:
        history: dict containing training history
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'SavedModels/training_history_{timestamp}.json'
    
    with open(filename, 'w') as f:
        json.dump({k: [float(x) for x in v] if isinstance(v, list) else float(v) 
                  for k, v in history.items()}, f, indent=4)
    
    print(f"Training history saved to: {filename}")
