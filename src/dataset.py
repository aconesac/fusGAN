import tensorflow as tf
from scipy.io import loadmat
from src.config import *
import os

class Dataset():
    def __init__(self, filenames: list = None, path: str = None, reshape: tuple = RESHAPE, batch_size: int = BATCH_SIZE, shuffle: bool = SHUFFLE, augment: bool = AUGMENT, resize: list = RESIZE, normalize: str = NORMALIZATION):
        self.filenames = filenames
        self.path = path
        self.num_files = len(filenames)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.resize = resize
        self.reshape = reshape
        self.normalize = normalize
        self.dataset = self.get_dataset()
        

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        return self.process_simulation(self.filenames[idx])

    def get_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.filenames)
        dataset = dataset.map(self.process_simulation, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.cache()
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset
    
    def readFile(self, filename):
        pass

    def process_simulation(self, filename_tensor):
        pass
    
    def augmentData(self, input, output):
        # Apply randomly a left-right flip
        if tf.random.uniform([]) < 0.5:
            input = tf.image.flip_left_right(input)
            output = tf.image.flip_left_right(output)

        # Apply randomly an up-down flip
        if tf.random.uniform([]) < 0.5:
            input = tf.image.flip_up_down(input)
            output = tf.image.flip_up_down(output)

        # Apply randomly a rotation
        k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
        input = tf.image.rot90(input, k=k)
        output = tf.image.rot90(output, k=k)

        return input, output
class MatDataset(Dataset):
    """
    Class to load the dataset from .mat files
    
    Args:
        filenames (list): List of filenames of the .mat files
        reshape (tuple): Shape of the images
        batch_size (int): Batch size
        shuffle (bool): Shuffle the dataset
        augment (bool): Augment the dataset
        resize (list): Resize the images
        normalize (str): Normalization method
    
    return:
        dataset (tf.data.Dataset): Dataset with the images and the PII
    """
    def __init__(self, filenames: list, path = None, reshape: tuple = RESHAPE, batch_size: int = BATCH_SIZE, shuffle: bool = SHUFFLE, augment: bool = AUGMENT, resize: list = RESIZE, normalize: str = NORMALIZATION):
        super().__init__(filenames, path, reshape, batch_size, shuffle, augment, resize, normalize)
    
    def readFile(self, filename):
        if type(filename) != str:
            filename = filename.numpy().decode('utf-8')
            
        simulation_data = loadmat(filename)

        # read the images from the .mat file
        mask = tf.convert_to_tensor(simulation_data['mask'].reshape(self.reshape + (1,)), dtype=tf.float32)
        density = tf.convert_to_tensor(simulation_data['density'].reshape(self.reshape + (1,)), dtype=tf.float32)

        # concatenate the mask and the density map
        input = tf.concat((mask, density), axis=-1)
        input = tf.image.resize(input, self.resize, method='bilinear')

        # Convert PII to a tensor and resize
        output = tf.convert_to_tensor(simulation_data['PII'].reshape(self.reshape + (1,)), dtype=tf.float32)
        output = tf.image.resize(output, self.resize, method='bilinear')

        return input, output

    def process_simulation(self, filename_tensor):
        input, output = tf.py_function(func=self.readFile, inp=[filename_tensor], Tout=[tf.float32, tf.float32])

        if self.augment:
            input, output = self.augmentData(input, output)
        
        if self.normalize == 'standard':
            # Normalize the images to have a mean of 0 and a standard deviation of 1
            input = (input - tf.reduce_mean(input)) / tf.math.reduce_std(input)
            output = (output - tf.reduce_mean(output)) / tf.math.reduce_std(output)
        
        elif self.normalize == 'minmax':
            # Normalize the images to [0, 1]
            input = (input - tf.reduce_min(input)) / (tf.reduce_max(input) - tf.reduce_min(input))
            output = (output - tf.reduce_min(output)) / (tf.reduce_max(output) - tf.reduce_min(output))
            
        elif self.normalize == 'negpos':
            # Normalize the images to [-1, 1]
            input = (input - 0.5) * 2
            output = (output - 0.5) * 2

        return input, output

    
class ImDataset():
    """
    Class to load the dataset from images files
    
    Args:
        filenames (list): List of filenames of the images
        reshape (tuple): Shape of the images
        batch_size (int): Batch size
        shuffle (bool): Shuffle the dataset
        augment (bool): Augment the dataset
        resize (list): Resize the images
        normalize (str): Normalization method
        
    return:
        dataset (tf.data.Dataset): Dataset with the images and the PII
    """
    
    def __init__(self, path: str, filenames: list = None, reshape: tuple = RESHAPE, batch_size: int = BATCH_SIZE, shuffle: bool = SHUFFLE, augment: bool = AUGMENT, resize: list = RESIZE, normalize: str = NORMALIZATION):
        super().__init__(filenames, path, reshape, batch_size, shuffle, augment, resize, normalize)
        
    def setFilenames(self, path):
        ctimages_path = os.path.join(path, 'ct_slices')
        piimages_path = os.path.join(path, 'pi_maps')
        transmask_path = os.path.join(path, 'tr_masks')
            
        ctimages = [os.path.join(ctimages_path, f) for f in os.listdir(ctimages_path)]
        piimages = [os.path.join(piimages_path, f) for f in os.listdir(piimages_path)]
        maskimages = [os.path.join(transmask_path, f) for f in os.listdir(transmask_path)]
        
        self.filenames = list(zip(ctimages, piimages, maskimages))

    
    def readFile(self, filesnames):
        for file in filesnames:
            if type(file) != str:
                file = file.numpy().decode('utf-8')
                
        ctimage = tf.io.read_file(file[0])
        ctimage = tf.image.decode_png(ctimage, channels=1, dtype=tf.float32)
        
        piimage = tf.io.read_file(file[1])
        piimage = tf.image.decode_png(piimage, channels=1, dtype=tf.float32)
        
        maskimage = tf.io.read_file(file[2])
        maskimage = tf.image.decode_png(maskimage, channels=1, dtype=tf.float32)
        
        input = tf.concat((ctimage, piimage), axis=-1)
        input = tf.image.resize(input, self.resize, method='bilinear')
        
        output = tf.image.resize(maskimage, self.resize, method='bilinear')
        
        return input, output
        
    def process_simulation(self, filename_tensor):
        input, output = tf.py_function(func=self.readFile, inp=[filename_tensor], Tout=[tf.float32, tf.float32])

        if self.augment:
            input, output = self.augmentData(input, output)
        
        if self.normalize == 'standard':
            # Normalize the images to have a mean of 0 and a standard deviation of 1
            input = (input - tf.reduce_mean(input)) / tf.math.reduce_std(input)
            output = (output - tf.reduce_mean(output)) / tf.math.reduce_std(output)
        
        elif self.normalize == 'minmax':
            # Normalize the images to [0, 1]
            input = (input - tf.reduce_min(input)) / (tf.reduce_max(input) - tf.reduce_min(input))
            output = (output - tf.reduce_min(output)) / (tf.reduce_max(output) - tf.reduce_min(output))
            
        elif self.normalize == 'negpos':
            # Normalize the images to [-1, 1]
            input = (input - 0.5) * 2
            output = (output - 0.5) * 2

        return input, output
        
    