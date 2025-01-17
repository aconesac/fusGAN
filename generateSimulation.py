import tensorflow as tf
from models import GAN, Generator, Discriminator
from src.dataset import MatDataset, ImDataset
from src.parser import parse_arguments_generation
import os

args = parse_arguments_generation()

generator_path = args.model_path
output_path = args.output_path
ct_image_path = args.ct_image_path
mask_path = args.mask_path
sim_path = args.sim_path

gan = GAN()
gan.load_model(generator_path, None)

dataset = ImDataset(filenames=[ct_image_path, mask_path, sim_path])