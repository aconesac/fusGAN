import argparse
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train GAN model on CT and mask data")
    
    # Adding arguments
    parser.add_argument('--ct_data_path', type=str, required=True, help="Path to the CT slices")
    parser.add_argument('--mask_data_path', type=str, required=True, help="Path to the mask data")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the trained models")

    # Parse the arguments
    args = parser.parse_args()
    
    # Return the arguments as an object
    return args

def parse_arguments_generation():
    parser = argparse.ArgumentParser(description="Generate synthetic simulation")
    
    # Adding arguments
    parser.add_argument('--model_path', type=str, required=True, help="Path to the generator model")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the generated simulation")
    parser.add_argument('--ct_image_path', type=str, required=True, help="Path to the CT slice")
    parser.add_argument('--mask_path', type=str, required=True, help="Path to the mask image")
    parser.add_argument('--sim_path', type=str, required=True, help="Path to the simulation image")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Return the arguments as an object
    return args