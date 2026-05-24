import sys
sys.path.append('/home/abbas/dream')  # Adding the path to the custom modules

# Importing necessary modules
from utils.load import load_model
from affectnet.ImageTest import TestData
import numpy as np
import os
import torch
from skimage.io import imsave
from pathlib import Path
from tqdm import auto
import argparse
from utils.io import save_obj, save_images, save_codes, test


def main():
    # Setting up command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, default= "/home/abbas/dream/input_image/happy.jpg")
    parser.add_argument('--output_folder', type=str, default="/home/abbas/dream/output_image", help="Output folder to save the results to.")
    parser.add_argument('--model_name', type=str, default='DREAM', help='Name of the model to use.')
    parser.add_argument('--path_to_models', type=str, default= "/home/abbas/dream/assets/Pre-Trained")
    parser.add_argument('--save_images', type=bool, default=True, help="If true, output images will be saved")
    parser.add_argument('--save_codes', type=bool, default=False, help="If true, output FLAME values for shape, expression, jaw pose will be saved")
    parser.add_argument('--save_mesh', type=bool, default=False, help="If true, output meshes will be saved")
    parser.add_argument('--mode', type=str, default='detail', help="coarse or detail")
    
    args = parser.parse_args()
    path_to_models = args.path_to_models
    input_folder = args.input_folder
    model_name = args.model_name
    output_folder = args.output_folder + "/" + model_name

    mode = args.mode

    # Loading the DREAM model
    dream, conf = load_model(path_to_models, model_name, mode)
    dream.cuda()  # Moving model to GPU
    dream.eval()  # Setting the model to evaluation mode

    # Creating dataset for testing
    dataset = TestData(input_folder, face_detector="fan", max_detection=20)

    # Iterating over the dataset
    for i in auto.tqdm( range(len(dataset))):  # tqdm for progress visualization
        batch = dataset[i]  # Get a batch of data
        vals, visdict = test(dream, batch)  # Perform testing

        current_bs = batch["image"].shape[0]  # Batch size

        # Iterating over each sample in the batch
        for j in range(current_bs):
            name =  batch["image_name"][j]  # Name of the image

            sample_output_folder = Path(output_folder) / name  # Path for saving output
            sample_output_folder.mkdir(parents=True, exist_ok=True)  # Create the output folder

            # Saving output based on user-defined options
            if args.save_mesh:
                save_obj(dream, str(sample_output_folder / "mesh_coarse.obj"), vals, j)
            if args.save_images:
                save_images(output_folder, name, visdict, with_detection=True, i=j)
            if args.save_codes:
                save_codes(Path(output_folder), name, vals, i=j)

    print("COMPLETED")


if __name__ == '__main__':
    main()
