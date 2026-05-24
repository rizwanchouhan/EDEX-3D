# EDEX-3D: Emotion-Aware Modeling for High-Fidelity 3D Facial Expression Reconstruction

<img style="max-width: 100%;" src="https://github.com/rizwanchouhan/emopoi/blob/main/resources/wax.png" alt="VERHM Overview">

## About the Project

The proposed approach, EDEX-3D, enhances parametric 3D face model reconstruction to capture nuanced emotional expressions. By integrating emotional detail encoding and decoding with FLAME and Render modules, the project synthesizes realistic 3D facial expressions. Training with deep perceptual emotion analysis improves expression fidelity, surpassing baseline techniques and demonstrating efficacy in health monitoring applications.

## Installation

1. **Clone the Repository**: Download the project from GitHub.
2. **Set Up Conda Environment**: Create a Conda environment named EDEX3D with Python 3.9:
    ```bash
    conda create --name EDEX3D python=3.9
    conda activate EDEX3D
    conda install mamba -n base -c conda-forge
    ```
3. **Install Dependencies**: Install the required packages using pip:
    ```bash
    pip install -r requirements.txt
    ```

## Datasets

The project utilizes three diverse datasets:

- **IEMOCAP Dataset**: Annotated with categorical labels like happiness, sadness, anger, and neutral, it offers ten hours of audio and video recordings from spontaneous sessions between actors. [IEMOCAP Dataset](https://sail.usc.edu/iemocap/iemocap_release.htm)
- **AffectNet Dataset**: With over one million facial images annotated with seven primary emotions, it provides a rich resource for emotion recognition research. [AffectNet Dataset](http://mohammadmahoor.com/affectnet/)
- **CMU-MOSEI Dataset**: Annotated with continuous emotion dimensions like valence and arousal, it includes multimodal data suitable for emotion recognition and sentiment analysis tasks. [CMU-MOSEI Dataset](http://multicomp.cs.cmu.edu/resources/cmu-mosi-dataset/)

## Steps for Training

1. **Download Dataset**: Download the dataset from the provided links.
2. **Data Processing**: Run the `process_data.py` script to preprocess the dataset.
3. **Initiate Training**: Execute `python training/train_EDEX3D.py` to begin the training process.

## Running the Demo

Follow these steps to run the demo of the project:

1. **Download Pre-trained Model**: Download the pre-trained model from the [link](https://drive.google.com) and put it in the Pre-Trained/EDEX3D directory.
2. **Download FLAME Model**: Download the FLAME model from the [link](https://drive.google.com) and put it in the FLAME directory.
3. **Demo for Images**: Run `python EDEX3D_image.py` for processing images.
4. **Demo for Videos**: Run `python EDEX3D_video.py` for processing videos.
