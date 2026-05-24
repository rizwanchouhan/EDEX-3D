import pickle as pkl
import compress_pickle as cpkl
import hickle as hkl
from pathlib import Path
import numpy as np
from timeit import default_timer as timer


def load_reconstruction_list(filename):
    """
    Load reconstruction list from file.

    Args:
        filename (str): Path to the file containing the reconstruction list.

    Returns:
        list: List of reconstructions.
    """
    reconstructions = hkl.load(filename)
    return reconstructions


def save_reconstruction_list(filename, reconstructions):
    """
    Save reconstruction list to file.

    Args:
        filename (str): Path to save the reconstruction list.
        reconstructions (list): List of reconstructions to be saved.
    """
    hkl.dump(reconstructions, filename)


def load_emotion_list(filename):
    """
    Load emotion list from file.

    Args:
        filename (str): Path to the file containing the emotion list.

    Returns:
        list: List of emotions.
    """
    emotions = hkl.load(filename)
    return emotions


def save_emotion_list(filename, emotions):
    """
    Save emotion list to file.

    Args:
        filename (str): Path to save the emotion list.
        emotions (list): List of emotions to be saved.
    """
    hkl.dump(emotions, filename)


def save_segmentation_list(filename, seg_images, seg_types, seg_names):
    """
    Save segmentation list to file.

    Args:
        filename (str): Path to save the segmentation list.
        seg_images (list): List of segmentation images.
        seg_types (list): List of segmentation types.
        seg_names (list): List of segmentation names.
    """
    with open(filename, "wb") as f:
        cpkl.dump([seg_types, seg_images, seg_names], f, compression='gzip')


def load_segmentation_list(filename):
    """
    Load segmentation list from file.

    Args:
        filename (str): Path to the file containing the segmentation list.

    Returns:
        tuple: Segmentation images, segmentation types, and segmentation names.
    """
    try:
        with open(filename, "rb") as f:
            seg = cpkl.load(f, compression='gzip')
            seg_types = seg[0]
            seg_images = seg[1]
            seg_names = seg[2]
    except EOFError as e:
        print(f"Error loading segmentation list: {filename}")
        raise e
    return seg_images, seg_types, seg_names


def save_segmentation(filename, seg_image, seg_type):
    """
    Save segmentation to file.

    Args:
        filename (str): Path to save the segmentation.
        seg_image (numpy.ndarray): Segmentation image.
        seg_type (str): Segmentation type.
    """
    with open(filename, "wb") as f:
        cpkl.dump([seg_type, seg_image], f, compression='gzip')


def load_segmentation(filename):
    """
    Load segmentation from file.

    Args:
        filename (str): Path to the file containing the segmentation.

    Returns:
        tuple: Segmentation image and segmentation type.
    """
    with open(filename, "rb") as f:
        seg = cpkl.load(f, compression='gzip')
        seg_type = seg[0]
        seg_image = seg[1]
    return seg_image, seg_type


def save_emotion(filename, emotion_features, emotion_type, version=0):
    """
    Save emotion features to file.

    Args:
        filename (str): Path to save the emotion features.
        emotion_features (list): List of emotion features.
        emotion_type (str): Type of emotion.
        version (int): Version of the saved data.
    """
    with open(filename, "wb") as f:
        cpkl.dump([version, emotion_type, emotion_features], f, compression='gzip')


def load_emotion(filename):
    """
    Load emotion features from file.

    Args:
        filename (str): Path to the file containing the emotion features.

    Returns:
        tuple: Emotion features, emotion type, and version.
    """
    with open(filename, "rb") as f:
        emo = cpkl.load(f, compression='gzip')
        version = emo[0]
        emotion_type = emo[1]
        emotion_features = emo[2]
    return emotion_features, emotion_type


face_parsing_labels = {
    0: 'background',
    1: 'skin',
    2: 'nose',
    3: 'eye_g',
    4: 'l_eye',
    5: 'r_eye',
    6: 'l_brow',
    7: 'r_brow',
    8: 'l_ear',
    9: 'r_ear',
    10: 'mouth',
    11: 'u_lip',
    12: 'l_lip',
    13: 'hair',
    14: 'hat',
    15: 'ear_r',
    16: 'neck_l',
    17: 'neck',
    18: 'cloth'
}

face_parsin_inv_labels = {v: k for k, v in face_parsing_labels.items()}

default_discarded_labels = [
    face_parsin_inv_labels['background'],
    face_parsin_inv_labels['l_ear'],
    face_parsin_inv_labels['r_ear'],
    face_parsin_inv_labels['hair'],
    face_parsin_inv_labels['hat'],
    face_parsin_inv_labels['neck'],
    face_parsin_inv_labels['neck_l']
]


def process_segmentation(segmentation, seg_type, discarded_labels=None):
    """
    Process segmentation image based on the segmentation type.

    Args:
        segmentation (numpy.ndarray): Segmentation image.
        seg_type (str): Type of segmentation.
        discarded_labels (list): List of labels to be discarded.

    Returns:
        numpy.ndarray: Processed segmentation image.
    """
    if seg_type == "face_parsing":
        discarded_labels = discarded_labels or default_discarded_labels
        segmentation_proc = np.isin(segmentation, discarded_labels)
        segmentation_proc = np.logical_not(segmentation_proc)
        segmentation_proc = segmentation_proc.astype(np.float32)
        return segmentation_proc
    else:
        raise ValueError(f"Invalid segmentation type '{seg_type}'")


def load_and_process_segmentation(path):
    """
    Load and process segmentation image from file.

    Args:
        path (str): Path to the file containing the segmentation image.

    Returns:
        numpy.ndarray: Processed segmentation image.
    """
    seg_image, seg_type = load_segmentation(path)
    seg_image = seg_image[np.newaxis, :, :, np.newaxis]
    seg_image = process_segmentation(seg_image, seg_type).astype(np.uint8)
    return seg_image
