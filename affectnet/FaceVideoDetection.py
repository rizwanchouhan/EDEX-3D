import numpy as np
import torch
from PIL import Image
from skimage.io import imread
from torchvision.transforms import ToTensor
from utils.FaceDetector import load_landmark
from affectnet.FaceAlignment import align_face
from skvideo.io import vread, vreader 
from types import GeneratorType
import pickle as pkl

# Define a class named FaceVideoDetection
class FaceVideoDetection(torch.utils.data.Dataset):

    # Initialize the FaceVideoDetection dataset
    def __init__(self, video_name, landmark_path, image_transforms=None, 
                align_landmarks=False, vid_read=None, output_im_range=None, 
                scale_adjustment=1.25,
                target_size_height=256, 
                target_size_width=256,
                ):
        super().__init__()
        # Set attributes
        self.video_name = video_name
        self.landmark_path = landmark_path / "landmarks_original.pkl"
        self.image_transforms = image_transforms
        self.vid_read = vid_read or 'skvreader' 
        self.prev_index = -1

        self.scale_adjustment=scale_adjustment
        self.target_size_height=target_size_height
        self.target_size_width=target_size_width

        self.video_frames = None 
        # Read video frames based on the specified method
        if self.vid_read == "skvread": 
            self.video_frames = vread(str(self.video_name))
        elif self.vid_read == "skvreader": 
            self.video_frames = vreader(str(self.video_name))

        # Load landmarks and landmark types
        with open(self.landmark_path, "rb") as f: 
            self.landmark_list = pkl.load(f)

        with open(landmark_path / "landmark_types.pkl", "rb") as f: 
            self.landmark_types = pkl.load(f)
        
        self.total_len = 0 
        self.frame_map = {} 
        self.index_for_frame_map = {}
        # Create mappings for frame indices
        for i in range(len(self.landmark_list)): 
            for j in range(len(self.landmark_list[i])): 
                self.frame_map[self.total_len + j] = i
                self.index_for_frame_map[self.total_len + j] = j
            self.total_len += len(self.landmark_list[i])

        self.output_im_range = output_im_range

    # Retrieve an item from the dataset
    def __getitem__(self, index):
        # Check if the index is sequential
        if index != self.prev_index + 1 and self.vid_read != 'skvread': 
            raise RuntimeError("This dataset is meant to be accessed in ordered way only (and with 0 or 1 workers)")

        # Retrieve frame and landmark information
        frame_index = self.frame_map[index]
        detection_in_frame_index = self.index_for_frame_map[index]
        landmark = self.landmark_list[frame_index][detection_in_frame_index]
        landmark_type = self.landmark_types[frame_index][detection_in_frame_index]

        # Read the image frame
        if isinstance(self.video_frames, np.ndarray): 
            img = self.video_frames[frame_index, ...]
        elif isinstance(self.video_frames, GeneratorType):
            img = next(self.video_frames)
        else: 
            raise NotImplementedError() 

        # Align the face in the image
        img = align_face(img, landmark, landmark_type, scale_adjustment=1.25, target_size_height=256, target_size_width=256,)
        # Adjust image range if necessary
        if self.output_im_range == 255: 
            img = img * 255.0
        img = img.astype(np.float32)
        img_torch = ToTensor()(img)
        # Apply image transformations if specified
        if self.image_transforms is not None:
            img_torch = self.image_transforms(img_torch)

        batch = {"image" : img_torch}

        self.prev_index += 1
        return batch

    # Get the length of the dataset
    def __len__(self):
        return self.total_len