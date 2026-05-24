import numpy as np
from pathlib import Path
from affectnet.ImageHelpers import bbox2point, bbpoint_warp
import skvideo
import types

# Function to align a single face image
def align_face(image, landmarks, landmark_type, scale_adjustment, target_size_height, target_size_width=None):
    # Determine bounding box coordinates
    left = landmarks[:,0].min()
    top =  landmarks[:,1].min()
    right =  landmarks[:,0].max()
    bottom = landmarks[:,1].max()

    # Calculate old size and center
    old_size, center = bbox2point(left, right, top, bottom, type=landmark_type)
    size = (old_size * scale_adjustment).astype(np.int32)

    # Warp image and landmarks
    img_warped, lmk_warped = bbpoint_warp(image, center, size, target_size_height, target_size_width, landmarks=landmarks)

    return img_warped

# Function to align a video
def align_video(video, centers, sizes, landmarks, target_size_height, target_size_width=None):
    # Read video if it's a path or convert to numpy array if already in memory
    if isinstance(video, (str, Path)):
        video = skvideo.io.vread(video)
    elif isinstance(video, (np.ndarray, types.GeneratorType)):
        pass
    else:
        raise ValueError("video must be a string, Path, or numpy array")

    aligned_video = []
    warped_landmarks = []
    # Iterate over frames in the video
    if isinstance(video, np.ndarray):
        for i in range(len(centers)): 
            img_warped, lmk_warped = bbpoint_warp(video[i], centers[i], sizes[i], 
                    target_size_height=target_size_height, target_size_width=target_size_width, 
                    landmarks=landmarks[i])
            aligned_video.append(img_warped)
            warped_landmarks += [lmk_warped]
    elif isinstance(video, types.GeneratorType): 
        for i, frame in enumerate(video):
            img_warped, lmk_warped = bbpoint_warp(frame, centers[i], sizes[i], 
                    target_size_height=target_size_height, target_size_width=target_size_width, 
                    landmarks=landmarks[i])
            aligned_video.append(img_warped)
            warped_landmarks += [lmk_warped] 

    aligned_video = np.stack(aligned_video, axis=0)
    return aligned_video, warped_landmarks

# Function to align a video and save
def align_and_save_video(video, out_video_path, centers, sizes, landmarks, target_size_height, target_size_width=None, output_dict=None):
    # Read video if it's a path or convert to numpy array if already in memory
    if isinstance(video, (str, Path)):
        video = skvideo.io.vread(video)
    elif isinstance(video, (np.ndarray, types.GeneratorType)):
        pass
    else:
        raise ValueError("video must be a string, Path, or numpy array")

    # Create video writer
    writer = skvideo.io.FFmpegWriter(str(out_video_path), outputdict=output_dict)
    warped_landmarks = []
    # Iterate over frames in the video
    if isinstance(video, np.ndarray):
        for i in range(len(centers)): 
            img_warped, lmk_warped = bbpoint_warp(video[i], centers[i], sizes[i], 
                    target_size_height=target_size_height, target_size_width=target_size_width, 
                    landmarks=landmarks[i])
            img_warped = (img_warped * 255).astype(np.uint8)
            writer.writeFrame(img_warped)
            warped_landmarks += [lmk_warped]
    elif isinstance(video, types.GeneratorType): 
        for i, frame in enumerate(video):
            img_warped, lmk_warped = bbpoint_warp(frame, centers[i], sizes[i], 
                    target_size_height=target_size_height, target_size_width=target_size_width, 
                    landmarks=landmarks[i])
            img_warped = (img_warped * 255).astype(np.uint8)
            writer.writeFrame(img_warped)
            warped_landmarks += [lmk_warped] 
    writer.close()

    return warped_landmarks
