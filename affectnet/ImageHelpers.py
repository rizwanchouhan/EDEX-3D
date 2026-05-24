import numpy as np
from skimage.transform import estimate_transform, warp

# Function to convert bounding box coordinates to center and old size
def bbox2point(left, right, top, bottom, type='bbox'):
    ''' 
    bbox from detector and landmarks are different
    Args:
        left (float): Left coordinate of the bounding box.
        right (float): Right coordinate of the bounding box.
        top (float): Top coordinate of the bounding box.
        bottom (float): Bottom coordinate of the bounding box.
        type (str): Type of bounding box, can be 'bbox', 'kpt68', or 'mediapipe'.
    Returns:
        tuple: A tuple containing old size and center coordinates.
    '''
    if type == 'kpt68':
        old_size = (right - left + bottom - top) / 2 * 1.1
        center_x = right - (right - left) / 2.0
        center_y =  bottom - (bottom - top) / 2.0
    elif type == 'bbox':
        old_size = (right - left + bottom - top) / 2
        center_x = right - (right - left) / 2.0 
        center_y = bottom - (bottom - top) / 2.0 + old_size * 0.12
    elif type == "mediapipe":
        old_size = (right - left + bottom - top) / 2 * 1.1
        center_x = right - (right - left) / 2.0 
        center_y = bottom - (bottom - top) / 2.0
    else:
        raise NotImplementedError(f" bbox2point not implemented for {type} ")
    if isinstance(center_x, np.ndarray):
        center = np.stack([center_x, center_y], axis=1)
    else: 
        center = np.array([center_x, center_y])
    return old_size, center

# Function to convert center and size to bounding box points
def point2bbox(center, size):
    size2 = size / 2

    src_pts = np.array(
        [[center[0] - size2, center[1] - size2], [center[0] - size2, center[1] + size2],
         [center[0] + size2, center[1] - size2]])
    return src_pts

# Function to estimate transformation matrix from center and size to target size
def point2transform(center, size, target_size_height, target_size_width):
    target_size_width = target_size_width or target_size_height
    src_pts = point2bbox(center, size)
    dst_pts = np.array([[0, 0], [0, target_size_width - 1], [target_size_height - 1, 0]])
    tform = estimate_transform('similarity', src_pts, dst_pts)
    return tform

# Function to warp an image based on bounding box and transformation
def bbpoint_warp(image, center, size, target_size_height, target_size_width=None, output_shape=None, inv=True, landmarks=None, 
        order=3 
        ):
    target_size_width = target_size_width or target_size_height
    tform = point2transform(center, size, target_size_height, target_size_width)
    tf = tform.inverse if inv else tform
    output_shape = output_shape or (target_size_height, target_size_width)
    dst_image = warp(image, tf, output_shape=output_shape, order=order)
    if landmarks is None:
        return dst_image
    
    if isinstance(landmarks, np.ndarray):
        assert isinstance(landmarks, np.ndarray)
        tf_lmk = tform if inv else tform.inverse
        dst_landmarks = tf_lmk(landmarks[:, :2])
    elif isinstance(landmarks, list): 
        tf_lmk = tform if inv else tform.inverse
        dst_landmarks = [] 
        for i in range(len(landmarks)):
            dst_landmarks += [tf_lmk(landmarks[i][:, :2])]
    elif isinstance(landmarks, dict): 
        tf_lmk = tform if inv else tform.inverse
        dst_landmarks = {}
        for key, value in landmarks.items():
            dst_landmarks[key] = tf_lmk(landmarks[key][:, :2])
    else: 
        raise ValueError("landmarks must be np.ndarray, list or dict")
    return dst_image, dst_landmarks