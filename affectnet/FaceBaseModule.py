import os
import sys
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image
from skimage.io import imread, imsave
from skvideo.io import FFmpegReader
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, Compose, Normalize
from tqdm import tqdm
from affectnet.IO import save_segmentation, save_segmentation_list
from affectnet.ImageHelpers import bbox2point, bbpoint_warp
from affectnet.UnsupImage import UnsupImage
from utils.FaceDetector import FAN, MTCNN, save_landmark
import pickle as pkl
import types

# Define a class named FaceBaseModule
class FaceBaseModule(pl.LightningDataModule):
    # Initialize the FaceBaseModule
    def __init__(self, root_dir, output_dir, processed_subfolder, device=None,
                 face_detector='fan',
                 face_detector_threshold=0.9,
                 image_size=224,
                 scale=1.25,
                 bb_center_shift_x=0.,
                 bb_center_shift_y=0.,
                 processed_ext=".png",
                 save_detection_images=True,
                 save_landmarks_frame_by_frame=True,
                 save_landmarks_one_file=False,
                 save_segmentation_frame_by_frame=True,
                 save_segmentation_one_file=False,
                 ):
        super().__init__()
        # Initialize attributes
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.bb_center_shift_x = bb_center_shift_x
        self.bb_center_shift_y = bb_center_shift_y
        self.processed_ext = processed_ext
        self.save_detection_images = save_detection_images
        self.save_landmarks_frame_by_frame = save_landmarks_frame_by_frame
        self.save_landmarks_one_file = save_landmarks_one_file
        assert not (save_landmarks_one_file and save_landmarks_frame_by_frame)
        self.save_segmentation_frame_by_frame = save_segmentation_frame_by_frame
        self.save_segmentation_one_file = save_segmentation_one_file
        assert not (save_segmentation_one_file and save_segmentation_frame_by_frame)

        # Generate processed folder path based on current time
        if processed_subfolder is None:
            import datetime
            date = datetime.datetime.now()
            processed_folder = os.path.join(output_dir, "processed_%s" % date.strftime("%Y_%b_%d_%H-%M-%S"))
        else:
            processed_folder = os.path.join(output_dir, processed_subfolder)
        self.output_dir = processed_folder

        # Set device to GPU if available, otherwise CPU
        self.device = device or torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Set parameters related to face detection
        self.face_detector_type = face_detector
        self.face_detector_threshold = face_detector_threshold

        self.image_size = image_size
        self.scale = scale

    # Get the maximum number of faces per image
    def _get_max_faces_per_image(self):
        return 1

    # Check if the dataset is a video dataset
    def _is_video_dataset(self):
        return False

    # Instantiate the face detector
    def _instantiate_detector(self, overwrite=False, face_detector=None):
        face_detector = face_detector or self.face_detector_type
        if hasattr(self, 'face_detector'):
            if not overwrite:
                return
            del self.face_detector
        if self.face_detector_type == 'fan':
            self.face_detector = FAN(self.device, threshold=self.face_detector_threshold)
        elif self.face_detector_type == 'mtcnn':
            self.face_detector = MTCNN(self.device)
        elif self.face_detector_type == '3fabrec':
            from utils.TFabRecLandmarkDetector import TFabRec
            self.face_detector = TFabRec(instantiate_detector='sfd', threshold=self.face_detector_threshold)
        elif self.face_detector_type == 'mediapipe':
            from utils.MediaPipeLandmarkDetector import MediaPipeLandmarkDetector
            self.face_detector = MediaPipeLandmarkDetector(threshold=self.face_detector_threshold,
                                                            video_based=self._is_video_dataset(),
                                                            max_faces=self._get_max_faces_per_image())
        elif self.face_detector_type == 'deep3dface':
            from utils.Deep3DFaceLandmarkDetector import Deep3DFaceLandmarkDetector
            self.face_detector = Deep3DFaceLandmarkDetector(instantiate_detector='mtcnn')
        else:
            raise ValueError("Invalid face detector specifier '%s'" % self.face_detector)

    # Detect faces in an image
    def _detect_faces_in_image(self, image_or_path, detected_faces=None):
        # Load image if the input is a path
        if isinstance(image_or_path, (str, Path)):
            image = np.array(imread(image_or_path))
        elif isinstance(image_or_path, np.ndarray):
            image = image_or_path
        else:
            raise ValueError("Invalid image type '%s'" % type(image_or_path))

        # Convert grayscale image to RGB
        if len(image.shape) == 2:
            image = np.tile(image[:, :, None], (1, 1, 3))
        # Remove alpha channel if it exists
        if len(image.shape) == 3 and image.shape[2] > 3:
            image = image[:, :, :3]

        # Get image dimensions
        h, w, _ = image.shape
        # Instantiate the face detector
        self._instantiate_detector()
        # Run face detection
        bounding_boxes, bbox_type, landmarks = self.face_detector.run(image,
                                                                      with_landmarks=True,
                                                                      detected_faces=detected_faces)
        # Normalize image
        image = image / 255.
        detection_images = []
        detection_centers = []
        detection_sizes = []
        detection_landmarks = []
        original_landmarks = landmarks
        if len(bounding_boxes) == 0:
            return detection_images, detection_centers, detection_images, \
                   bbox_type, detection_landmarks, original_landmarks

        # Process detected faces
        for bi, bbox in enumerate(bounding_boxes):
            left = bbox[0]
            right = bbox[2]
            top = bbox[1]
            bottom = bbox[3]
            old_size, center = bbox2point(left, right, top, bottom, type=bbox_type)

            center[0] += abs(right - left) * self.bb_center_shift_x
            center[1] += abs(bottom - top) * self.bb_center_shift_y

            size = int(old_size * self.scale)

            dst_image, dts_landmark = bbpoint_warp(image, center, size, self.image_size, landmarks=landmarks[bi])

            detection_images += [(dst_image * 255).astype(np.uint8)]
            detection_centers += [center]
            detection_sizes += [size]
            detection_landmarks += [dts_landmark]

        del image
        return detection_images, detection_centers, detection_sizes, bbox_type, detection_landmarks, original_landmarks

    # Detect faces in an image wrapper
    def _detect_faces_in_image_wrapper(self, frame_list, fid, out_detection_folder, out_landmark_folder, bb_outfile,
                                       centers_all, sizes_all, detection_fnames_all, landmark_fnames_all,
                                       out_landmarks_all=None, out_landmarks_orig_all=None, out_bbox_type_all=None):
        # Process image frame by frame
        if isinstance(frame_list, (str, Path, list)):
            frame_fname = frame_list[fid]
            detection_ims, centers, sizes, bbox_type, landmarks, orig_landmarks = self._detect_faces_in_image(
                Path(self.output_dir) / frame_fname)
        elif isinstance(frame_list, (np.ndarray, types.GeneratorType)):
            frame_fname = Path(f"{fid:05d}.png")
            if isinstance(frame_list, np.ndarray):
                frame = frame_list[fid]
            else:
                frame = next(frame_list)
            detection_ims, centers, sizes, bbox_type, landmarks, orig_landmarks = self._detect_faces_in_image(frame)

        centers_all += [centers]
        sizes_all += [sizes]
        if out_landmarks_all is not None:
            out_landmarks_all += [landmarks]
        if out_landmarks_orig_all is not None:
            out_landmarks_orig_all += [orig_landmarks]
        if out_bbox_type_all is not None:
            out_bbox_type_all += [[bbox_type] * len(landmarks)]

        detection_fnames = []
        landmark_fnames = []
        # Save detection images and landmarks
        for di, detection in enumerate(detection_ims):
            stem = frame_fname.stem + "_%.03d" % di
            if self.save_detection_images:
                out_detection_fname = out_detection_folder / (stem + self.processed_ext)
                detection_fnames += [out_detection_fname.relative_to(self.output_dir)]
                if self.processed_ext in ['.JPG', '.jpg', ".jpeg", ".JPEG"]:
                    imsave(out_detection_fname, detection, quality=100)
                else:
                    imsave(out_detection_fname, detection)
            if self.save_landmarks_frame_by_frame:
                if self.save_detection_images:
                    out_landmark_fname = out_landmark_folder / (stem + ".pkl")
                    landmark_fnames += [out_landmark_fname.relative_to(self.output_dir)]
                    save_landmark(out_landmark_fname, landmarks[di], bbox_type)
                else:
                    out_landmark_fname = out_landmark_folder / (stem + ".pkl")
                    landmark_fnames += [out_landmark_fname.relative_to(self.output_dir)]
                    save_landmark(out_landmark_fname, orig_landmarks[di], bbox_type)

        detection_fnames_all += [detection_fnames]
        landmark_fnames_all += [landmark_fnames]

        torch.cuda.empty_cache()
        checkpoint_frequency = 100
        if fid % checkpoint_frequency == 0:
            FaceBaseModule.save_detections(bb_outfile, detection_fnames_all, landmark_fnames_all,
                                            centers_all, sizes_all, fid)

    # Segment images
    def _segment_images(self, detection_fnames_or_ims, out_segmentation_folder, path_depth=0, landmarks=None):
        # Check if segmentation already exists
        if self.save_landmarks_one_file:
            overwrite = False
            single_out_file = out_segmentation_folder / "segmentations.pkl"
            if single_out_file.is_file() and not overwrite:
                print(f"Segmentation already found in {single_out_file}, skipping")
                return

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(device)
        net, seg_type, batch_size = self._get_segmentation_net(device)
        ref_size = None
        transforms = None

        # Determine image reading method based on input type
        if isinstance(detection_fnames_or_ims, types.GeneratorType):
            im_read = "skvreader"
        elif isinstance(detection_fnames_or_ims, (FFmpegReader)):
            im_read = "skvffmpeg"
        else:
            im_read = 'pil' if not isinstance(detection_fnames_or_ims[0], np.ndarray) else None

        # Create dataset and data loader
        dataset = UnsupImage(detection_fnames_or_ims, image_transforms=transforms,
                             landmark_list=landmarks,
                             im_read=im_read)
        loader = DataLoader(dataset, batch_size=batch_size,
                            num_workers=4 if im_read not in ["skvreader", "skvffmpeg"] else 1,
                            shuffle=False)

        if self.save_segmentation_one_file:
            out_segmentation_names = []
            out_segmentations = []
            out_segmentation_types = []

        for i, batch in enumerate(tqdm(loader)):
            images = batch['image'].cuda()
            start = time.time()
            with torch.no_grad():
                segmentation = images
            end = time.time()

            if ref_size is None:
                ref_size = Resize((images.shape[2], images.shape[3]), interpolation=Image.NEAREST)

            segmentation = ref_size(segmentation)
            segmentation = segmentation.cpu().numpy()

            if self.save_segmentation_frame_by_frame:
                start = time.time()
                for j in range(segmentation.shape[0]):
                    image_path = batch['path'][j]
                    if path_depth > 0:
                        rel_path = Path(image_path).parent.relative_to(Path(image_path).parents[path_depth])
                        segmentation_path = out_segmentation_folder / rel_path / (Path(image_path).stem + ".pkl")
                    else:
                        segmentation_path = out_segmentation_folder / (Path(image_path).stem + ".pkl")
                    segmentation_path.parent.mkdir(exist_ok=True, parents=True)
                    save_segmentation(segmentation_path, segmentation[j], seg_type)
                print(f" Saving batch {i} took: {end - start}")
                end = time.time()
            if self.save_segmentation_one_file:
                segmentation_names = []
                segmentations = []
                for j in range(segmentation.shape[0]):
                    image_path = batch['path'][j]
                    if path_depth > 0:
                        rel_path = Path(image_path).parent.relative_to(Path(image_path).parents[path_depth])
                        segmentation_path = rel_path / (Path(image_path).stem + ".pkl")
                    else:
                        segmentation_path = Path(image_path).stem
                    segmentation_names += [segmentation_path]
                    segmentations += [segmentation[j]]
                out_segmentation_names += segmentation_names
                out_segmentations += segmentations
                out_segmentation_types += [seg_type] * len(segmentation_names)

        if self.save_landmarks_one_file:
            save_segmentation_list(single_out_file, out_segmentations, out_segmentation_types, out_segmentation_names)
            print("Segmentation saved to %s" % single_out_file)

    # Get segmentation network
    def _get_segmentation_net(self, device, method='bisenet'):
        net = 1
        seg_type = 'face_parsing'
        batch_size = 64

        return net, seg_type, batch_size

    @staticmethod
    def save_landmark_list(fname, landmarks):
        with open(fname, "wb") as f:
            pkl.dump(landmarks, f)

    @staticmethod
    def load_landmark_list(fname):
        with open(fname, "rb") as f:
            landmarks = pkl.load(f)
        return landmarks

    @staticmethod
    def save_landmark_list_v2(fname, landmarks, landmark_confidences, landmark_types):
        with open(fname, "wb") as f:
            pkl.dump(landmarks, f)
            pkl.dump(landmark_confidences, f)
            pkl.dump(landmark_types, f)

    @staticmethod
    def load_landmark_list_v2(fname):
        with open(fname, "rb") as f:
            landmarks = pkl.load(f)
            landmark_confidences = pkl.load(f)
            landmark_types = pkl.load(f)
        return landmarks, landmark_confidences, landmark_types

    @staticmethod
    def save_detections(fname, detection_fnames, landmark_fnames, centers, sizes, last_frame_id):
        with open(fname, "wb") as f:
            pkl.dump(detection_fnames, f)
            pkl.dump(centers, f)
            pkl.dump(sizes, f)
            pkl.dump(last_frame_id, f)
            pkl.dump(landmark_fnames, f)

    @staticmethod
    def load_detections(fname):
        with open(fname, "rb") as f:
            detection_fnames = pkl.load(f)
            centers = pkl.load(f)
            sizes = pkl.load(f)
            try:
                last_frame_id = pkl.load(f)
            except:
                last_frame_id = -1
            try:
                landmark_fnames = pkl.load(f)
            except:
                landmark_fnames = [None] * len(detection_fnames)

        return detection_fnames, landmark_fnames, centers, sizes, last_frame_id