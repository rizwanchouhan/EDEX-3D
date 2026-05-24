import numpy as np
import torch
from skimage.io import imread
import imgaug
from torch.utils.data._utils.collate import default_collate
from utils.keypoints import KeypointScale, KeypointNormalization
from utils.FaceDetector import load_landmark
from utils.image import numpy_image_to_torch
from .IO import load_segmentation, process_segmentation

class EmotionalImageDatasetOld(torch.utils.data.Dataset):
    def __init__(self, image_list, annotations, labels, image_transforms,
                 path_prefix=None,
                 landmark_list=None,
                 landmark_transform=None,
                 segmentation_list=None,
                 segmentation_transform=None,
                 segmentation_discarded_lables=None,
                 K=None,
                 K_policy=None):
        # Initialize dataset parameters
        self.image_list = image_list
        self.annotations = annotations
        if len(labels) != len(image_list):
            raise RuntimeError("There must be a label for every image")
        self.labels = labels
        self.image_transforms = image_transforms
        self.path_prefix = path_prefix
        if landmark_list is not None and len(landmark_list) != len(image_list):
            raise RuntimeError("There must be a landmark for every image")
        self.landmark_list = landmark_list
        self.landmark_transform = landmark_transform
        if segmentation_list is not None and len(segmentation_list) != len(segmentation_list):
            raise RuntimeError("There must be a segmentation for every image")
        self.segmentation_list = segmentation_list
        self.segmentation_transform = segmentation_transform
        self.segmentation_discarded_labels = segmentation_discarded_lables
        self.K = K 
        self.K_policy = K_policy
        if self.K_policy is not None:
            if self.K_policy not in ['random', 'sequential']:
                raise ValueError(f"Invalid K policy {self.K_policy}")

        # Create label index mapping
        self.labels_set = sorted(list(set(self.labels)))
        self.label2index = {}
        for label in self.labels_set:
            self.label2index[label] = [i for i in range(len(self.labels))
                                       if self.labels[i] == label]

    def __len__(self):
        return len(self.image_list)

    def _get_sample(self, index):
        try:
            path = self.image_list[index]
            if self.path_prefix is not None:
                path = self.path_prefix / path
            img = imread(path)
        except Exception as e:
            print(f"Failed to read '{path}'. "
                  f"File is probably corrupted. Rerun data processing")
            raise e
        img = img.transpose([2, 0, 1]).astype(np.float32)/255.
        img_torch = torch.from_numpy(img)
        if self.image_transforms is not None:
            img_torch = self.image_transforms(img_torch)

        # Construct sample dictionary
        sample = {"image": img_torch, "path": str(self.image_list[index])}

        # Add annotations to sample
        for key in self.annotations.keys():
            sample[key] = torch.tensor(self.annotations[key][index], dtype=torch.float32)

        # Load and transform landmarks if available
        if self.landmark_list is not None:
            landmark_type, landmark = load_landmark(self.path_prefix / self.landmark_list[index])
            landmark_torch = torch.from_numpy(landmark)
            if self.image_transforms is not None:
                if isinstance(self.landmark_transform, KeypointScale):
                    self.landmark_transform.set_scale(img_torch.shape[1] / img.shape[1],
                                                      img_torch.shape[2] / img.shape[2])
                elif isinstance(self.landmark_transform, KeypointNormalization):
                    self.landmark_transform.set_scale(img.shape[1], img.shape[2])
                else:
                    raise ValueError(f"This transform is not supported for landmarks: {type(self.landmark_transform)}")
                landmark_torch = self.landmark_transform(landmark_torch)
            sample["landmark"] = landmark_torch

        # Load and transform segmentations if available
        if self.segmentation_list is not None:
            if self.segmentation_list[index].stem != self.image_list[index].stem:
                raise RuntimeError(f"Name mismatch {self.segmentation_list[index].stem}"
                                   f" vs {self.image_list[index.stem]}")
            seg_image, seg_type = load_segmentation(self.path_prefix / self.segmentation_list[index])
            seg_image = process_segmentation(seg_image, seg_type)
            seg_image_torch = torch.from_numpy(seg_image)
            seg_image_torch = seg_image_torch.view(1, seg_image_torch.shape[0], seg_image_torch.shape[0])
            if self.image_transforms is not None:
                seg_image_torch = self.segmentation_transform(seg_image_torch)
            sample["mask"] = seg_image_torch

        return sample

    def __getitem__(self, index):
        if self.K is None:
            return self._get_sample(index)
        
        # Get samples for KNN
        label = self.labels[index]
        label_indices = self.label2index[label]
        if self.K_policy == 'random':
            indices = np.arange(len(label_indices), dtype=np.int32)
            np.random.shuffle(indices)
            indices = indices[:self.K-1]
        elif self.K_policy == 'sequential':
            indices = []
            idx = label_indices.index(index) + 1
            while len(indices) != self.K-1:
                indices += [label_indices[idx]]
                idx += 1
                idx = idx % len(label_indices)
        else:
            raise ValueError(f"Invalid K policy {self.K_policy}")

        # Create combined batch for KNN
        batches = [self._get_sample(index)]
        for i in range(self.K-1):
            idx = indices[i]
            batches += [self._get_sample(idx)]
        combined_batch = torch.utils.data.dataloader.default_collate(batches)

        return combined_batch


class EmotionalImageDatasetBase(torch.utils.data.Dataset):
    def _augment(self, img, seg_image, landmark, input_img_shape=None):
        # Apply data augmentation transforms
        if self.transforms is not None:
            assert img.dtype == np.uint8
            res = self.transforms(image=img,
                                  segmentation_maps=seg_image,
                                  keypoints=landmark)
            if seg_image is not None and landmark is not None:
                img, seg_image, landmark = res
            elif seg_image is not None:
                img, seg_image = res
            elif landmark is not None:
                img, _, landmark = res
            else:
                img = res

            # Normalize image to range [0, 1]
            assert img.dtype == np.uint8
            if img.dtype != np.float32:
                img = img.astype(np.float32) / 255.0

            assert img.dtype == np.float32

        # Ensure segmentation image is in the correct shape
        if seg_image is not None:
            seg_image = np.squeeze(seg_image)[..., np.newaxis].astype(np.float32)

        # Normalize landmarks if available
        if landmark is not None:
            landmark = np.squeeze(landmark)
            if isinstance(self.landmark_normalizer, KeypointScale):
                self.landmark_normalizer.set_scale(
                    img.shape[0] / input_img_shape[0],
                    img.shape[1] / input_img_shape[1])
            elif isinstance(self.landmark_normalizer, KeypointNormalization):
                self.landmark_normalizer.set_scale(img.shape[0], img.shape[1])
            else:
                raise ValueError(f"Unsupported landmark normalizer type: {type(self.landmark_normalizer)}")
            landmark = self.landmark_normalizer(landmark)

        return img, seg_image, landmark

    def visualize_sample(self, sample):
        # Visualize a sample or a batch of samples
        if isinstance(sample, int):
            sample = self[sample]

        import matplotlib.pyplot as plt
        num_images = 1
        if 'mask' in sample.keys():
            num_images += 1
        if 'landmark' in sample.keys():
            num_images += 1
        if 'landmark_mediapipe' in sample.keys():
            num_images += 1
        if len(sample["image"].shape) >= 4:
            K = sample["image"].shape[0]
            fig, axs = plt.subplots(K, num_images)
        else:
            K = None
            fig, axs = plt.subplots(1, num_images)

        for k in range(K or 1):
            self._plot(axs, K, k, sample)
        plt.show()

    def _plot(self, axs, K, k, sample):
        # Helper function to plot sample details
        from utils.DecaUtils import tensor_vis_landmarks

        def index_axis(i, k):
            if K == 1 or K is None:
                return axs[i]
            return axs[k, i]

        im = sample["image"][k, ...] if K is not None else sample["image"]
        im_expanded = im[np.newaxis, ...]

        i = 0
        index_axis(i, k).imshow(im.numpy().transpose([1, 2, 0]))
        i += 1
        if 'landmark' in sample.keys():
            lmk = sample["landmark"][k, ...] if K is not None else sample["landmark"]
            lmk_expanded = lmk[np.newaxis, ...]
            lmk_im = tensor_vis_landmarks(im_expanded,
                                          self.landmark_normalizer.inv(lmk_expanded),
                                          isScale=False, rgb2bgr=False, scale_colors=True).numpy()[0] \
                .transpose([1, 2, 0])
            index_axis(i, k).imshow(lmk_im)
            i += 1

        if 'landmark_mediapipe' in sample.keys():
            lmk = sample["landmark_mediapipe"][k, ...] if K is not None else sample["landmark_mediapipe"]
            lmk_expanded = lmk[np.newaxis, ...]
            lmk_im = tensor_vis_landmarks(im_expanded,
                                          self.landmark_normalizer.inv(lmk_expanded),
                                          isScale=False, rgb2bgr=False, scale_colors=True).numpy()[0] \
                .transpose([1, 2, 0])
            index_axis(i, k).imshow(lmk_im)
            i += 1

        if 'mask' in sample.keys():
            mask = sample["mask"][k, ...] if K is not None else sample["mask"]
            if mask.ndim == 2:
                mask = mask[np.newaxis, ...]
            index_axis(i, k).imshow(mask.numpy().transpose([1, 2, 0]).squeeze(), cmap='gray')
            i += 1

        if 'path' in sample.keys() and 'label' in sample.keys():
            if K is None:
                print(f"Path = {sample['path']}")
                print(f"Label = {sample['label']}")
            else:
                print(f"Path {k} = {sample['path'][k]}")
                print(f"Label {k} = {sample['label'][k]}")


class EmotionsDataset(EmotionalImageDatasetBase):
    def __init__(self,
                 image_list: list,
                 annotations,
                 labels,
                 transforms: imgaug.augmenters.Augmenter,
                 path_prefix=None,
                 landmark_list=None,
                 segmentation_list=None,
                 segmentation_discarded_lables=None,
                 K=None,
                 K_policy=None
                 ):
        # Initialize dataset parameters
        self.image_list = image_list
        self.annotations = annotations

        # Check if annotations match image list size
        for key in annotations:
            if len(annotations[key]) != len(image_list):
                raise RuntimeError("There must be an annotation of each type for every image but "
                                   f"this is not the case for '{key}'")
        # Check if labels match image list size
        if len(labels) != len(image_list):
            raise RuntimeError("There must be a label for every image")
        self.labels = labels
        self.transforms = transforms
        self.path_prefix = path_prefix

        # Check if landmark list matches image list size
        if landmark_list is not None and len(landmark_list) != len(image_list):
            raise RuntimeError("There must be a landmark for every image")
        self.landmark_list = landmark_list

        # Check if segmentation list matches image list size
        if segmentation_list is not None and len(segmentation_list) != len(segmentation_list):
            raise RuntimeError("There must be a segmentation for every image")
        self.segmentation_list = segmentation_list

        # Initialize landmark normalizer
        self.landmark_normalizer = KeypointNormalization()

        # Initialize segmentation discarded labels
        self.segmentation_discarded_labels = segmentation_discarded_lables

        # Initialize K and K_policy
        self.K = K
        self.K_policy = K_policy

        # Check if K_policy is valid
        if self.K_policy is not None:
            if self.K_policy not in ['random', 'sequential']:
                raise ValueError(f"Invalid K policy {self.K_policy}")

        # Create label to index mapping
        self.labels_set = sorted(list(set(self.labels)))
        self.label2index = {}

        # Include strings samples flag
        self.include_strings_samples = False

        # Populate label to index mapping
        for label in self.labels_set:
            self.label2index[label] = [i for i in range(len(self.labels))
                                       if self.labels[i] == label]

    def __len__(self):
        return len(self.image_list)

    def _get_sample(self, index):
        try:
            # Load image
            path = self.image_list[index]
            if self.path_prefix is not None:
                path = self.path_prefix / path
            img = imread(path)
            input_img_shape = img.shape
        except Exception as e:
            print(f"Failed to read '{path}'. "
                  f"File is probably corrupted. Rerun data processing")
            raise e

        # Load landmarks if available
        if self.landmark_list is not None:
            landmark_type, landmark = load_landmark(self.path_prefix / self.landmark_list[index])
            landmark = landmark[np.newaxis, ...]
        else:
            landmark = None

        # Load segmentation if available
        if self.segmentation_list is not None:
            if self.segmentation_list[index].stem != self.image_list[index].stem:
                raise RuntimeError(f"Name mismatch {self.segmentation_list[index].stem}"
                                   f" vs {self.image_list[index.stem]}")
            seg_image, seg_type = load_segmentation(self.path_prefix / self.segmentation_list[index])
            seg_image = seg_image[np.newaxis, :, :, np.newaxis]
            seg_image = process_segmentation(seg_image, seg_type).astype(np.uint8)
        else:
            seg_image = None

        # Apply data augmentation
        img, seg_image, landmark = self._augment(img, seg_image, landmark, input_img_shape)

        # Construct sample dictionary
        sample = {"image": numpy_image_to_torch(img)}

        # Include string samples if flagged
        if self.include_strings_samples:
            sample["path"] = str(self.image_list[index])
            sample["label"] = str(self.labels[index])

        # Process annotations
        for key in self.annotations.keys():
            annotation = self.annotations[key][index]
            if isinstance(annotation, int):
                annotation = [annotation]

            # Handle empty or None annotations
            if annotation is None or len(annotation) == 0:
                if key == 'au8':
                    sample[key] = torch.tensor([float('nan')] * 8)
                elif key == 'expr7':
                    sample[key] = torch.tensor([float('nan')] * 2)[0:1]
                elif key == 'va':
                    sample[key] = torch.tensor([float('nan')] * 2)
                else:
                    raise RuntimeError(f"Unknown annotation type: '{key}'")
                if len(sample[key].size()) == 0:
                    print(f"[WARNING] Annotation '{key}' is empty for some reason and will be invalidated")
                continue
            sample[key] = torch.tensor(annotation, dtype=torch.float32)
            if len(sample[key].size()) == 0:
                print(f"[WARNING] Annotation '{key}' is empty for some reason (even though it was not None and will be invalidated")
                print("annotation value: ")
                print(annotation)

        # Include landmarks and segmentation in sample if available
        if landmark is not None:
            sample["landmark"] = torch.from_numpy(landmark)
        if seg_image is not None:
            sample["mask"] = numpy_image_to_torch(seg_image)

        return sample

    def __getitem__(self, index):
        # Handle K samples per label policy
        if self.K is None:
            return self._get_sample(index)
        label = self.labels[index]
        label_indices = self.label2index[label]

        if self.K_policy == 'random':
            picked_label_indices = np.arange(len(label_indices), dtype=np.int32)
            np.random.shuffle(picked_label_indices)
            if len(label_indices) < self.K - 1:
                print(f"[WARNING]. Label '{label}' only has {len(label_indices)} samples which is less than {self.K}. S"
                      f"ome samples will be duplicated")
                picked_label_indices = np.concatenate(self.K * [picked_label_indices], axis=0)

            picked_label_indices = picked_label_indices[:self.K - 1]
            indices = [label_indices[i] for i in picked_label_indices]
        elif self.K_policy == 'sequential':
            indices = []
            idx = label_indices.index(index) + 1
            idx = idx % len(label_indices)
            while len(indices) != self.K - 1:
                indices += [label_indices[idx]]
                idx += 1
                idx = idx % len(label_indices)
        else:
            raise ValueError(f"Invalid K policy {self.K_policy}")

        # Collect batches for K samples
        batches = []
        batches += [self._get_sample(index)]
        for i in range(self.K - 1):
            idx = indices[i]
            batches += [self._get_sample(idx)]
        try:
            combined_batch = default_collate(batches)
        except RuntimeError as e:
            print(f"Failed for index {index}")
            for bi, batch in enumerate(batches):
                print(f"Index= {bi}")
                print(f"Path='{batch['path']}")
                print(f"Label='{batch['label']}")
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        print(f"{key} shape='{batch[key].shape}")
            raise e

        return combined_batch


