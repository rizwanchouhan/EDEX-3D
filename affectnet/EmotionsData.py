import os, sys
from enum import Enum
from pathlib import Path
import numpy as np
import scipy as sp
import torch
import pytorch_lightning as pl
import pandas as pd
import pickle as pkl
from skimage.io import imread, imsave
from affectnet.IO import load_segmentation, process_segmentation, load_emotion, save_emotion
from utils.image import numpy_image_to_torch
from utils.keypoints import KeypointNormalization
import imgaug
from affectnet.FaceBaseModule import FaceBaseModule
from affectnet.ImageHelpers import bbox2point, bbpoint_warp
from affectnet.EmotionsDataset import EmotionalImageDatasetBase
from affectnet.UnsupImage import UnsupImage
from utils.FaceDetector import save_landmark, load_landmark
from tqdm import auto
import traceback
from torch.utils.data.dataloader import DataLoader
from utils.imgaug import create_image_augmenter
from torchvision.transforms import Resize, Compose
from sklearn.neighbors import NearestNeighbors
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.sampler import WeightedRandomSampler
from enum import Enum


class ActionUnitTypes(Enum):

    EMOTIONET12 = 1
    EMOTIONET23 = 2
    ALL = 3

    @staticmethod
    def AUtype2AUlist(t):
        if t == ActionUnitTypes.EMOTIONET12:
            return [1, 2, 4, 5, 6, 9, 12, 17, 20, 25, 26, 43]
        elif t == ActionUnitTypes.EMOTIONET23:
            return [1, 2, 4, 5, 6, 9, 10, 12, 15, 17, 18, 20, 24, 25, 26, 28, 43, 51, 52, 53, 54, 55, 56]
        elif t == ActionUnitTypes.ALL:
            return list(range(1,60))
        raise ValueError(f"Invalid action unit type {t}")

    @staticmethod
    def AUtype2AUstring_list(t):
        l = ActionUnitTypes.AUtype2AUlist(t)
        string_list = [f"AU{i}" for i in l]
        return string_list

    @staticmethod
    def numAUs(t):
        return len(ActionUnitTypes.AUtype2AUlist(t))

    @staticmethod
    def AU_num_2_name(num):
        d = {}
        d[1] = "Inner Brow Raiser"
        d[2] = "Outer Brow Raiser"
        d[4] = "Brow Lowerer"
        d[5] = "Upper Lid Raiser"
        d[6] = "Cheek Raiser"
        d[9] = "Nose Wrinkler"
        d[10] = "Upper Lip Raiser"
        d[12] = "Lip Corner Puller"
        d[15] = "Lip Corner Depressor"
        d[17] = "Chin Raiser"
        d[18] = "Lip Puckerer"
        d[20] = "Lip Stretcher"
        d[24] = "Lip Pressor"
        d[25] = "Lips Part"
        d[26] = "Jaw Drop"
        d[28] = "Lip Suck"
        d[43] = "Eyes Closed"
        if num not in d.keys():
            raise ValueError(f"invalid AU {num}")
        return d[num]


class EmotionsData(FaceBaseModule):

    def __init__(self,
                 input_dir,
                 output_dir,
                 processed_subfolder = None,
                 ignore_invalid = False,
                 face_detector='fan',
                 face_detector_threshold=0.9,
                 image_size=224,
                 scale=1.25,
                 bb_center_shift_x = 0.,
                 bb_center_shift_y = 0.,
                 processed_ext=".jpg",
                 device=None,
                 augmentation=None,
                 train_batch_size=64,
                 val_batch_size=64,
                 test_batch_size=64,
                 num_workers=0,
                 drop_last=False,
                 au_type = ActionUnitTypes.EMOTIONET12
                 ):
        super().__init__(input_dir, output_dir, processed_subfolder,
                         face_detector=face_detector,
                         face_detector_threshold=face_detector_threshold,
                         image_size=image_size,
                         bb_center_shift_x=bb_center_shift_x,
                         bb_center_shift_y=bb_center_shift_y,
                         processed_ext=processed_ext,
                         scale=scale,
                         device=device)
        self.input_dir = Path(self.root_dir)
        dfs_fnames = sorted(list(self.input_dir.glob("image_list_*.csv")))
        dfs = [pd.read_csv(dfs_fname) for dfs_fname in dfs_fnames]
        self.df = pd.concat(dfs, ignore_index=True, sort=False)

        self.face_detector_type = 'fan'
        self.scale = scale

        self.image_path = Path(self.output_dir) / "detections"
        self.au_type = au_type
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.augmentation = augmentation
        self.drop_last = drop_last

    @property
    def subset_size(self):
        return 1000

    @property
    def num_subsets(self):
        num_subsets = len(self.df) // self.subset_size
        if len(self.df) % self.subset_size != 0:
            num_subsets += 1
        return num_subsets

    def _detect_faces(self):
        num_subsets = self.num_subsets
        if len(self.df) % self.subset_size != 0:
            num_subsets += 1
        for sid in range(self.num_subsets):
            self._detect_landmarks_and_segment_subset(self.subset_size * sid, min((sid + 1) * self.subset_size, len(self.df)))

    def _extract_emotion_features(self):
        num_subsets = len(self.df) // self.subset_size
        if len(self.df) % self.subset_size != 0:
            num_subsets += 1
        for sid in range(self.num_subsets):
            self._extract_emotion_features_from_subset(self.subset_size * sid, min((sid + 1) * self.subset_size, len(self.df)))

    def _path_to_detections(self):
        return Path(self.output_dir) / "detections"

    def _path_to_segmentations(self):
        return Path(self.output_dir) / "segmentations"

    def _path_to_landmarks(self):
        return Path(self.output_dir) / "landmarks"

    def _path_to_emotions(self):
        return Path(self.output_dir) / "emotions"

    def _get_emotion_net(self, device):
        from losses.EmonetLoader import get_emonet

        net = get_emonet()
        net = net.to(device)

        return net, "emo_net"

    def _extract_emotion_features_from_subset(self, start_i, end_i):
        self._path_to_emotions().mkdir(parents=True, exist_ok=True)

        print(f"Processing subset {start_i // self.subset_size}")
        image_file_list = []
        for i in auto.tqdm(range(start_i, end_i)):
            im_file = self.df.loc[i]["subDirectory_filePath"]
            in_detection_fname = self._path_to_detections() / Path(im_file).parent / (Path(im_file).stem + self.processed_ext)
            if in_detection_fname.is_file():
                image_file_list += [in_detection_fname]

        transforms = Compose([
            Resize((256, 256)),
        ])
        batch_size = 32
        dataset = UnsupImage(image_file_list, image_transforms=transforms, im_read='pil')
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(device)
        net, emotion_type = self._get_emotion_net(device)

        for i, batch in enumerate(auto.tqdm(loader)):
            images = batch['image'].cuda()
            with torch.no_grad():
                out = net(images, intermediate_features=True)
            emotion_features = {key : val.detach().cpu().numpy() for key, val in out.items()}

            for j in range(images.size()[0]):
                image_path = batch['path'][j]
                out_emotion_folder = self._path_to_emotions() / Path(image_path).parent.name
                out_emotion_folder.mkdir(exist_ok=True, parents=True)
                emotion_path = out_emotion_folder / (Path(image_path).stem + ".pkl")
                emotion_feature_j = {key: val[j] for key, val in emotion_features.items()}
                del emotion_feature_j['emo_feat'] 
                del emotion_feature_j['heatmap'] 
                save_emotion(emotion_path, emotion_feature_j, emotion_type)


    def _detect_landmarks_and_segment_subset(self, start_i, end_i):
        self._path_to_detections().mkdir(parents=True, exist_ok=True)
        self._path_to_segmentations().mkdir(parents=True, exist_ok=True)
        self._path_to_landmarks().mkdir(parents=True, exist_ok=True)

        detection_fnames = []
        out_segmentation_folders = []

        status_array = np.memmap(self.status_array_path,
                                 dtype=np.bool,
                                 mode='r',
                                 shape=(self.num_subsets,)
                                 )

        completed = status_array[start_i // self.subset_size]
        if not completed:
            print(f"Processing subset {start_i // self.subset_size}")
            for i in auto.tqdm(range(start_i, end_i)):
                im_file = self.df.loc[i]["path"]

                im_fullfile = Path(self.input_dir) / "images" / im_file
                try:
                    detection, _, _, bbox_type, landmarks = self._detect_faces_in_image(im_fullfile, detected_faces=None)
                except Exception as e:
                    print(f"Failed to load file:")
                    print(f"{im_fullfile}")
                    print(traceback.print_exc())
                    continue

                if len(detection) == 0:
                    print(f"Skipping file {im_fullfile} because no face was detected.")
                    continue

                out_detection_fname = self._path_to_detections() / Path(im_file).parent / (Path(im_file).stem + self.processed_ext)
                out_detection_fname.parent.mkdir(exist_ok=True)
                detection_fnames += [out_detection_fname]
                if self.processed_ext in [".jpg", ".JPG"]:
                    imsave(out_detection_fname, detection[0], quality=100)
                else:
                    imsave(out_detection_fname, detection[0])

                out_landmark_fname = self._path_to_landmarks() / Path(im_file).parent / (Path(im_file).stem + ".pkl")
                out_landmark_fname.parent.mkdir(exist_ok=True)
                save_landmark(out_landmark_fname, landmarks[0], bbox_type)

            self._segment_images(detection_fnames, self._path_to_segmentations(), path_depth=1)

            status_array = np.memmap(self.status_array_path,
                                     dtype=np.bool,
                                     mode='r+',
                                     shape=(self.num_subsets,)
                                     )
            status_array[start_i // self.subset_size] = True
            status_array.flush()
            del status_array
            print(f"Processing subset {start_i // self.subset_size} finished")
        else:
            print(f"Subset {start_i // self.subset_size} is already processed")

    @property
    def status_array_path(self):
        return Path(self.output_dir) / "status.memmap"

    @property
    def is_processed(self):
        status_array = np.memmap(self.status_array_path,
                                 dtype=np.bool,
                                 mode='r',
                                 shape=(self.num_subsets,)
                                 )
        all_processed = status_array.all()
        return all_processed

    def _split_train_val(self, seed=0, ratio=0.9):
        self.val_dataframe_path = Path(self.output_dir) / f"validation_set_{seed}_{ratio:0.4f}.csv"
        self.train_dataframe_path = Path(self.output_dir) / f"training_set_{seed}_{ratio:0.4f}.csv"
        self.full_dataframe_path = Path(self.output_dir) / f"full_dataset.csv"

        if self.val_dataframe_path.is_file() and self.train_dataframe_path.is_file():
            self.train_df = pd.read_csv(self.train_dataframe_path)
            self.val_df = pd.read_csv(self.val_dataframe_path)
            pass
        else:

            if self.full_dataframe_path.is_file():
                cleaned_df = pd.read_csv(self.full_dataframe_path)
            else:
                indices_to_delete = []
                for i in auto.tqdm(range(len(self.df))):
                    detection_path = Path(self.output_dir) / "detections" / self.df["path"][i]
                    if not detection_path.is_file():
                        indices_to_delete += [i]
                cleaned_df = self.df.drop(indices_to_delete)
                cleaned_df.to_csv(self.full_dataframe_path)

                print(f"Kept {len(cleaned_df)}/{len(self.df)} images because the detection was missing. Dropping {len(indices_to_delete)}")

            N = len(self.df)
            indices = np.arange(N, dtype=np.int32)
            np.random.seed(seed)
            np.random.shuffle(indices)
            train_indices = indices[:int(0.9*N)]
            val_indices = indices[len(train_indices):]
            self.train_df = self.df.iloc[train_indices.tolist()]
            self.val_df = self.df.iloc[val_indices.tolist()]

            self.train_df.to_csv(self.train_dataframe_path)
            self.val_df.to_csv(self.val_dataframe_path)


    def _dataset_anaylysis(self):
        cleaned_df = pd.read_csv(self.full_dataframe_path)
        arr = cleaned_df[ActionUnitTypes.AUtype2AUstring_list(ActionUnitTypes.EMOTIONET12)].to_numpy(np.float)
        unique_au_configs, counts = np.unique(arr, return_counts=True, axis=0)
        print("There is {len(unique_au_configs)} configurations in the dataset. ")
        import matplotlib.pyplot  as plt
        plt.figure()
        plt.plot(counts)
        plt.figure()
        plt.plot(np.sort(counts))
        plt.show()


    def prepare_data(self):
        if not self.status_array_path.is_file():
            print(f"Status file does not exist. Creating '{self.status_array_path}'")
            self.status_array_path.parent.mkdir(exist_ok=True, parents=True)
            status_array = np.memmap(self.status_array_path,
                                     dtype=np.bool,
                                     mode='w+',
                                     shape=(self.num_subsets,)
                                     )
            status_array[...] = False
            del status_array

        all_processed = self.is_processed
        if not all_processed:
            self._detect_faces()

        self._split_train_val(0,0.9)

    def _new_training_set(self, for_training=True):
        if for_training:
            im_transforms_train = create_image_augmenter(self.image_size, self.augmentation)

            return EmotioNet(self.image_path, self.train_dataframe_path, self.image_size, self.scale,
                             im_transforms_train,
                             ext = self.processed_ext,
                             au_type=self.au_type,
                             )

        return EmotioNet(self.image_path, self.train_dataframe_path, self.image_size, self.scale,
                         None,
                         ext=self.processed_ext,
                             au_type=self.au_type,
                         )

    def setup(self, stage=None):
        self.training_set = self._new_training_set()
        self.validation_set = EmotioNet(self.image_path, self.val_dataframe_path, self.image_size, self.scale,
                                        None,
                                        ext=self.processed_ext,
                                        au_type=self.au_type,
                                        )

    def train_dataloader(self):
        sampler = None
        dl = DataLoader(self.training_set, shuffle=sampler is None, num_workers=self.num_workers,
                        batch_size=self.train_batch_size, drop_last=self.drop_last, sampler=sampler)
        return dl

    def val_dataloader(self):
        return DataLoader(self.validation_set, shuffle=False, num_workers=self.num_workers,
                          batch_size=self.val_batch_size, drop_last=self.drop_last)

    def test_dataloader(self):
        return DataLoader(self.test_set, shuffle=False, num_workers=self.num_workers,
                          batch_size=self.test_batch_size, drop_last=self.drop_last)

    def _get_retrieval_array(self, prefix, feature_label, dataset_size, feature_shape, feature_dtype, modifier='w+'):
        outfile_name = self._path_to_emotion_nn_retrieval_file(prefix, feature_label)
        if outfile_name.is_file() and modifier != 'r':
            raise RuntimeError(f"The retrieval array already exists! '{outfile_name}'")

        shape = tuple([dataset_size] + list(feature_shape))
        outfile_name.parent.mkdir(exist_ok=True, parents=True)
        array = np.memmap(outfile_name,
                         dtype=feature_dtype,
                         mode=modifier,
                         shape=shape
                         )
        return array


class EmotioNet(EmotionalImageDatasetBase):

    def __init__(self,
                 image_path,
                 dataframe_path,
                 image_size,
                 scale=1.4,
                 transforms=None,
                 nn_indices_array=None,
                 nn_distances_array=None,
                 ext=".jpg",
                 au_type=ActionUnitTypes.EMOTIONET12,
                 allow_missing_gt=None
                 ):
        # Initialize EmotioNet dataset
        self.dataframe_path = dataframe_path
        self.image_path = image_path
        self.df = pd.read_csv(dataframe_path)
        self.image_size = image_size
        self.transforms = transforms or imgaug.augmenters.Resize((image_size, image_size))
        self.scale = scale
        self.landmark_normalizer = KeypointNormalization()
        self.ext = ext
        self.au_type = au_type
        self.au_strs = ActionUnitTypes.AUtype2AUstring_list(self.au_type)

        # Set allow_missing_gt based on AU type
        self.allow_missing_gt = allow_missing_gt or au_type == ActionUnitTypes.EMOTIONET23
        
        # Calculate AU positive weights
        num_positive = self.df[ActionUnitTypes.AUtype2AUstring_list(au_type)].to_numpy().astype(np.float64).sum(axis=0)
        num_negative = len(self.df) - num_positive
        self.au_positive_weights = (num_negative / num_positive).astype(np.float32)

    def __len__(self):
        return len(self.df)

    def _get_sample(self, index):
        try:
            # Load image
            im_rel_path = self.df.loc[index]["path"]
            im_file = Path(self.image_path) / im_rel_path
            im_file = im_file.parent / (im_file.stem + self.ext)
            input_img = imread(im_file)
        except Exception as e:
            # Handle exception by finding next available image
            while True:
                index += 1
                index = index % len(self)
                im_rel_path = self.df.loc[index]["path"]
                im_file = Path(self.image_path) / im_rel_path
                im_file = im_file.parent / (im_file.stem + self.ext)
                try:
                    input_img = imread(im_file)
                    success = True
                except Exception as e2:
                    success = False
                if success:
                    break

        # Get AU labels
        AUs = self.df.loc[index, self.au_strs]
        AUs = np.array(AUs).astype(np.float64)

        # Check for missing AU labels
        if not self.allow_missing_gt and np.prod(np.logical_or(AUs == 1., AUs == 0.)) != 1:
            raise RuntimeError(f"It seems an AU label value in sample idx:{index}, {im_rel_path} is undefined. AUs: {AUs}")

        img = input_img

        # Load landmarks
        landmark_path = Path(self.image_path).parent / "landmarks" / im_rel_path
        landmark_path = landmark_path.parent / (landmark_path.stem + ".pkl")
        landmark_type, landmark = load_landmark(landmark_path)
        landmark = landmark[np.newaxis, ...]

        # Load segmentations
        segmentation_path = Path(self.image_path).parent / "segmentations" / im_rel_path
        segmentation_path = segmentation_path.parent / (segmentation_path.stem + ".pkl")
        seg_image, seg_type = load_segmentation(segmentation_path)
        seg_image = seg_image[np.newaxis, :, :, np.newaxis]
        seg_image = process_segmentation(seg_image, seg_type).astype(np.uint8)

        # Augment image, segmentation, and landmark
        img, seg_image, landmark = self._augment(img, seg_image, landmark)

        # Construct sample dictionary
        sample = {
            "image": numpy_image_to_torch(img.astype(np.float32)),
            "path": str(im_file),
            "label": str(im_file.stem),
            "au": AUs,
            "au_pos_weights": self.au_positive_weights,
        }

        # Add landmark and segmentation to sample if available
        if landmark is not None:
            sample["landmark"] = torch.from_numpy(landmark)
        if seg_image is not None:
            sample["mask"] = numpy_image_to_torch(seg_image)
        return sample

    def __getitem__(self, index):
        sample = self._get_sample(index)
        return sample

def sample_representative_set(dataset, output_file, sample_step=0.1, num_per_bin=2):
    # Create empty array for binning
    va_array = []
    size = int(2 / sample_step)
    for i in range(size):
        va_array += [[]]
        for j in range(size):
            va_array[i] += [[]]

    print("Binning dataset")
    # Bin dataset based on valence and arousal
    for i in tqdm(range(len(dataset.df))):
        v = max(-1., min(1., dataset.df.loc[i]["valence"]))
        a = max(-1., min(1., dataset.df.loc[i]["arousal"]))
        row_ = int((v + 1) / sample_step)
        col_ = int((a + 1) / sample_step)
        va_array[row_][ col_] += [i]

    # Select representative samples from each bin
    selected_indices = []
    for i in range(len(va_array)):
        for j in range(len(va_array[i])):
            if len(va_array[i][j]) > 0:
                selected_indices += va_array[i][j][0:num_per_bin]
            else:
                print(f"No value for {i} and {j}")

    # Save selected samples to output file
    selected_samples = dataset.df.loc[selected_indices]
    selected_samples.to_csv(output_file)
    print(f"Selected samples saved to '{output_file}'")

if __name__ == "__main__":
    # Load augmenter from YAML file
    augmenter = yaml.load(open(Path(__file__).parents[2] / "gdl_apps" / "EmotionRecognition" / "emodeca_conf" / "data" / "augmentations" / "default_with_resize.yaml"))["augmentation"]

    # Initialize EmotionsData
    dm = EmotionsData(
             "/home/abbas/dream/EmotionalFacialAnimation/data/emotionnet/emotioNet_challenge_files_server_challenge_1.2_aws_downloaded/",
             "/home/abbas/data/emotionet/",
             processed_subfolder="processed_2021_Aug_31_21-33-44",
             scale=1.7,
             ignore_invalid=True,
             image_size=224,
             bb_center_shift_x=0,
             bb_center_shift_y=-0.3,
            augmentation=augmenter,
            )
    print(dm.num_subsets)
    dm.prepare_data()
    dm.setup()
