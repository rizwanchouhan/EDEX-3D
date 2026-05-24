from affectnet.FaceVideoModule import FaceVideoModule
from affectnet.AffectNetModule import AffectNetExpressions
from affectnet.EmotionsDataset import EmotionsDataset
from enum import Enum
import pickle as pkl
from pathlib import Path
import hashlib
from tqdm import auto
import numpy as np
import PIL
from collections import OrderedDict


from enum import Enum
from collections import OrderedDict
import numpy as np
import pickle as pkl
import hashlib
import PIL
from PIL import ImageDraw

class Expression7(Enum):
    """Enum class representing expressions."""
    Neutral = 0
    Anger = 1
    Disgust = 2
    Fear = 3
    Happiness = 4
    Sadness = 5
    Surprise = 6
    None_ = 7  # Consider renaming this to 'NoneType' to avoid confusion with 'None'.

def affect_net_to_expr7(aff: AffectNetExpressions) -> Expression7:
    """Converts AffectNetExpressions to Expression7."""
    if aff == AffectNetExpressions.Happy:
        return Expression7.Happiness
    if aff == AffectNetExpressions.Sad:
        return Expression7.Sadness
    if aff == AffectNetExpressions.Contempt:
        return Expression7.None_
    return Expression7[aff.name]

def expr7_to_affect_net(expr: Expression7) -> AffectNetExpressions:
    """Converts Expression7 to AffectNetExpressions."""
    if isinstance(expr, int) or isinstance(expr, np.int32) or isinstance(expr, np.int64):
        expr = Expression7(expr)
    if expr == Expression7.Happiness:
        return AffectNetExpressions.Happy
    if expr == Expression7.Sadness:
        return AffectNetExpressions.Sad
    return AffectNetExpressions[expr.name]

class AU8(Enum):
    """Enum class representing Action Units."""
    AU1 = 0
    AU2 = 1
    AU4 = 2
    AU6 = 3
    AU12 = 4
    AU15 = 5
    AU20 = 6
    AU25 = 7

class AffWild2DMBase(FaceVideoModule):
    """Class representing AffWild2DMBase."""

    def _get_processed_annotations_for_sequence(self, sid):
        """Get processed annotations for a sequence."""
        pass  # You may want to implement this method.

    def _create_emotional_image_dataset(self, annotation_list=None, filter_pattern=None,
                                        with_landmarks=False, with_segmentation=False,
                                        crash_on_missing_file=False):
        """Create an emotional image dataset."""
        annotation_list = annotation_list or ['va', 'expr7', 'au8']
        detections_all = []
        annotations_all = OrderedDict()
        for a in annotation_list:
            annotations_all[a] = []
        recognition_labels_all = []

        import re
        if filter_pattern is not None:
            p = re.compile(filter_pattern, re.IGNORECASE)

        for si in auto.tqdm(range(self.num_sequences)):
            sequence_name = self.video_list[si]

            if filter_pattern is not None:
                res = p.match(str(sequence_name))
                if res is None:
                    continue

            detection_fnames, annotations, recognition_labels, discarded_annotations, detection_not_found = \
                self._get_validated_annotations_for_sequence(si, crash_on_failure=False)

            if detection_fnames is None:
                continue

            current_list = annotation_list.copy()
            for annotation_name, detection_list in detection_fnames.items():
                detections_all += detection_list
                for annotation_key in annotations[annotation_name].keys():
                    if annotation_key in current_list:
                        current_list.remove(annotation_key)
                    array = annotations[annotation_name][annotation_key]
                    annotations_all[annotation_key] += array.tolist()
                    n = array.shape[0]

                recognition_labels_all += len(detection_list)*[annotation_name + "_" + str(recognition_labels[annotation_name])]
                if len(current_list) != len(annotation_list):
                    print("No desired GT is found. Skipping sequence %d" % si)

                for annotation_name in current_list:
                    annotations_all[annotation_name] += [None] * n

        print("Data gathered")
        print(f"Found {len(detections_all)} detections with annotations "
              f"of {len(set(recognition_labels_all))} identities")

        invalid_indices = set()
        if not with_landmarks:
            landmarks = None
        else:
            landmarks = []
            print("Checking if every frame has a corresponding landmark file")
            for det_i, det in enumerate(auto.tqdm(detections_all)):
                lmk = det.parents[3]
                lmk = lmk / "landmarks" / (det.relative_to(lmk / "detections"))
                lmk = lmk.parent / (lmk.stem + ".pkl")
                file_exists = (self.output_dir / lmk).is_file()
                if not file_exists and crash_on_missing_file:
                    raise RuntimeError(f"Landmark does not exist {lmk}")
                elif not file_exists:
                    invalid_indices.add(det_i)
                landmarks += [lmk]

        if not with_segmentation:
            segmentations = None
        else:
            segmentations = []
            print("Checking if every frame has a corresponding segmentation file")
            for det_i, det in enumerate(auto.tqdm(detections_all)):
                seg = det.parents[3]
                seg = seg / "segmentations" / (det.relative_to(seg / "detections"))
                seg = seg.parent / (seg.stem + ".pkl")
                file_exists = (self.output_dir / seg).is_file()
                if not file_exists and crash_on_missing_file:
                    raise RuntimeError(f"Landmark does not exist {seg}")
                elif not file_exists:
                    invalid_indices.add(det_i)
                segmentations += [seg]

        invalid_indices = sorted(list(invalid_indices), reverse=True)
        for idx in invalid_indices:
            del detections_all[idx]
            del landmarks[idx]
            del segmentations[idx]
            del recognition_labels_all[idx]
            for key in annotations_all.keys():
                del annotations_all[key][idx]

        return detections_all, landmarks, segmentations, annotations_all, recognition_labels_all

    def get_annotated_emotion_dataset(self, annotation_list=None, filter_pattern=None,
                                      image_transforms=None, split_ratio=None,
                                      split_style=None, with_landmarks=False,
                                      with_segmentations=False, K=None, K_policy=None,
                                      load_from_cache=True):
        """Get annotated emotion dataset."""
        str_to_hash = pkl.dumps(tuple([annotation_list, filter_pattern]))
        inter_cache_hash = hashlib.md5(str_to_hash).hexdigest()
        inter_cache_folder = Path(self.output_dir) / "cache" / str(inter_cache_hash)
        if (inter_cache_folder / "lists.pkl").exists() and load_from_cache:
            print(f"Found processed filelists in '{str(inter_cache_folder)}'. "
                  f"Reprocessing will not be needed. Loading ...")
            with open(inter_cache_folder / "lists.pkl", "rb") as f:
                detections = pkl.load(f)
                landmarks = pkl.load(f)
                segmentations = pkl.load(f)
                annotations = pkl.load(f)
                recognition_labels = pkl.load(f)
            print("Loading done")

        else:
            detections, landmarks, segmentations, annotations, recognition_labels = \
                self._create_emotional_image_dataset(
                    annotation_list, filter_pattern, with_landmarks, with_segmentations)
            inter_cache_folder.mkdir(exist_ok=True, parents=True)
            print(f"Dataset processed. Saving into: '{str(inter_cache_folder)}'.")
            with open(inter_cache_folder / "lists.pkl", "wb") as f:
                pkl.dump(detections, f)
                pkl.dump(landmarks, f)
                pkl.dump(segmentations, f)
                pkl.dump(annotations, f)
                pkl.dump(recognition_labels, f)
            print(f"Saving done.")

        if split_ratio is not None and split_style is not None:

            hash_list = tuple([annotation_list,
                               filter_pattern,
                               split_ratio,
                               split_style,
                               with_landmarks,
                               with_segmentations,
                               K,
                               K_policy,
                               ])
            cache_hash = hashlib.md5(pkl.dumps(hash_list)).hexdigest()
            cache_folder = Path(self.output_dir) / "cache" / "tmp" / str(cache_hash)
            cache_folder.mkdir(exist_ok=True, parents=True)
            if load_from_cache and (cache_folder / "lists_train.pkl").is_file() and \
                (cache_folder / "lists_val.pkl").is_file():
                print(f"Dataset split found in: '{str(cache_folder)}'. Loading ...")
                with open(cache_folder / "lists_train.pkl", "rb") as f:
                     detection_train = pkl.load(f)
                     landmarks_train = pkl.load(f)
                     segmentations_train = pkl.load(f)
                     annotations_train = pkl.load(f)
                     recognition_labels_train = pkl.load(f)
                     idx_train = pkl.load(f)
                with open(cache_folder / "lists_val.pkl", "rb") as f:
                     detection_val = pkl.load(f)
                     landmarks_val = pkl.load(f)
                     segmentations_val = pkl.load(f)
                     annotations_val = pkl.load(f)
                     recognition_labels_val = pkl.load(f)
                     idx_val = pkl.load(f)
                print("Loading done")
            else:
                print(f"Splitting the dataset. Split style '{split_style}', split ratio: '{split_ratio}'")
                if image_transforms is not None:
                    if not isinstance(image_transforms, list) or len(image_transforms) != 2:
                        raise ValueError("You have to provide image transforms for both trainng and validation sets")
                idxs = np.arange(len(detections), dtype=np.int32)
                if split_style == 'random':
                    np.random.seed(0)
                    np.random.shuffle(idxs)
                    split_idx = int(idxs.size * split_ratio)
                    idx_train = idxs[:split_idx]
                    idx_val = idxs[split_idx:]
                elif split_style == 'manual':
                    idx_train = []
                    idx_val = []
                    for i, det in enumerate(auto.tqdm(detections)):
                        if 'Train_Set' in str(det):
                            idx_train += [i]
                        elif 'Validation_Set' in str(det):
                            idx_val += [i]
                        else:
                            idx_val += [i]

                elif split_style == 'sequential':
                    split_idx = int(idxs.size * split_ratio)
                    idx_train = idxs[:split_idx]
                    idx_val = idxs[split_idx:]
                elif split_style == 'random_by_label':
                    idx_train = []
                    idx_val = []
                    unique_labels = sorted(list(set(recognition_labels)))
                    np.random.seed(0)
                    print(f"Going through {len(unique_labels)} unique labels and splitting its samples into "
                          f"training/validations set randomly.")
                    for li, label in enumerate(auto.tqdm(unique_labels)):
                        label_indices = np.array([i for i in range(len(recognition_labels)) if recognition_labels[i] == label],
                                                 dtype=np.int32)
                        np.random.shuffle(label_indices)
                        split_idx = int(len(label_indices) * split_ratio)
                        i_train = label_indices[:split_idx]
                        i_val = label_indices[split_idx:]
                        idx_train += i_train.tolist()
                        idx_val += i_val.tolist()
                    idx_train = np.array(idx_train, dtype= np.int32)
                    idx_val = np.array(idx_val, dtype= np.int32)
                elif split_style == 'sequential_by_label':
                    idx_train = []
                    idx_val = []
                    unique_labels = sorted(list(set(recognition_labels)))
                    print(f"Going through {len(unique_labels)} unique labels and splitting its samples into "
                          f"training/validations set sequentially.")
                    for li, label in enumerate(auto.tqdm(unique_labels)):
                        label_indices = [i for i in range(len(recognition_labels)) if recognition_labels[i] == label]
                        split_idx = int(len(label_indices) * split_ratio)
                        i_train = label_indices[:split_idx]
                        i_val = label_indices[split_idx:]
                        idx_train += i_train
                        idx_val += i_val
                    idx_train = np.array(idx_train, dtype= np.int32)
                    idx_val = np.array(idx_val, dtype= np.int32)
                else:
                    raise ValueError(f"Invalid split style {split_style}")

                if split_ratio < 0 or split_ratio > 1:
                    raise ValueError(f"Invalid split ratio {split_ratio}")

                def index_list_by_list(l, idxs):
                    return [l[i] for i in idxs]

                def index_dict_by_list(d, idxs):
                    res = d.__class__()
                    for key in d.keys():
                        res[key] = [d[key][i] for i in idxs]
                    return res

                detection_train = index_list_by_list(detections, idx_train)
                annotations_train = index_dict_by_list(annotations, idx_train)
                recognition_labels_train = index_list_by_list(recognition_labels, idx_train)
                if with_landmarks:
                    landmarks_train = index_list_by_list(landmarks, idx_train)
                else:
                    landmarks_train = None

                if with_segmentations:
                    segmentations_train = index_list_by_list(segmentations, idx_train)
                else:
                    segmentations_train = None

                detection_val = index_list_by_list(detections, idx_val)
                annotations_val = index_dict_by_list(annotations, idx_val)
                recognition_labels_val = index_list_by_list(recognition_labels, idx_val)

                if with_landmarks:
                    landmarks_val = index_list_by_list(landmarks, idx_val)
                else:
                    landmarks_val = None

                if with_segmentations:
                    segmentations_val = index_list_by_list(segmentations, idx_val)
                else:
                    segmentations_val = None

                print(f"Dataset split processed. Saving into: '{str(cache_folder)}'.")
                with open(cache_folder / "lists_train.pkl", "wb") as f:
                    pkl.dump(detection_train, f)
                    pkl.dump(landmarks_train, f)
                    pkl.dump(segmentations_train, f)
                    pkl.dump(annotations_train, f)
                    pkl.dump(recognition_labels_train, f)
                    pkl.dump(idx_train, f)
                with open(cache_folder / "lists_val.pkl", "wb") as f:
                    pkl.dump(detection_val, f)
                    pkl.dump(landmarks_val, f)
                    pkl.dump(segmentations_val, f)
                    pkl.dump(annotations_val, f)
                    pkl.dump(recognition_labels_val, f)
                    pkl.dump(idx_val, f)
                print(f"Saving done.")

            dataset_train = EmotionsDataset(
                detection_train,
                annotations_train,
                recognition_labels_train,
                image_transforms[0],
                self.output_dir,
                landmark_list=landmarks_train,
                segmentation_list=segmentations_train,
                K=K,
                K_policy=K_policy)

            dataset_val = EmotionsDataset(
                detection_val,
                annotations_val,
                recognition_labels_val,
                image_transforms[1],
                self.output_dir,
                landmark_list=landmarks_val,
                segmentation_list=segmentations_val,
                K=1,
                K_policy='sequential')

            return dataset_train, dataset_val, idx_train, idx_val

        dataset = EmotionsDataset(
            detections,
            annotations,
            recognition_labels,
            image_transforms,
            self.output_dir,
            landmark_list=landmarks,
            segmentation_list=segmentations,
            K=K,
            K_policy=K_policy)
        return dataset

    def _draw_annotation(self, frame_draw: PIL.ImageDraw.Draw, val_gt: dict, font, color):
        """Draw annotations on the frame."""
        all_str = ''
        for gt_type, val in val_gt.items():
            if gt_type == 'va':
                va_str = "V: %.02f  A: %.02f" % (val[0], val[1])
                all_str += "\n" + va_str
            elif gt_type == 'expr7':
                frame_draw.text((bb[0, 0], bb[0, 1] - 30,), Expression7(val).name, font=font, fill=color)
            elif gt_type == 'au8':
                au_str = ''
                for li, label in enumerate(val):
                    if label:
                        au_str += AU8(li).name + ' '
                all_str += "\n" + au_str
            else:
                raise ValueError(f"Unable to visualize this gt_type: '{gt_type}")
            frame_draw.text((bb[0, 0], bb[1, 1] + 10,), str(all_str), font=font, fill=color)


    def test_annotations(self, net=None, annotation_list=None, filter_pattern=None):
        """Test annotations."""
        net = net or self._get_emonet(self.device)

        dataset = self.get_annotated_emotion_dataset(annotation_list, filter_pattern)


def main():
    # Define root directory and paths
    root = Path("/home/abbas/dream/EmotionalFacialAnimation/data/aff-wild2/")
    root_path = root / "Aff-Wild2_ready"
    output_path = root / "processed"
    subfolder = 'processed_2021_Jan_19_20-25-10'
    
    # Initialize AffWild2DMBase object
    dm = AffWild2DMBase(str(root_path), str(output_path), processed_subfolder=subfolder)
    
    # Prepare data
    dm.prepare_data()
    
    # Define frame index
    fj = 9
    
    # Define retargeting source (optional)
    retarget_from = None
    
    # Reconstruct faces in sequence using DREAM method
    dm._reconstruct_faces_in_sequence(fj, rec_method="dream", retarget_from=retarget_from, retarget_suffix="_retarget_cena")
    
    # Create reconstruction videos with various retargeting suffixes
    dm.create_reconstruction_video(fj, overwrite=True, rec_method='dream', retarget_suffix="_retarget_soubhik")
    dm.create_reconstruction_video(fj, overwrite=True, rec_method='dream', retarget_suffix="_retarget_obama")
    dm.create_reconstruction_video(fj, overwrite=True, rec_method='dream', retarget_suffix="_retarget_cumberbatch")
    dm.create_reconstruction_video(fj, overwrite=True, rec_method='dream', retarget_suffix="_retarget_cena")
    
    # Print completion message
    print("Peace out")


if __name__ == "__main__":
    main()