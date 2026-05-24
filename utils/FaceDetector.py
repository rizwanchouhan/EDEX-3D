from abc import abstractmethod, ABC
import numpy as np
import torch
import pickle as pkl
from face_alignment.utils import flip, get_preds_fromhm

# Function to save landmark data to a file
def save_landmark(fname, landmark, landmark_type):
    with open(fname, "wb") as f:
        pkl.dump(landmark_type, f)
        pkl.dump(landmark, f)

# Function to load landmark data from a file
def load_landmark(fname):
    with open(fname, "rb") as f:
        landmark_type = pkl.load(f)
        landmark = pkl.load(f)
    return landmark_type, landmark

# Function to save extended landmark data to a file
def save_landmark_v2(fname, landmark, landmark_confidence, landmark_type):
    with open(fname, "wb") as f:
        pkl.dump(landmark_type, f)
        pkl.dump(landmark_confidence, f)
        pkl.dump(landmark, f)

# Function to load extended landmark data from a file
def load_landmark_v2(fname):
    with open(fname, "rb") as f:
        landmark_type = pkl.load(f)
        landmark_confidence = pkl.load(f)
        landmark = pkl.load(f)
    return landmark_type, landmark_confidence, landmark

# Abstract class for face detectors
class FaceDetector(ABC):

    @abstractmethod
    def run(self, image, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        self.run(*args, **kwargs)

    # Abstract method to get landmarks from a batch of images without face detection
    @abstractmethod
    def landmarks_from_batch_no_face_detection(self, images):
        raise NotImplementedError()

    # Abstract method to return the optimal image size for the landmark detector
    @abstractmethod
    def optimal_landmark_detector_im_size(self):
        raise NotImplementedError()

    # Abstract method to return the type of landmarks
    @abstractmethod
    def landmark_type(self):
        raise NotImplementedError()

# Face Alignment Network (FAN) class
class FAN(FaceDetector):

    def __init__(self, device='cuda', threshold=0.5):
        import face_alignment
        self.face_detector = 'sfd'
        self.face_detector_kwargs = {
            "filter_threshold": threshold
        }
        self.flip_input = False
        self.model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,
                                                  device=str(device),
                                                  flip_input=self.flip_input,
                                                  face_detector=self.face_detector,
                                                  face_detector_kwargs=self.face_detector_kwargs)

    # Method to run FAN on an image
    def run(self, image, with_landmarks=False, detected_faces=None):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box list
        '''
        out = self.model.get_landmarks(image, detected_faces=detected_faces)
        torch.cuda.empty_cache()
        if out is None:
            del out
            if with_landmarks:
                return [], 'kpt68', []
            else:
                return [], 'kpt68'
        else:
            boxes = []
            kpts = []
            for i in range(len(out)):
                kpt = out[i].squeeze()
                left = np.min(kpt[:, 0])
                right = np.max(kpt[:, 0])
                top = np.min(kpt[:, 1])
                bottom = np.max(kpt[:, 1])
                bbox = [left, top, right, bottom]
                boxes += [bbox]
                kpts += [kpt]
            del out
            if with_landmarks:
                return boxes, 'kpt68', kpts
            else:
                return boxes, 'kpt68'

    # Method to get landmarks from a batch of images without face detection using FAN
    @torch.no_grad()
    def landmarks_from_batch_no_face_detection(self, images):
        out = self.model.face_alignment_net(images).detach()
        if self.flip_input:
            out += flip(self.model.face_alignment_net(flip(images)).detach(), is_label=True)

        out = out.cpu().numpy()
        center = None
        scale = None
        B = out.shape[0]
        pts, pts_img, scores = get_preds_fromhm(out, center, scale)
        pts, pts_img = torch.from_numpy(pts), torch.from_numpy(pts_img)
        pts, pts_img = pts.view(B, 68, 2) * 4, pts_img.view(B, 68, 2)
        scores = scores
        pts /= images.shape[-1]
        pts = pts.cpu().numpy()
        pts_img = pts_img.cpu().numpy()
        return pts, scores

    # Method to return the optimal image size for the landmark detector
    def optimal_landmark_detector_im_size(self):
        return 256

    # Method to return the type of landmarks
    def landmark_type(self):
        return 'kpt68'

# Multi-Task Cascaded Convolutional Neural Network (MTCNN) class
class MTCNN(FaceDetector):

    def __init__(self, device='cuda'):
        '''
        https://github.com/timesler/facenet-pytorch/blob/master/examples/infer.ipynb
        '''
        from facenet_pytorch import MTCNN as mtcnn
        self.device = device
        self.model = mtcnn(keep_all=True, device=device)

    # Method to run MTCNN on an image
    def run(self, input, **kwargs):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box
        '''
        out = self.model.detect(input[None, ...])
        if out[0][0] is None:
            return [], 'bbox'
        else:
            bboxes = []
            for i in range(out.shape[0]):
                bbox = out[0][0].squeeze()
                bboxes += [bbox]
            return bboxes, 'bbox'

    # Method to return the type of landmarks (bounding box)
    def landmark_type(self):
        return 'bbox'
