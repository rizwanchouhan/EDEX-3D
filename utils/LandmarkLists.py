from mediapipe.python.solutions.face_mesh_connections import (
    FACEMESH_CONTOURS,
    FACEMESH_FACE_OVAL,
    FACEMESH_IRISES,
    FACEMESH_LEFT_EYE,
    FACEMESH_LEFT_EYEBROW,
    FACEMESH_LEFT_IRIS,
    FACEMESH_LIPS,
    FACEMESH_RIGHT_EYE,
    FACEMESH_RIGHT_EYEBROW,
    FACEMESH_RIGHT_IRIS,
    FACEMESH_TESSELATION,
)
import numpy as np


def unpack_mediapipe_set(edge_set):
    """
    Unpacks a set of edges to obtain the set of vertices involved.
    Args:
        edge_set: Set of edges.

    Returns:
        Set of vertices.
    """
    vertex_set = set()
    for i in edge_set:
        vertex_set.add(i[0])
        vertex_set.add(i[1])
    return vertex_set


def left_eye_eyebrow_landmark_indices(sorted=True):
    """
    Returns the landmark indices for the left eye and eyebrow regions.
    Args:
        sorted: Boolean indicating whether to sort the indices.

    Returns:
        Array of landmark indices.
    """
    left_eye = list(
        unpack_mediapipe_set(FACEMESH_LEFT_EYE)
        .union(unpack_mediapipe_set(FACEMESH_LEFT_IRIS))
        .union(unpack_mediapipe_set(FACEMESH_LEFT_EYEBROW))
    )
    if sorted:
        left_eye.sort()
    left_eye = np.array(left_eye, dtype=np.int32)
    return left_eye


def right_eye_eyebrow_landmark_indices(sorted=True):
    """
    Returns the landmark indices for the right eye and eyebrow regions.
    Args:
        sorted: Boolean indicating whether to sort the indices.

    Returns:
        Array of landmark indices.
    """
    right_eye = list(
        unpack_mediapipe_set(FACEMESH_RIGHT_EYE)
        .union(unpack_mediapipe_set(FACEMESH_RIGHT_IRIS))
        .union(unpack_mediapipe_set(FACEMESH_RIGHT_EYEBROW))
    )
    if sorted:
        right_eye.sort()
    right_eye = np.array(right_eye, dtype=np.int32)
    return right_eye


def left_eye_landmark_indices(sorted=True):
    """
    Returns the landmark indices for the left eye region.
    Args:
        sorted: Boolean indicating whether to sort the indices.

    Returns:
        Array of landmark indices.
    """
    left_eye = list(unpack_mediapipe_set(FACEMESH_LEFT_EYE))
    if sorted:
        left_eye.sort()
    left_eye = np.array(left_eye, dtype=np.int32)
    return left_eye


def right_eye_landmark_indices(sorted=True):
    """
    Returns the landmark indices for the right eye region.
    Args:
        sorted: Boolean indicating whether to sort the indices.

    Returns:
        Array of landmark indices.
    """
    right_eye = list(unpack_mediapipe_set(FACEMESH_RIGHT_EYE))
    if sorted:
        right_eye.sort()
    right_eye = np.array(right_eye, dtype=np.int32)
    return right_eye


def mouth_landmark_indices(sorted=True):
    """
    Returns the landmark indices for the mouth region.
    Args:
        sorted: Boolean indicating whether to sort the indices.

    Returns:
        Array of landmark indices.
    """
    mouth = list(unpack_mediapipe_set(FACEMESH_LIPS))
    if sorted:
        mouth.sort()
    mouth = np.array(mouth, dtype=np.int32)
    return mouth


def face_oval_landmark_indices(sorted=True):
    """
    Returns the landmark indices for the face oval region.
    Args:
        sorted: Boolean indicating whether to sort the indices.

    Returns:
        Array of landmark indices.
    """
    face_oval = list(unpack_mediapipe_set(FACEMESH_FACE_OVAL))
    if sorted:
        face_oval.sort()
    face_oval = np.array(face_oval, dtype=np.int32)
    return face_oval


def all_face_landmark_indices(sorted=True):
    """
    Returns the landmark indices for all face regions.
    Args:
        sorted: Boolean indicating whether to sort the indices.

    Returns:
        Array of landmark indices.
    """
    face_all = list(unpack_mediapipe_set(FACEMESH_TESSELATION))
    if sorted:
        face_all.sort()
    face_all = np.array(face_all, dtype=np.int32)
    return face_all
