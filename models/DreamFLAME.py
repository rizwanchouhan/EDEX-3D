import torch
import torch.nn as nn
import numpy as np
import pickle
import torch.nn.functional as F

from utils.lbs import lbs, batch_rodrigues, vertices2landmarks


def to_tensor(array, dtype=torch.float32):
    """
    Converts an array-like object to a PyTorch tensor.
    Args:
        array: Array-like object to convert.
        dtype (torch.dtype): Data type of the tensor.

    Returns:
        torch.Tensor: Converted tensor.
    """
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)


def to_np(array, dtype=np.float32):
    """
    Converts a tensor to a NumPy array.
    Args:
        array (torch.Tensor): Tensor to convert.
        dtype (np.dtype): Data type of the NumPy array.

    Returns:
        np.ndarray: Converted NumPy array.
    """
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


class Struct(object):
    """
    Simple class to create objects with attributes from dictionaries.
    """

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def rot_mat_to_euler(rot_mats):
    """
    Converts rotation matrices to Euler angles.
    Args:
        rot_mats (torch.Tensor): Rotation matrices.

    Returns:
        torch.Tensor: Euler angles.
    """
    sy = torch.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] +
                    rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
    return torch.atan2(-rot_mats[:, 2, 0], sy)


class FLAME(nn.Module):
    """
    Differentiable FLAME function to generate mesh and facial landmarks.
    """

    def __init__(self, config):
        """
        Initializes the FLAME model.
        Args:
            config: Configuration object.
        """
        super(FLAME, self).__init__()
        print("creating the FLAME Decoder")
        with open('/home/abbas/dream/assets/FLAME/geometry/generic_model.pkl', 'rb') as f:
            ss = pickle.load(f, encoding='latin1')
            flame_model = Struct(**ss)

        self.cfg = config
        self.dtype = torch.float32
        # Registering buffers for constant parameters
        self.register_buffer('faces_tensor', to_tensor(to_np(flame_model.f, dtype=np.int64), dtype=torch.long))
        self.register_buffer('v_template', to_tensor(to_np(flame_model.v_template), dtype=self.dtype))
        shapedirs = to_tensor(to_np(flame_model.shapedirs), dtype=self.dtype)
        shapedirs = torch.cat([shapedirs[:, :, :config.n_shape], shapedirs[:, :, 300:300 + config.n_exp]], 2)
        self.register_buffer('shapedirs', shapedirs)
        num_pose_basis = flame_model.posedirs.shape[-1]
        posedirs = np.reshape(flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', to_tensor(to_np(posedirs), dtype=self.dtype))
        self.register_buffer('J_regressor', to_tensor(to_np(flame_model.J_regressor), dtype=self.dtype))
        parents = to_tensor(to_np(flame_model.kintree_table[0])).long();
        parents[0] = -1
        self.register_buffer('parents', parents)
        self.register_buffer('lbs_weights', to_tensor(to_np(flame_model.weights), dtype=self.dtype))
        default_eyball_pose = torch.zeros([1, 6], dtype=self.dtype, requires_grad=False)
        self.register_parameter('eye_pose', nn.Parameter(default_eyball_pose,
                                                         requires_grad=False))
        default_neck_pose = torch.zeros([1, 3], dtype=self.dtype, requires_grad=False)
        self.register_parameter('neck_pose', nn.Parameter(default_neck_pose,
                                                          requires_grad=False))
        # Loading and registering landmark embeddings
        lmk_embeddings = np.load('/home/abbas/dream/assets/FLAME/geometry/landmark_embedding.npy',
                                 allow_pickle=True, encoding='latin1')
        lmk_embeddings = lmk_embeddings[()]
        self.register_buffer('lmk_faces_idx', torch.tensor(lmk_embeddings['static_lmk_faces_idx'], dtype=torch.long))
        self.register_buffer('lmk_bary_coords',
                             torch.tensor(lmk_embeddings['static_lmk_bary_coords'], dtype=self.dtype))
        self.register_buffer('dynamic_lmk_faces_idx',
                             torch.tensor(lmk_embeddings['dynamic_lmk_faces_idx'], dtype=torch.long))
        self.register_buffer('dynamic_lmk_bary_coords',
                             torch.tensor(lmk_embeddings['dynamic_lmk_bary_coords'], dtype=self.dtype))
        self.register_buffer('full_lmk_faces_idx',
                             torch.tensor(lmk_embeddings['full_lmk_faces_idx'], dtype=torch.long))
        self.register_buffer('full_lmk_bary_coords',
                             torch.tensor(lmk_embeddings['full_lmk_bary_coords'], dtype=self.dtype))

        # Finding neck kinematic chain
        neck_kin_chain = [];
        NECK_IDX = 1
        curr_idx = torch.tensor(NECK_IDX, dtype=torch.long)
        while curr_idx != -1:
            neck_kin_chain.append(curr_idx)
            curr_idx = self.parents[curr_idx]
        self.register_buffer('neck_kin_chain', torch.stack(neck_kin_chain))

    def _find_dynamic_lmk_idx_and_bcoords(self, pose, dynamic_lmk_faces_idx,
                                          dynamic_lmk_b_coords,
                                          neck_kin_chain, dtype=torch.float32):
        """
        Finds dynamic landmark indices and barycentric coordinates.
        Args:
            pose (torch.Tensor): Pose tensor.
            dynamic_lmk_faces_idx (torch.Tensor): Dynamic landmark faces indices.
            dynamic_lmk_b_coords (torch.Tensor): Dynamic landmark barycentric coordinates.
            neck_kin_chain (torch.Tensor): Neck kinematic chain.
            dtype (torch.dtype): Data type of tensors.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Dynamic landmark indices and barycentric coordinates.
        """
        batch_size = pose.shape[0]

        aa_pose = torch.index_select(pose.view(batch_size, -1, 3), 1,
                                     neck_kin_chain)
        rot_mats = batch_rodrigues(
            aa_pose.view(-1, 3), dtype=dtype).view(batch_size, -1, 3, 3)

        rel_rot_mat = torch.eye(3, device=pose.device,
                                dtype=dtype).unsqueeze_(dim=0).expand(batch_size, -1, -1)
        for idx in range(len(neck_kin_chain)):
            rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)

        y_rot_angle = torch.round(
            torch.clamp(rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi,
                        max=39)).to(dtype=torch.long)

        neg_mask = y_rot_angle.lt(0).to(dtype=torch.long)
        mask = y_rot_angle.lt(-39).to(dtype=torch.long)
        neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
        y_rot_angle = (neg_mask * neg_vals +
                       (1 - neg_mask) * y_rot_angle)

        dyn_lmk_faces_idx = torch.index_select(dynamic_lmk_faces_idx,
                                               0, y_rot_angle)
        dyn_lmk_b_coords = torch.index_select(dynamic_lmk_b_coords,
                                              0, y_rot_angle)
        return dyn_lmk_faces_idx, dyn_lmk_b_coords

    def _vertices2landmarks(self, vertices, faces, lmk_faces_idx, lmk_bary_coords):
        """
        Converts vertices to landmarks.
        Args:
            vertices (torch.Tensor): Vertices tensor.
            faces (torch.Tensor): Faces tensor.
            lmk_faces_idx (torch.Tensor): Landmark faces indices.
            lmk_bary_coords (torch.Tensor): Landmark barycentric coordinates.

        Returns:
            torch.Tensor: Landmarks.
        """
        batch_size, num_verts = vertices.shape[:2]
        lmk_faces = torch.index_select(faces, 0, lmk_faces_idx.view(-1)).view(
            1, -1, 3).view(batch_size, lmk_faces_idx.shape[1], -1)

        lmk_faces += torch.arange(batch_size, dtype=torch.long).view(-1, 1, 1).to(
            device=vertices.device) * num_verts

        lmk_vertices = vertices.view(-1, 3)[lmk_faces]
        landmarks = torch.einsum('blfi,blf->bli', [lmk_vertices, lmk_bary_coords])
        return landmarks

    def _vertices2landmarks2d(self, vertices, full_pose):
        """
        Converts vertices to 2D landmarks.
        Args:
            vertices (torch.Tensor): Vertices tensor.
            full_pose (torch.Tensor): Full pose tensor.

        Returns:
            torch.Tensor: 2D landmarks.
        """
        batch_size = vertices.shape[0]
        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(dim=0).expand(batch_size, -1)
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).expand(batch_size, -1, -1)

        dyn_lmk_faces_idx, dyn_lmk_bary_coords = self._find_dynamic_lmk_idx_and_bcoords(
            full_pose, self.dynamic_lmk_faces_idx,
            self.dynamic_lmk_bary_coords,
            self.neck_kin_chain, dtype=self.dtype)
        lmk_faces_idx = torch.cat([dyn_lmk_faces_idx, lmk_faces_idx], 1)
        lmk_bary_coords = torch.cat([dyn_lmk_bary_coords, lmk_bary_coords], 1)

        landmarks2d = vertices2landmarks(vertices, self.faces_tensor,
                                         lmk_faces_idx,
                                         lmk_bary_coords)
        return landmarks2d

    def seletec_3d68(self, vertices):
        """
        Selects 3D landmarks.
        Args:
            vertices (torch.Tensor): Vertices tensor.

        Returns:
            torch.Tensor: 3D landmarks.
        """
        landmarks3d = vertices2landmarks(vertices, self.faces_tensor,
                                         self.full_lmk_faces_idx.repeat(vertices.shape[0], 1),
                                         self.full_lmk_bary_coords.repeat(vertices.shape[0], 1, 1))
        return landmarks3d

    def forward(self, shape_params=None, expression_params=None, pose_params=None, eye_pose_params=None):
        """
        Forward pass of the FLAME model.
        Args:
            shape_params (torch.Tensor): Shape parameters.
            expression_params (torch.Tensor): Expression parameters.
            pose_params (torch.Tensor): Pose parameters.
            eye_pose_params (torch.Tensor): Eye pose parameters.

        Returns:
            Tuple[torch.Tensor]: Vertices, 2D landmarks, 3D landmarks.
        """
        batch_size = shape_params.shape[0]
        if pose_params is None:
            pose_params = self.eye_pose.expand(batch_size, -1)
        if eye_pose_params is None:
            eye_pose_params = self.eye_pose.expand(batch_size, -1)
        if expression_params is None:
            expression_params = torch.zeros(batch_size, self.cfg.n_exp).to(shape_params.device)

        betas = torch.cat([shape_params, expression_params], dim=1)
        full_pose = torch.cat(
            [pose_params[:, :3], self.neck_pose.expand(batch_size, -1), pose_params[:, 3:], eye_pose_params], dim=1)
        template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)

        vertices, _ = lbs(betas, full_pose, template_vertices,
                          self.shapedirs, self.posedirs,
                          self.J_regressor, self.parents,
                          self.lbs_weights, dtype=self.dtype,
                          detach_pose_correctives=False)

        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(dim=0).expand(batch_size, -1)
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).expand(batch_size, -1, -1)

        dyn_lmk_faces_idx, dyn_lmk_bary_coords = self._find_dynamic_lmk_idx_and_bcoords(
            full_pose, self.dynamic_lmk_faces_idx,
            self.dynamic_lmk_bary_coords,
            self.neck_kin_chain, dtype=self.dtype)
        lmk_faces_idx = torch.cat([dyn_lmk_faces_idx, lmk_faces_idx], 1)
        lmk_bary_coords = torch.cat([dyn_lmk_bary_coords, lmk_bary_coords], 1)

        landmarks2d = vertices2landmarks(vertices, self.faces_tensor,
                                         lmk_faces_idx,
                                         lmk_bary_coords)
        bz = vertices.shape[0]
        landmarks3d = vertices2landmarks(vertices, self.faces_tensor,
                                         self.full_lmk_faces_idx.repeat(bz, 1),
                                         self.full_lmk_bary_coords.repeat(bz, 1, 1))

        return vertices, landmarks2d, landmarks3d


class FLAME_mediapipe(FLAME):
    """
    FLAME model for mediapipe.
    """

    def __init__(self, config):
        """
        Initializes the FLAME mediapipe model.
        Args:
            config: Configuration object.
        """
        super().__init__(config)
        lmk_embeddings_mediapipe = np.load(config.flame_mediapipe_lmk_embedding_path,
                                           allow_pickle=True, encoding='latin1')
        self.register_buffer('lmk_faces_idx_mediapipe',
                             torch.tensor(lmk_embeddings_mediapipe['lmk_face_idx'].astype(np.int64), dtype=torch.long))
        self.register_buffer('lmk_bary_coords_mediapipe',
                             torch.tensor(lmk_embeddings_mediapipe['lmk_b_coords'], dtype=self.dtype))

    def forward(self, shape_params=None, expression_params=None, pose_params=None, eye_pose_params=None):
        """
        Forward pass of the FLAME mediapipe model.
        Args:
            shape_params (torch.Tensor): Shape parameters.
            expression_params (torch.Tensor): Expression parameters.
            pose_params (torch.Tensor): Pose parameters.
            eye_pose_params (torch.Tensor): Eye pose parameters.

        Returns:
            Tuple[torch.Tensor]: Vertices, 2D landmarks, 3D landmarks, mediapipe landmarks.
        """
        vertices, landmarks2d, landmarks3d = super().forward(shape_params, expression_params, pose_params,
                                                              eye_pose_params)
        batch_size = shape_params.shape[0]
        lmk_faces_idx_mediapipe = self.lmk_faces_idx_mediapipe.unsqueeze(dim=0).expand(batch_size, -1).contiguous()
        lmk_bary_coords_mediapipe = self.lmk_bary_coords_mediapipe.unsqueeze(dim=0).expand(batch_size, -1,
                                                                                            -1).contiguous()
        landmarks2d_mediapipe = vertices2landmarks(vertices, self.faces_tensor,
                                                    lmk_faces_idx_mediapipe,
                                                    lmk_bary_coords_mediapipe)

        return vertices, landmarks2d, landmarks3d, landmarks2d_mediapipe


class FLAMETex(nn.Module):
    """
    FLAME texture model.
    """

    def __init__(self, config):
        """
        Initializes the FLAME texture model.
        Args:
            config: Configuration object.
        """
        super(FLAMETex, self).__init__()
        if config.tex_type == 'BFM':
            mu_key = 'MU'
            pc_key = 'PC'
            n_pc = 199
            tex_path = config.tex_path
            tex_space = np.load(tex_path)
            texture_mean = tex_space[mu_key].reshape(1, -1)
            texture_basis = tex_space[pc_key].reshape(-1, n_pc)

        elif config.tex_type == 'FLAME':
            mu_key = 'mean'
            pc_key = 'tex_dir'
            n_pc = 200
            tex_path = config.tex_path
            tex_space = np.load(tex_path)
            texture_mean = tex_space[mu_key].reshape(1, -1) / 255.
            texture_basis = tex_space[pc_key].reshape(-1, n_pc) / 255.

        else:
            print('texture type ', config.tex_type, 'not exist!')
            exit()

        n_tex = config.n_tex
        num_components = texture_basis.shape[1]
        texture_mean = torch.from_numpy(texture_mean).float()[None, ...]
        texture_basis = torch.from_numpy(texture_basis[:, :n_tex]).float()[None, ...]
        self.register_buffer('texture_mean', texture_mean)
        self.register_buffer('texture_basis', texture_basis)

    def forward(self, texcode):
        """
        Forward pass of the FLAME texture model.
        Args:
            texcode (torch.Tensor): Texture code.

        Returns:
            torch.Tensor: Texture.
        """
        texture = self.texture_mean + (self.texture_basis * texcode[:, None, :]).sum(-1)
        texture = texture.reshape(texcode.shape[0], 512, 512, 3).permute(0, 3, 1, 2)
        texture = F.interpolate(texture, [256, 256])
        texture = texture[:, [2, 1, 0], :, :]
        return texture


class FLAMETex_trainable(nn.Module):
    """
    Trainable FLAME texture model.
    """

    def __init__(self, config):
        """
        Initializes the trainable FLAME texture model.
        Args:
            config: Configuration object.
        """
        super(FLAMETex_trainable, self).__init__()
        tex_params = config.tex_params
        texture_model = np.load(config.tex_path)

        num_tex_pc = texture_model['PC'].shape[-1]
        tex_shape = texture_model['MU'].shape

        MU = torch.from_numpy(np.reshape(texture_model['MU'], (1, -1))).float()[None, ...]
        PC = torch.from_numpy(np.reshape(texture_model['PC'], (-1, num_tex_pc))[:, :tex_params]).float()[None, ...]
        self.register_buffer('MU', MU)
        self.register_buffer('PC', PC)

        num_tex_pc = PC.shape[2]

        tex_mean = torch.from_numpy(texture_model['mean'] / 255.).float().unsqueeze(0).unsqueeze(0)
        tex_map = torch.from_numpy(texture_model['tex_map']).float().unsqueeze(0).unsqueeze(0)
        self.register_buffer('tex_mean', tex_mean)
        self.register_buffer('tex_map', tex_map)

        self.cfg = config

        self.register_parameter('tex_code', nn.Parameter(torch.zeros((1, num_tex_pc), requires_grad=True)))

    def forward(self):
        """
        Forward pass of the trainable FLAME texture model.

        Returns:
            torch.Tensor: Texture.
        """
        texcode = torch.matmul(self.tex_code, self.PC.permute(0, 2, 1)) + self.MU

        texture = self.tex_mean + F.conv2d(texcode.view(-1, 1, 1, self.cfg.tex_params),
                                           self.tex_map, padding=1)

        return texture.permute(0, 2, 3, 1)

