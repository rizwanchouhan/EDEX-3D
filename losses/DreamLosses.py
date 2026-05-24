import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
import torchvision.models as models

# L2 distance between two sets of vertices
def l2_distance(verts1, verts2):
    return torch.sqrt(((verts1 - verts2) ** 2).sum(2)).mean(1).mean()

# Kullback-Leibler divergence loss
def kl_loss(texcode):
    mu, logvar = texcode[:, :128], texcode[:, 128:]
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return KLD

# Loss for ensuring shading is close to white
def shading_white_loss(shading):
    rgb_diff = (shading.mean([0, 2, 3]) - 0.99) ** 2
    return rgb_diff.mean()

# Loss for ensuring smooth shading
def shading_smooth_loss(shading):
    dx = shading[:, :, 1:-1, 1:] - shading[:, :, 1:-1, :-1]
    dy = shading[:, :, 1:, 1:-1] - shading[:, :, :-1, 1:-1]
    gradient_image = (dx ** 2).mean() + (dy ** 2).mean()
    return gradient_image.mean()

# Loss for albedo constancy
def albedo_constancy_loss(albedo, alpha=15, weight=1.):
    albedo_chromaticity = albedo / (torch.sum(albedo, dim=1, keepdim=True) + 1e-6)
    weight_x = torch.exp(-alpha * (albedo_chromaticity[:, :, 1:, :] - albedo_chromaticity[:, :, :-1, :]) ** 2).detach()
    weight_y = torch.exp(-alpha * (albedo_chromaticity[:, :, :, 1:] - albedo_chromaticity[:, :, :, :-1]) ** 2).detach()
    albedo_const_loss_x = ((albedo[:, :, 1:, :] - albedo[:, :, :-1, :]) ** 2) * weight_x
    albedo_const_loss_y = ((albedo[:, :, :, 1:] - albedo[:, :, :, :-1]) ** 2) * weight_y
    albedo_constancy_loss = albedo_const_loss_x.mean() + albedo_const_loss_y.mean()
    return albedo_constancy_loss * weight

# Loss for albedo ring consistency
def albedo_ring_loss(texcode, ring_elements, margin, weight=1.):
    tot_ring_loss = (texcode[0] - texcode[0]).sum()
    diff_stream = texcode[-1]
    count = 0.0
    for i in range(ring_elements - 1):
        for j in range(ring_elements - 1):
            pd = (texcode[i] - texcode[j]).pow(2).sum(1)
            nd = (texcode[i] - diff_stream).pow(2).sum(1)
            tot_ring_loss = torch.add(tot_ring_loss, (torch.nn.functional.relu(margin + pd - nd).mean()))
            count += 1.0
    tot_ring_loss = (1.0 / count) * tot_ring_loss
    return tot_ring_loss * weight

# Loss for ensuring consistent albedo
def albedo_same_loss(albedo, ring_elements, weight=1.):
    loss = 0
    for i in range(ring_elements - 1):
        for j in range(ring_elements - 1):
            pd = (albedo[i] - albedo[j]).pow(2).mean()
            loss += pd
    loss = loss / ring_elements
    return loss * weight

# Loss for batch of 2D keypoint L1 loss
def batch_kp_2d_l1_loss(real_2d_kp, predicted_2d_kp, weights=None):
    if weights is not None:
        real_2d_kp[:, :, 2] = weights[None, :] * real_2d_kp[:, :, 2]
    kp_gt = real_2d_kp.view(-1, 3)
    kp_pred = predicted_2d_kp.contiguous().view(-1, 2)
    vis = kp_gt[:, 2]
    k = torch.sum(vis) * 2.0 + 1e-8
    dif_abs = torch.abs(kp_gt[:, :2] - kp_pred).sum(1)
    return torch.matmul(dif_abs, vis) * 1.0 / k

# Loss for landmark consistency
def landmark_loss(predicted_landmarks, landmarks_gt, weight=1.):
    if torch.is_tensor(landmarks_gt) is not True:
        real_2d = torch.cat(landmarks_gt) 
    else:
        real_2d = torch.cat([landmarks_gt, torch.ones((landmarks_gt.shape[0], 68, 1))], dim=-1)
    loss_lmk_2d = batch_kp_2d_l1_loss(real_2d, predicted_landmarks)
    return loss_lmk_2d * weight

# Loss for eye distance consistency
def eye_dis(landmarks):
    eye_up = landmarks[:, [37, 38, 43, 44], :]
    eye_bottom = landmarks[:, [41, 40, 47, 46], :]
    dis = torch.sqrt(((eye_up - eye_bottom) ** 2).sum(2))  
    return dis

# Loss for eye distance difference
def eyed_loss(predicted_landmarks, landmarks_gt, weight=1.):
    if torch.is_tensor(landmarks_gt) is not True:
        real_2d = torch.cat(landmarks_gt) 
    else:
        real_2d = torch.cat([landmarks_gt, torch.ones((landmarks_gt.shape[0], 68, 1))], dim=-1)
    pred_eyed = eye_dis(predicted_landmarks[:, :, :2])
    gt_eyed = eye_dis(real_2d[:, :, :2])
    loss = (pred_eyed - gt_eyed).abs().mean()
    return loss

# Loss for lip distance consistency
def lip_dis(landmarks):
    lip_up = landmarks[:, [61, 62, 63], :]
    lip_down = landmarks[:, [67, 66, 65], :]
    dis = torch.sqrt(((lip_up - lip_down) ** 2).sum(2))  
    return dis

# Loss for lip distance difference
def lipd_loss(predicted_landmarks, landmarks_gt, weight=1.):
    if torch.is_tensor(landmarks_gt) is not True:
        real_2d = torch.cat(landmarks_gt)
    else:
        real_2d = torch.cat([landmarks_gt, torch.ones((landmarks_gt.shape[0], 68, 1))], dim=-1)
    pred_lipd = lip_dis(predicted_landmarks[:, :, :2])
    gt_lipd = lip_dis(real_2d[:, :, :2])
    loss = (pred_lipd - gt_lipd).abs().mean()
    return loss

# Loss for mouth corner distance difference
def mouth_corner_loss(predicted_landmarks, landmarks_gt, weight=1.):
    if torch.is_tensor(landmarks_gt) is not True:
        real_2d = torch.cat(landmarks_gt)
    else:
        real_2d = torch.cat([landmarks_gt, torch.ones((landmarks_gt.shape[0], 68, 1))], dim=-1)
    pred_lipd = mouth_corner_dis(predicted_landmarks[:, :, :2])
    gt_lipd = mouth_corner_dis(real_2d[:, :, :2])
    loss = (pred_lipd - gt_lipd).abs().mean()
    return loss

# Loss for weighted landmark consistency
def weighted_landmark_loss(predicted_landmarks, landmarks_gt, weight=1.):
    real_2d = landmarks_gt
    weights = torch.ones((68,))
    weights[5:7] = 2
    weights[10:12] = 2
    weights[27:36] = 1.5
    weights[30] = 3
    weights[31] = 3
    weights[35] = 3
    weights[60:68] = 1.5
    weights[48:60] = 1.5
    weights[48] = 3
    weights[54] = 3
    if real_2d.shape[2] == 2:
        real_2d = torch.cat([real_2d, torch.ones((real_2d.shape[0], real_2d.shape[1], 1))], dim=2)
    loss_lmk_2d = batch_kp_2d_l1_loss(real_2d, predicted_landmarks, weights)
    return loss_lmk_2d * weight

# Loss for landmark consistency using tensor inputs
def landmark_loss_tensor(predicted_landmarks, landmarks_gt, weight=1.):
    loss_lmk_2d = batch_kp_2d_l1_loss(landmarks_gt, predicted_landmarks)
    return loss_lmk_2d * weight

# Loss for ring consistency
def ring_loss(ring_outputs, ring_type, margin, weight=1.):
    tot_ring_loss = (ring_outputs[0] - ring_outputs[0]).sum()
    if ring_type == '51':
        diff_stream = ring_outputs[-1]
        count = 0.0
        for i in range(6):
            for j in range(6):
                pd = (ring_outputs[i] - ring_outputs[j]).pow(2).sum(1)
                nd = (ring_outputs[i] - diff_stream).pow(2).sum(1)
                tot_ring_loss = torch.add(tot_ring_loss, (torch.nn.functional.relu(margin + pd - nd).mean()))
                count += 1.0
    elif ring_type == '33':
        perm_code = [(0, 1, 3), (0, 1, 4), (0, 1, 5), (0, 2, 3), (0, 2, 4), (0, 2, 5), (1, 0, 3), (1, 0, 4), (1, 0, 5),
                     (1, 2, 3), (1, 2, 4), (1, 2, 5), (2, 0, 3), (2, 0, 4), (2, 0, 5), (2, 1, 3), (2, 1, 4), (2, 1, 5)]
        count = 0.0
        for i in perm_code:
            pd = (ring_outputs[i[0]] - ring_outputs[i[1]]).pow(2).sum(1)
            nd = (ring_outputs[i[1]] - ring_outputs[i[2]]).pow(2).sum(1)
            tot_ring_loss = torch.add(tot_ring_loss, (torch.nn.functional.relu(margin + pd - nd).mean()))
            count += 1.0
    tot_ring_loss = (1.0 / count) * tot_ring_loss
    return tot_ring_loss * weight

# Loss for difference in gradients between prediction and ground truth
def gradient_dif_loss(prediction, gt):
    prediction_diff_x = prediction[:, :, 1:-1, 1:] - prediction[:, :, 1:-1, :-1]
    prediction_diff_y = prediction[:, :, 1:, 1:-1] - prediction[:, :, 1:, 1:-1]
    gt_x = gt[:, :, 1:-1, 1:] - gt[:, :, 1:-1, :-1]
    gt_y = gt[:, :, 1:, 1:-1] - gt[:, :, :-1, 1:-1]
    diff = torch.mean((prediction_diff_x - gt_x) ** 2) + torch.mean((prediction_diff_y - gt_y) ** 2)
    return diff.mean()

# Function to get 2D Laplacian kernel
def get_laplacian_kernel2d(kernel_size: int):
    if not isinstance(kernel_size, int) or kernel_size % 2 == 0 or \
            kernel_size <= 0:
        raise TypeError("ksize must be an odd positive integer. Got {}"
                        .format(kernel_size))
    kernel = torch.ones((kernel_size, kernel_size))
    mid = kernel_size // 2
    kernel[mid, mid] = 1 - kernel_size ** 2
    kernel_2d: torch.Tensor = kernel
    return kernel_2d

# Loss for high-quality Laplacian
def laplacian_hq_loss(prediction, gt):
    b, c, h, w = prediction.shape
    kernel_size = 3
    kernel = get_laplacian_kernel2d(kernel_size).to(device=prediction.device).to(prediction.dtype)
    kernel = kernel.repeat(c, 1, 1, 1)
    padding = (kernel_size - 1) // 2
    lap_pre = F.conv2d(prediction, kernel, padding=padding, stride=1, groups=c)
    lap_gt = F.conv2d(gt, kernel, padding=padding, stride=1, groups=c)
    return ((lap_pre - lap_gt) ** 2).mean()

# Class for VGG19 feature layer extraction
class VGG19FeatLayer(nn.Module):
    def __init__(self):
        super(VGG19FeatLayer, self).__init__()
        self.vgg19 = models.vgg19(pretrained=True).features.eval() 
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) )
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        out = {}
        x = x - self.mean
        x = x / self.std
        ci = 1
        ri = 0
        for layer in self.vgg19.children():
            if isinstance(layer, nn.Conv2d):
                ri += 1
                name = 'conv{}_{}'.format(ci, ri)
            elif isinstance(layer, nn.ReLU):
                ri += 1
                name = 'relu{}_{}'.format(ci, ri)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                ri = 0
                name = 'pool_{}'.format(ci)
                ci += 1
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(ci)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
            x = layer(x)
            out[name] = x
        return out

# Class for IDMRFLoss
class IDMRFLoss(nn.Module):
    def __init__(self, featlayer=VGG19FeatLayer):
        super(IDMRFLoss, self).__init__()
        self.featlayer = featlayer()
        self.feat_style_layers = {'relu3_2': 1.0, 'relu4_2': 1.0}
        self.feat_content_layers = {'relu4_2': 1.0}
        self.bias = 1.0
        self.nn_stretch_sigma = 0.5
        self.lambda_style = 1.0
        self.lambda_content = 1.0

    def sum_normalize(self, featmaps):
        reduce_sum = torch.sum(featmaps, dim=1, keepdim=True)
        return featmaps / reduce_sum

    def patch_extraction(self, featmaps):
        patch_size = 1
        patch_stride = 1
        patches_as_depth_vectors = featmaps.unfold(2, patch_size, patch_stride).unfold(3, patch_size, patch_stride)
        self.patches_OIHW = patches_as_depth_vectors.permute(0, 2, 3, 1, 4, 5)
        dims = self.patches_OIHW.size()
        self.patches_OIHW = self.patches_OIHW.view(-1, dims[3], dims[4], dims[5])
        return self.patches_OIHW

    def compute_relative_distances(self, cdist):
        epsilon = 1e-5
        div = torch.min(cdist, dim=1, keepdim=True)[0]
        relative_dist = cdist / (div + epsilon)
        return relative_dist

    def exp_norm_relative_dist(self, relative_dist):
        scaled_dist = relative_dist
        dist_before_norm = torch.exp((self.bias - scaled_dist) / self.nn_stretch_sigma)
        self.cs_NCHW = self.sum_normalize(dist_before_norm)
        return self.cs_NCHW

    def mrf_loss(self, gen, tar):
        meanT = torch.mean(tar, 1, keepdim=True)
        gen_feats, tar_feats = gen - meanT, tar - meanT

        gen_feats_norm = torch.norm(gen_feats, p=2, dim=1, keepdim=True)
        tar_feats_norm = torch.norm(tar_feats, p=2, dim=1, keepdim=True)

        gen_normalized = gen_feats / gen_feats_norm
        tar_normalized = tar_feats / tar_feats_norm

        cosine_dist_l = []
        BatchSize = tar.size(0)

        for i in range(BatchSize):
            tar_feat_i = tar_normalized[i:i + 1, :, :, :]
            gen_feat_i = gen_normalized[i:i + 1, :, :, :]
            patches_OIHW = self.patch_extraction(tar_feat_i)

            cosine_dist_i = F.conv2d(gen_feat_i, patches_OIHW)
            cosine_dist_l.append(cosine_dist_i)
        cosine_dist = torch.cat(cosine_dist_l, dim=0)
        cosine_dist_zero_2_one = - (cosine_dist - 1) / 2
        relative_dist = self.compute_relative_distances(cosine_dist_zero_2_one)
        rela_dist = self.exp_norm_relative_dist(relative_dist)
        dims_div_mrf = rela_dist.size()
        k_max_nc = torch.max(rela_dist.view(dims_div_mrf[0], dims_div_mrf[1], -1), dim=2)[0]
        div_mrf = torch.mean(k_max_nc, dim=1)
        div_mrf_sum = -torch.log(div_mrf)
        div_mrf_sum = torch.sum(div_mrf_sum)
        return div_mrf_sum

    def forward(self, gen, tar):
        gen_vgg_feats = self.featlayer(gen)
        tar_vgg_feats = self.featlayer(tar)
        style_loss_list = [self.feat_style_layers[layer] * self.mrf_loss(gen_vgg_feats[layer], tar_vgg_feats[layer]) for
                           layer in self.feat_style_layers]
        self.style_loss = reduce(lambda x, y: x + y, style_loss_list) * self.lambda_style

        content_loss_list = [self.feat_content_layers[layer] * self.mrf_loss(gen_vgg_feats[layer], tar_vgg_feats[layer])
                             for layer in self.feat_content_layers]
        self.content_loss = reduce(lambda x, y: x + y, content_loss_list) * self.lambda_content

        return self.style_loss + self.content_loss

    def train(self, b = True):
        return super().train(False)

class VGG_16(nn.Module):
    """
    Main Class
    """

    def __init__(self):
        """
        Constructor
        """
        super().__init__()
        self.block_size = [2, 2, 3, 3, 3]
        self.conv_1_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv_1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv_2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv_2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv_3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv_3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv_4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.fc6 = nn.Linear(512 * 7 * 7, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 2622)

        self.register_buffer('mean', torch.Tensor(np.array([129.1863, 104.7624, 93.5940]) / 255.).float().view(1, 3, 1, 1))


    def load_weights(self, path="pretrained/VGG_FACE.t7"):
        """ Function to load luatorch pretrained
        Args:
            path: path for the luatorch pretrained
        """
        model = torchfile.load(path)
        counter = 1
        block = 1
        for i, layer in enumerate(model.modules):
            if layer.weight is not None:
                if block <= 5:
                    self_layer = getattr(self, "conv_%d_%d" % (block, counter))
                    counter += 1
                    if counter > self.block_size[block - 1]:
                        counter = 1
                        block += 1
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]
                else:
                    self_layer = getattr(self, "fc%d" % (block))
                    block += 1
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]

    def forward(self, x):
        """ Pytorch forward
        Args:
            x: input image (224x224)
        Returns: class logits
        """
        out = {}
        x = x - self.mean
        x = F.relu(self.conv_1_1(x))
        x = F.relu(self.conv_1_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2_1(x))
        x = F.relu(self.conv_2_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_3_1(x))
        x = F.relu(self.conv_3_2(x))
        out['relu3_2'] = x
        x = F.relu(self.conv_3_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_4_1(x))
        x = F.relu(self.conv_4_2(x))
        out['relu4_2'] = x
        x = F.relu(self.conv_4_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_5_1(x))
        x = F.relu(self.conv_5_2(x))
        x = F.relu(self.conv_5_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        x = F.dropout(x, 0.5, self.training)
        x = F.relu(self.fc7(x))
        x = F.dropout(x, 0.5, self.training)
        x = self.fc8(x)
        out['last'] = x
        return out


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.featlayer = VGG_16().float()
        self.featlayer.load_weights(path="data/face_recognition_model/vgg_face_torch/VGG_FACE.t7")
        self.featlayer = self.featlayer.eval()
        self.feat_style_layers = {'relu3_2': 1.0, 'relu4_2': 1.0}
        self.feat_content_layers = {'relu4_2': 1.0}
        self.bias = 1.0
        self.nn_stretch_sigma = 0.5
        self.lambda_style = 1.0
        self.lambda_content = 1.0

    def sum_normalize(self, featmaps):
        reduce_sum = torch.sum(featmaps, dim=1, keepdim=True)
        return featmaps / reduce_sum

    def patch_extraction(self, featmaps):
        patch_size = 1
        patch_stride = 1
        patches_as_depth_vectors = featmaps.unfold(2, patch_size, patch_stride).unfold(3, patch_size, patch_stride)
        self.patches_OIHW = patches_as_depth_vectors.permute(0, 2, 3, 1, 4, 5)
        dims = self.patches_OIHW.size()
        self.patches_OIHW = self.patches_OIHW.view(-1, dims[3], dims[4], dims[5])
        return self.patches_OIHW

    def compute_relative_distances(self, cdist):
        epsilon = 1e-5
        div = torch.min(cdist, dim=1, keepdim=True)[0]
        relative_dist = cdist / (div + epsilon)
        return relative_dist

    def exp_norm_relative_dist(self, relative_dist):
        scaled_dist = relative_dist
        dist_before_norm = torch.exp((self.bias - scaled_dist) / self.nn_stretch_sigma)
        self.cs_NCHW = self.sum_normalize(dist_before_norm)
        return self.cs_NCHW

    def mrf_loss(self, gen, tar):
        meanT = torch.mean(tar, 1, keepdim=True)
        gen_feats, tar_feats = gen - meanT, tar - meanT

        gen_feats_norm = torch.norm(gen_feats, p=2, dim=1, keepdim=True)
        tar_feats_norm = torch.norm(tar_feats, p=2, dim=1, keepdim=True)

        gen_normalized = gen_feats / gen_feats_norm
        tar_normalized = tar_feats / tar_feats_norm

        cosine_dist_l = []
        BatchSize = tar.size(0)

        for i in range(BatchSize):
            tar_feat_i = tar_normalized[i:i + 1, :, :, :]
            gen_feat_i = gen_normalized[i:i + 1, :, :, :]
            patches_OIHW = self.patch_extraction(tar_feat_i)

            cosine_dist_i = F.conv2d(gen_feat_i, patches_OIHW)
            cosine_dist_l.append(cosine_dist_i)
        cosine_dist = torch.cat(cosine_dist_l, dim=0)
        cosine_dist_zero_2_one = - (cosine_dist - 1) / 2
        relative_dist = self.compute_relative_distances(cosine_dist_zero_2_one)
        rela_dist = self.exp_norm_relative_dist(relative_dist)
        dims_div_mrf = rela_dist.size()
        k_max_nc = torch.max(rela_dist.view(dims_div_mrf[0], dims_div_mrf[1], -1), dim=2)[0]
        div_mrf = torch.mean(k_max_nc, dim=1)
        div_mrf_sum = -torch.log(div_mrf)
        div_mrf_sum = torch.sum(div_mrf_sum)
        return div_mrf_sum

    def forward(self, gen, tar):
        ## gen: [bz,3,h,w] rgb [0,1]
        gen_vgg_feats = self.featlayer(gen)
        tar_vgg_feats = self.featlayer(tar)
        style_loss_list = [self.feat_style_layers[layer] * self.mrf_loss(gen_vgg_feats[layer], tar_vgg_feats[layer]) for
                           layer in self.feat_style_layers]
        self.style_loss = reduce(lambda x, y: x + y, style_loss_list) * self.lambda_style

        content_loss_list = [self.feat_content_layers[layer] * self.mrf_loss(gen_vgg_feats[layer], tar_vgg_feats[layer])
                             for layer in self.feat_content_layers]
        self.content_loss = reduce(lambda x, y: x + y, content_loss_list) * self.lambda_content

        return self.style_loss + self.content_loss

from facenet_pytorch import InceptionResnetV1


class IdentityLoss(nn.Module):
    def __init__(self, pretrained_data='vggface2'):
        super(IdentityLoss, self).__init__()
        # Initialize InceptionResnetV1 model
        self.reg_model = InceptionResnetV1(pretrained=pretrained_data).eval()
        # Set parameters
        self.bias = 1.0
        self.nn_stretch_sigma = 0.5
        self.lambda_style = 1.0
        self.lambda_content = 1.0

    def _cos_metric(self, x, y, dim=1):
        return F.cosine_similarity(x, y)

    def _l2_metric(self, x, y):
        return ((x - y) ** 2).mean()

    def reg_features(self, x):
        # Extract features from InceptionResnetV1 model
        out = []
        x = F.interpolate(x * 2. - 1., [160, 160])
        x = self.reg_model.conv2d_1a(x)
        x = self.reg_model.conv2d_2a(x)
        x = self.reg_model.conv2d_2b(x)
        x = self.reg_model.maxpool_3a(x)
        x = self.reg_model.conv2d_3b(x)
        x = self.reg_model.conv2d_4a(x)
        x = self.reg_model.conv2d_4b(x)
        x = self.reg_model.repeat_1(x)
        x = self.reg_model.mixed_6a(x)
        x = self.reg_model.repeat_2(x)
        out.append(x)
        x = self.reg_model.mixed_7a(x)
        x = self.reg_model.repeat_3(x)
        x = self.reg_model.block8(x)
        out.append(x)
        x = self.reg_model.avgpool_1a(x)
        x = self.reg_model.dropout(x)
        x = self.reg_model.last_linear(x.view(x.shape[0], -1))
        x = self.reg_model.last_bn(x)
        x = F.normalize(x, p=2, dim=1)
        out.append(x)
        return out

    def sum_normalize(self, featmaps):
        # Normalize feature maps
        reduce_sum = torch.sum(featmaps, dim=1, keepdim=True)
        return featmaps / reduce_sum

    def patch_extraction(self, featmaps):
        # Extract patches from feature maps
        patch_size = 1
        patch_stride = 1
        patches_as_depth_vectors = featmaps.unfold(2, patch_size, patch_stride).unfold(3, patch_size, patch_stride)
        self.patches_OIHW = patches_as_depth_vectors.permute(0, 2, 3, 1, 4, 5)
        dims = self.patches_OIHW.size()
        self.patches_OIHW = self.patches_OIHW.view(-1, dims[3], dims[4], dims[5])
        return self.patches_OIHW

    def compute_relative_distances(self, cdist):
        epsilon = 1e-5
        div = torch.min(cdist, dim=1, keepdim=True)[0]
        relative_dist = cdist / (div + epsilon)
        return relative_dist

    def exp_norm_relative_dist(self, relative_dist):
        scaled_dist = relative_dist
        dist_before_norm = torch.exp((self.bias - scaled_dist) / self.nn_stretch_sigma)
        self.cs_NCHW = self.sum_normalize(dist_before_norm)
        return self.cs_NCHW

    def mrf_loss(self, gen, tar):
        meanT = torch.mean(tar, 1, keepdim=True)
        gen_feats, tar_feats = gen - meanT, tar - meanT

        gen_feats_norm = torch.norm(gen_feats, p=2, dim=1, keepdim=True)
        tar_feats_norm = torch.norm(tar_feats, p=2, dim=1, keepdim=True)

        gen_normalized = gen_feats / gen_feats_norm
        tar_normalized = tar_feats / tar_feats_norm

        cosine_dist_l = []
        BatchSize = tar.size(0)

        for i in range(BatchSize):
            tar_feat_i = tar_normalized[i:i + 1, :, :, :]
            gen_feat_i = gen_normalized[i:i + 1, :, :, :]
            patches_OIHW = self.patch_extraction(tar_feat_i)

            cosine_dist_i = F.conv2d(gen_feat_i, patches_OIHW)
            cosine_dist_l.append(cosine_dist_i)
        cosine_dist = torch.cat(cosine_dist_l, dim=0)
        cosine_dist_zero_2_one = - (cosine_dist - 1) / 2
        relative_dist = self.compute_relative_distances(cosine_dist_zero_2_one)
        rela_dist = self.exp_norm_relative_dist(relative_dist)
        dims_div_mrf = rela_dist.size()
        k_max_nc = torch.max(rela_dist.view(dims_div_mrf[0], dims_div_mrf[1], -1), dim=2)[0]
        div_mrf = torch.mean(k_max_nc, dim=1)
        div_mrf_sum = -torch.log(div_mrf)
        div_mrf_sum = torch.sum(div_mrf_sum)
        return div_mrf_sum

    def forward(self, gen, tar, content_loss=False, identity_loss=True, content_type='mrf', identity_type='l2'):
        # Get features from input images
        gen_out = self.reg_features(gen)
        tar_out = self.reg_features(tar)

        # Calculate identity loss if enabled
        if identity_loss:
            if identity_type == 'l2':
                loss = ((gen_out[-1] - tar_out[-1]) ** 2).mean()
            else:
                loss = 1 - F.cosine_similarity(gen_out[-1], tar_out[-1]).mean()
        else:
            loss = 0.

        # Calculate content loss if enabled
        if content_loss:
            weight = [1, 1, 1, 1]
            for i in range(len(gen_out) - 1):
                if content_type == 'mrf':
                    loss_curr = self.mrf_loss(gen_out[i], tar_out[i]) * 0.0001
                elif content_type == 'l2':
                    loss_curr = self._l2_metric(gen_out[i], tar_out[i]) * 0.02
                loss = loss + loss_curr * weight[i]

        return loss


class VGGFace2Loss(nn.Module):
    def __init__(self, pretrained_checkpoint_path=None, metric='cosine_similarity', trainable=False):
        super(VGGFace2Loss, self).__init__()
        # Initialize resnet50 model
        self.reg_model = resnet50(num_classes=8631, include_top=False).eval()
        # Load pretrained weights
        checkpoint = pretrained_checkpoint_path or \
                     '/home/abbas/dream/assets/FaceRecognition/resnet50_ft_weight.pkl'
        load_state_dict(self.reg_model, checkpoint)
        # Set mean_bgr tensor
        self.register_buffer('mean_bgr', torch.tensor([91.4953, 103.8827, 131.0912]))
        # Set trainable parameter
        self.trainable = trainable

        # Validate metric
        if metric not in ["l1", "l1_loss", "l2", "mse", "mse_loss", "cosine_similarity",
                          "barlow_twins", "barlow_twins_headless"]:
            raise ValueError(f"Invalid metric for face recognition feature loss: {metric}")

        # Initialize Barlow Twins loss if required
        if metric == "barlow_twins_headless":
            feature_size = self.reg_model.fc.in_features
            self.bt_loss = BarlowTwinsLossHeadless(feature_size)
        elif metric == "barlow_twins":
            feature_size = self.reg_model.fc.in_features
            self.bt_loss = BarlowTwinsLoss(feature_size)
        else:
            self.bt_loss = None

        self.metric = metric

    def _get_trainable_params(self):
        params = []
        if self.trainable:
            params += list(self.reg_model.parameters())
        if self.bt_loss is not None:
            params += list(self.bt_loss.parameters())
        return params

    def train(self, b = True):
        if not self.trainable:
            ret = super().train(False)
        else:
            ret = super().train(b)
        if self.bt_loss is not None:
            self.bt_loss.train(b)
        return ret

    def requires_grad_(self, b):
        super().requires_grad_(False) 
        if self.bt_loss is not None:
            self.bt_loss.requires_grad_(b)

    def freeze_nontrainable_layers(self):
        if not self.trainable:
            super().requires_grad_(False)
        else:
            super().requires_grad_(True)
        if self.bt_loss is not None:
            self.bt_loss.requires_grad_(True)

    def reg_features(self, x):
        # Extract features from resnet50 model
        margin = 10
        x = x[:, :, margin:224 - margin, margin:224 - margin]
        x = F.interpolate(x * 2. - 1., [224, 224], mode='bilinear')
        feature = self.reg_model(x)
        feature = feature.view(x.size(0), -1)
        return feature

    def transform(self, img):
        # Transform image
        img = img[:, [2, 1, 0], :, :].permute(0, 2, 3, 1) * 255 - self.mean_bgr
        img = img.permute(0, 3, 1, 2)
        return img

    def _cos_metric(self, x1, x2):
        return 1.0 - F.cosine_similarity(x1, x2, dim=1)

    def forward(self, gen, tar, is_crop=True, batch_size=None, ring_size=None):
        # Transform images
        gen = self.transform(gen)
        tar = self.transform(tar)

        # Extract features
        gen_out = self.reg_features(gen)
        tar_out = self.reg_features(tar)

        # Calculate loss based on metric
        if self.metric == "cosine_similarity":
            loss = self._cos_metric(gen_out, tar_out).mean()
        elif self.metric in ["l1", "l1_loss", "mae"]:
            loss = torch.nn.functional.l1_loss(gen_out, tar_out)
        elif self.metric in ["mse", "mse_loss", "l2", "l2_loss"]:
            loss = torch.nn.functional.mse_loss(gen_out, tar_out)
        elif self.metric in ["barlow_twins_headless", "barlow_twins"]:
            loss = self.bt_loss(gen_out, tar_out, batch_size=batch_size, ring_size=ring_size)
        else:
            raise ValueError(f"Invalid metric for face recognition feature loss: {self.metric}")

        return loss
