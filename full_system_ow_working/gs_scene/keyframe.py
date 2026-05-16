import torch
import torch.nn.functional as F
import math


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


class Keyframe:
    def __init__(self, image, depth, extrinsics, K, mask=None, pyr_levels=2, use_exposure=False, lr_exposure=1e-3, feat_map=None, mono_idepth=None, mono_depth_conf=None):
        self.image_pyr = [image]
        self.pyr_lvl = pyr_levels - 1
        for _ in range(self.pyr_lvl):
            self.image_pyr.append(F.avg_pool2d(self.image_pyr[-1], 2))
        
        self.depth = depth
        self.depth_loss_weight = 1e-2
        self.depth_loss_weight_decay = 0.9
        self.extrinsics = extrinsics
        self.K = K
        self.height, self.width = image.shape[1:3]
        
        self.mask_pyr = [mask.unsqueeze(0).float()] if mask is not None else None
        if self.mask_pyr is not None:
            for _ in range(self.pyr_lvl):
                self.mask_pyr.append(F.avg_pool2d(self.mask_pyr[-1], 2))
            for i in range(len(self.mask_pyr)):
                self.mask_pyr[i] = self.mask_pyr[i] > (1 - 1e-6)
                
        self.approx_centre = -self.get_R().T @ self.get_t()
        self.num_steps = 0
        
        # Exposure compensation
        self.use_exposure = use_exposure
        if use_exposure:
            # 3x4 matrix initialized as [I | 0]
            exposure = torch.eye(3, 4, device=image.device)
            self.exposure = torch.nn.Parameter(exposure)
            self.optimizer_exposure = torch.optim.Adam([self.exposure], lr=lr_exposure)
        else:
            self.exposure = None
            self.optimizer_exposure = None
        
        # Guided MVS support
        self.feat_map = feat_map
        self.mono_idepth = mono_idepth
        self.mono_depth_conf = mono_depth_conf
        
        # For compatibility with GuidedMVS call (expects .f and .centre tensors)
        self.f = torch.tensor([(K[0, 0] + K[1, 1]) / 2.0], device=image.device)
        self.centre = torch.tensor([(self.width - 1) / 2, (self.height - 1) / 2], device=image.device)

    def step(self):
        self.depth_loss_weight *= self.depth_loss_weight_decay
        self.num_steps += 1
        
        # Progressively reduce pyramid level for multiscale optimization
        if self.num_steps % 5 == 0:
            if self.pyr_lvl > 0:
                # self.image_pyr.pop() # Don't pop, just decrease pyr_lvl index if we want to keep it? 
                # on-the-fly-nvs pops it. Let's follow that if we want to save memory.
                # However, if we pop, we can't go back. 
                # Let's just decrease the index pointer.
                self.pyr_lvl -= 1
        
        if self.use_exposure and self.optimizer_exposure is not None:
            self.optimizer_exposure.step()
            self.optimizer_exposure.zero_grad()

    def get_fov(self):
        FoVx = focal2fov(self.K[0, 0], self.width)
        FoVy = focal2fov(self.K[1, 1], self.height)
        return FoVx, FoVy

    def get_t(self):
        return self.extrinsics[:3, 3]

    def get_R(self):
        return self.extrinsics[:3, :3]

    def get_Rt(self):
        Rt = torch.eye(4, device=self.extrinsics.device)
        Rt[:3, :3] = self.get_R()
        Rt[:3, 3] = self.get_t()
        return Rt

    def apply_exposure(self, image):
        if not self.use_exposure:
            return image
        # image: [3, H, W]
        H, W = image.shape[1:]
        image_flat = image.view(3, -1)
        # Exposure application: E_R @ image + E_T
        image_exposed = (self.exposure[:, :3] @ image_flat) + self.exposure[:, 3:]
        return image_exposed.view(3, H, W).clamp(0, 1)
