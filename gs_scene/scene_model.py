import numpy as np
import torch
import torch.nn.functional as F

from .keyframe import Keyframe, focal2fov
from .optimizers import SparseGaussianAdam, fused_ssim
from .anchor import Anchor
from .guided_mvs import GuidedMVS
from .dense_extractor import DenseExtractor
import math
from gsplat import rasterization
import rerun as rr


def depth2points(uv, depth, f, centre):
    xyz = torch.cat([(uv[..., :2] - centre) / f,
                    torch.ones_like(uv[..., 0:1])], dim=-1)
    return depth * xyz


def get_lapla_norm(img, kernel):
    laplacian_kernel = (
        torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], device="cuda", dtype=torch.float32
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )
    laplacian_kernel = laplacian_kernel.repeat(1, img.shape[0], 1, 1)
    laplacian = F.conv2d(img[None], laplacian_kernel, padding="same")
    laplacian_norm = torch.linalg.vector_norm(
        laplacian, ord=1, dim=1, keepdim=True)
    laplacian_norm[..., :, 0] = 0
    laplacian_norm[..., :, -1] = 0
    laplacian_norm[..., 0, :] = 0
    laplacian_norm[..., -1, :] = 0
    return F.conv2d(laplacian_norm, kernel, padding="same")[0, 0].clamp(0, 1)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def RGB2SH(rgb):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


class SceneModel:
    """
    This is a simplified version of GS (on-the-fly-nvs) model 
    https://vscode.dev/github/graphdeco-inria/on-the-fly-nvs/blob/main/scene/scene_model.py#L53
    """

    def __init__(self, width=504, height=504, f=None, num_steps=30, init_proba_scaler=2.0,
                 use_anchors=False, anchor_overlap=0.1, anchor_dist_threshold=2.0,
                 use_exposure=False, pyr_levels=1, use_guided_mvs=False):
        self.width = width
        self.height = height
        self.centre = torch.tensor(
            [(self.width - 1) / 2, (self.height - 1) / 2], device="cuda")
        self.f = f
        self.num_steps = num_steps
        
        # Hyperparameters
        self.position_lr_init = 5e-5
        self.feature_lr = 5e-3
        self.opacity_lr = 1e-1
        self.scaling_lr = 1e-2
        self.rotation_lr = 2e-3
        self.position_lr_decay = 1 - 2e-5
        
        self.active_sh_degree = 3
        self.max_sh_degree = 3
        self.init_proba_scaler = init_proba_scaler
        self.lambda_dssim = 0.2
        
        # Optional Components Configuration
        self.use_anchors = use_anchors
        self.anchor_overlap = anchor_overlap
        self.anchor_dist_threshold = anchor_dist_threshold
        self.use_exposure = use_exposure
        self.pyr_levels = pyr_levels
        self.use_guided_mvs = use_guided_mvs
        
        if self.use_guided_mvs:
            print("Initializing Guided MVS and Dense Extractor...")
            # Default to 6 neighboring frames for MVS
            self.guided_mvs = GuidedMVS(num_cams=6)
            self.dense_extractor = DenseExtractor(width, height)
        else:
            self.guided_mvs = None
            self.dense_extractor = None
        
        # Gaussian Parameters and Anchors
        self.gaussian_params = self._create_gaussian_params()
        self.anchors = []
        if self.use_anchors:
            self.active_anchor = Anchor(self.gaussian_params)
            self.anchors.append(self.active_anchor)
        else:
            self.active_anchor = None

        self.lr_dict = {
            "xyz": {
                "lr_init": self.position_lr_init,
                "lr_decay": self.position_lr_decay,
            }
        }
        self.reset_optimizer()

        radius = 3
        self.disc_kernel = torch.zeros(1, 1, 2 * radius + 1, 2 * radius + 1)
        y, x = torch.meshgrid(
            torch.arange(-radius, radius + 1),
            torch.arange(-radius, radius + 1),
            indexing="ij",
        )
        self.disc_kernel[0, 0, torch.sqrt(x**2 + y**2) <= radius + 0.5] = 1
        self.disc_kernel = self.disc_kernel.cuda() / self.disc_kernel.sum()

        self.uv = (
            torch.stack(
                torch.meshgrid(
                    torch.arange(0, self.width), torch.arange(0, self.height), indexing="xy"
                ),
                dim=-1,
            )
            .float()
            .cuda()
        )
        self.keyframes = []
        self.intrin_set = False

    def _create_gaussian_params(self):
        params = {
            "xyz": {"val": torch.empty(0, 3, device="cuda"), "lr": self.position_lr_init},
            "f_dc": {"val": torch.empty(0, 1, 3, device="cuda"), "lr": self.feature_lr},
            "f_rest": {"val": torch.empty(0, (self.max_sh_degree + 1)**2 - 1, 3, device="cuda"), "lr": self.feature_lr / 20.0},
            "scaling": {"val": torch.empty(0, 3, device="cuda"), "lr": self.scaling_lr},
            "rotation": {"val": torch.empty(0, 4, device="cuda"), "lr": self.rotation_lr},
            "opacity": {"val": torch.empty(0, 1, device="cuda"), "lr": self.opacity_lr},
        }
        for key in params:
            params[key]["val"].requires_grad = True
        return params

    def reset_optimizer(self):
        self.optimizer = SparseGaussianAdam(
            self.gaussian_params, (0.5, 0.99), lr_dict=self.lr_dict
        )

    @property
    def xyz(self):
        return self.gaussian_params["xyz"]["val"]

    @property
    def f_dc(self):
        return self.gaussian_params["f_dc"]["val"]

    @property
    def f_rest(self):
        return self.gaussian_params["f_rest"]["val"]

    @property
    def scaling(self):
        return torch.exp(self.gaussian_params["scaling"]["val"])

    @property
    def rotation(self):
        return F.normalize(self.gaussian_params["rotation"]["val"])

    @property
    def opacity(self):
        return torch.sigmoid(self.gaussian_params["opacity"]["val"])

    @property
    def n_active_gaussians(self):
        return self.xyz.shape[0]

    @property
    def colors(self):
        return SH2RGB(self.f_dc)

    def optimization_step(self, finetuning=False):
        if self.n_active_gaussians == 0:
            return
            
        if self.use_anchors and self.active_anchor:
            if len(self.active_anchor.keyframe_ids) == 0:
                keyframe_id = -1
            else:
                if np.random.rand() > 0.2:
                    # Pick from keyframes associated with THIS anchor
                    # Need to map keyframe_id (index in self.keyframes)
                    keyframe_id = np.random.choice(self.active_anchor.keyframe_ids)
                else:
                    keyframe_id = -1
        else:
            if np.random.rand() > 0.2:
                keyframe_id = np.random.choice(len(self.keyframes))
            else:
                keyframe_id = -1

        keyframe = self.keyframes[keyframe_id]
        lvl = keyframe.pyr_lvl

        # Zero gradients
        self.optimizer.zero_grad()
        if keyframe.optimizer_exposure is not None:
            keyframe.optimizer_exposure.zero_grad()

        # Render image and depth
        render_pkg = self.render_from_frame(keyframe, pyr_lvl=lvl)
        image = render_pkg["render"]
        
        # Apply exposure compensation if enabled
        if self.use_exposure:
            image = keyframe.apply_exposure(image)

        gt_image = keyframe.image_pyr[lvl]

        # Mask image if necessary
        if keyframe.mask_pyr is not None:
            image = image * keyframe.mask_pyr[lvl]
            gt_image = gt_image * keyframe.mask_pyr[lvl]

        # Loss
        l1_loss = (image - gt_image).abs().mean()
        ssim_loss = 1 - fused_ssim(image[None], gt_image[None])
        loss = (
            self.lambda_dssim * ssim_loss
            + (1 - self.lambda_dssim) * l1_loss
        )
        print(f"Loss: {loss.item():.6f} (L1: {l1_loss.item():.6f}, SSIM: {ssim_loss.item():.6f})")
        
        loss.backward()

        # Optimizers step
        with torch.no_grad():
            # Step keyframe (includes exposure optimizer and pyramid progression)
            keyframe.step()

            # Scene Gaussian optimization
            # Ensure we only update the active anchor's parameters
            # and that the visibility filter matches the number of active Gaussians
            vis = render_pkg["visibility_filter"]
            blended_ids = render_pkg.get("blended_ids")
            
            if blended_ids is not None and len(blended_ids) > 1:
                # Find which index in blended_ids is the active anchor
                try:
                    active_id = self.anchors.index(self.active_anchor)
                    if active_id in blended_ids.tolist():
                        idx = blended_ids.tolist().index(active_id)
                        # Find the slice
                        n_gauss_list = [self.anchors[i].gaussian_params["xyz"]["val"].shape[0] for i in blended_ids]
                        start = sum(n_gauss_list[:idx])
                        end = start + n_gauss_list[idx]
                        vis = vis[start:end]
                    else:
                        # Active anchor not visible, skip optimization?
                        # Or just use the whole filter (will likely crash anyway)
                        vis = vis[:0] # empty
                except ValueError:
                    pass
            
            self.optimizer.step(
                vis, self.n_active_gaussians
            )

    def optimization_loop(self, num_steps=30):
        for _ in range(num_steps):
            self.optimization_step()

    def make_dummy_ext_tensor(self):
        return {
            "xyz": self.xyz[:0].detach(),
            "f_dc": self.f_dc[:0].detach(),
            "f_rest": self.f_rest[:0].detach(),
            "opacity": self.opacity[:0].detach(),
            "scaling": self.scaling[:0].detach(),
            "rotation": self.rotation[:0].detach(),
        }

    def add_new_gaussians(self, frame):
        # Anchor check: if new frame is far from active anchor, start a new one
        if self.use_anchors and self.active_anchor:
            dist = torch.linalg.vector_norm(frame.approx_centre - self.active_anchor.position)
            if dist > self.anchor_dist_threshold:
                # Store current params in old anchor.
                self.active_anchor.gaussian_params = {k: {v_k: v_v.clone() for v_k, v_v in v.items()} for k, v in self.gaussian_params.items()}
                # Start new anchor
                self.gaussian_params = self._create_gaussian_params()
                self.active_anchor = Anchor(self.gaussian_params, position=frame.approx_centre.clone())
                self.anchors.append(self.active_anchor)
                self.reset_optimizer()

        with torch.no_grad():
            img = frame.image_pyr[0]
            img = F.avg_pool2d(img, 2)
            img = F.interpolate(
                img[None], (self.height, self.width), mode="bilinear", align_corners=True
            )[0]
            init_proba = get_lapla_norm(img, self.disc_kernel)
            penalty = 0
            rendered_depth = None
            if self.n_active_gaussians > 0:
                render_pkg = self.render_from_frame(frame)
                render = render_pkg["render"]
                # Apply exposure if needed for penalty calculation
                if self.use_exposure:
                    render = frame.apply_exposure(render)
                rendered_depth = 1 / render_pkg["invdepth"][0].clamp_min(1e-8)
                penalty = get_lapla_norm(render, self.disc_kernel)

            init_proba *= self.init_proba_scaler
            penalty *= self.init_proba_scaler
            sample_mask = torch.rand_like(init_proba) < init_proba - penalty

            if frame.mask_pyr is not None:
                sample_mask = sample_mask * frame.mask_pyr[0][0]

            sampled_uv = self.uv[sample_mask]
            
            if self.use_guided_mvs and len(self.keyframes) >= 3:
                # Use Guided MVS to refine depth
                # We need at least a few previous keyframes
                prev_kfs = self.keyframes[-6:] # Use up to last 6
                # Filter out current frame if it's already in self.keyframes (it usually isn't yet)
                # GuidedMVS expects (uv, refKeyframe, neighboringKeyframes)
                depth, valid_mvs = self.guided_mvs(sampled_uv, frame, prev_kfs)
                
                # Fallback to monocular depth where MVS is invalid
                mono_depth = frame.depth[sampled_uv[:, 1].int(), sampled_uv[:, 0].int()]
                depth[~valid_mvs] = mono_depth[~valid_mvs]
            else:
                depth = frame.depth[sampled_uv[:, 1].int(), sampled_uv[:, 0].int()]

            if self.n_active_gaussians > 0:
                main_gaussians_map = render_pkg["mainGaussID"]
                # Simplified pruning logic
                ids, counts = torch.unique(
                    main_gaussians_map[main_gaussians_map >= 0],
                    return_counts=True,
                )
                valid_gs_mask = torch.ones_like(self.xyz[:, 0], dtype=torch.bool)
                valid_gs_mask[ids.long()] = counts < 10

                self.optimizer.add_and_prune(
                    self.make_dummy_ext_tensor(), valid_gs_mask
                )
                render_pkg = self.render_from_frame(frame)
                rendered_depth = 1 / render_pkg["invdepth"][0].clamp_min(1e-8)

            if rendered_depth is not None:
                valid_mask = depth < rendered_depth[sample_mask]
                sample_mask[sample_mask.clone()] = valid_mask
                depth = depth[valid_mask]
                sampled_uv = sampled_uv[valid_mask]

            new_pts = depth2points(sampled_uv, depth.unsqueeze(-1), self.f, self.centre)
            new_pts = (new_pts - frame.get_t()) @ frame.get_R()

            f_dc = img[:, sample_mask]
            f_dc = RGB2SH(f_dc.permute(1, 0).unsqueeze(1))

            sampled_init_proba = init_proba[sample_mask]
            scales = 1 / (torch.sqrt(sampled_init_proba))
            scales.clamp_(1, self.width / 10)
            scales.mul_(1 / self.f)

            camera_approx_centre = frame.approx_centre[None]
            scales *= torch.linalg.vector_norm(new_pts - camera_approx_centre, dim=-1)
            scales = torch.log(scales.clamp(1e-6, 1e6)).unsqueeze(-1).repeat(1, 3)

            opacities = 0.1 * torch.ones(f_dc.shape[0], 1, device="cuda")
            opacities = inverse_sigmoid(opacities)

            f_rest = torch.zeros(f_dc.shape[0], (self.max_sh_degree + 1)**2 - 1, 3, device="cuda")
            rots = torch.zeros(f_dc.shape[0], 4, device="cuda")
            rots[:, 0] = 1

            if self.n_active_gaussians > 0:
                valid_gs_mask = self.opacity[:, 0] > 0.05
                dist = torch.linalg.vector_norm(self.xyz - camera_approx_centre, dim=-1)
                screen_size = self.f * self.scaling.max(dim=-1)[0] / dist
                valid_gs_mask *= screen_size < 0.5 * self.width
            else:
                valid_gs_mask = torch.ones(0, device="cuda", dtype=torch.bool)

            extension_tensors = {
                "xyz": new_pts,
                "f_dc": f_dc,
                "f_rest": f_rest,
                "opacity": opacities,
                "scaling": scales,
                "rotation": rots,
            }
            self.optimizer.add_and_prune(extension_tensors, valid_gs_mask)

    def check_visible(self, view_mat):
        with torch.no_grad():
            render_pkg = self.render(self.width, self.height, view_mat)
        if render_pkg["visibility_filter"].numel() == 0:
            return 0.0
        return render_pkg["visibility_filter"].sum().item() / render_pkg["visibility_filter"].numel()

    def check_covisible(self, view_mat1, view_mat2):
        with torch.no_grad():
            render_pkg1 = self.render(self.width, self.height, view_mat1)
            mainGaussID1 = torch.unique(render_pkg1["mainGaussID"][render_pkg1["mainGaussID"] >= 0]).long()
            render_pkg2 = self.render(self.width, self.height, view_mat2, GuassianID=mainGaussID1)
        if render_pkg2["visibility_filter"].numel() == 0:
            return 0.0
        return render_pkg2["visibility_filter"].sum().item() / render_pkg2["visibility_filter"].numel()

    def check_iou(self, view_mat1, view_mat2):
        # Implementation from backup scene_model.py
        with torch.no_grad():
            render_pkg1 = self.render(self.width, self.height, view_mat1)
            render_pkg2 = self.render(self.width, self.height, view_mat2)
        GS_ID_1 = set(torch.unique(render_pkg1["mainGaussID"][render_pkg1["mainGaussID"] >= 0]).tolist())
        GS_ID_2 = set(torch.unique(render_pkg2["mainGaussID"][render_pkg2["mainGaussID"] >= 0]).tolist())
        intersection = GS_ID_1.intersection(GS_ID_2)
        union = GS_ID_1.union(GS_ID_2)
        if len(union) == 0:
            return 0.0
        return len(intersection) / len(union)

    def check_oc(self, view_mat1, view_mat2):
        # Implementation from backup scene_model.py
        with torch.no_grad():
            render_pkg1 = self.render(self.width, self.height, view_mat1)
            render_pkg2 = self.render(self.width, self.height, view_mat2)
        GS_ID_1 = set(torch.unique(render_pkg1["mainGaussID"][render_pkg1["mainGaussID"] >= 0]).tolist())
        GS_ID_2 = set(torch.unique(render_pkg2["mainGaussID"][render_pkg2["mainGaussID"] >= 0]).tolist())
        intersection = GS_ID_1.intersection(GS_ID_2)
        factor = min(len(GS_ID_1), len(GS_ID_2))
        if factor == 0:
            return 0.0
        return len(intersection) / factor

    def render(self, width, height, view_matrix, scaling_modifier=1.0, bg=torch.zeros(3),
               fov_x=None, fov_y=None, GuassianID=None):
        
        device = "cuda"
        cam_centre = view_matrix.detach().inverse()[:3, 3]
        
        # Handle Anchor blending if enabled
        blended_ids = None
        if self.use_anchors and len(self.anchors) > 1:
            params, weights, blended_ids = Anchor.blend(cam_centre, self.anchors, self.anchor_overlap)
            xyz = params["xyz"]["val"]
            f_dc = params["f_dc"]["val"]
            f_rest = params["f_rest"]["val"]
            scaling = torch.exp(params["scaling"]["val"])
            opacity = torch.sigmoid(params["opacity"]["val"])
            rotation = F.normalize(params["rotation"]["val"])
        else:
            xyz = self.xyz
            f_dc = self.f_dc
            f_rest = self.f_rest
            scaling = self.scaling
            opacity = self.opacity
            rotation = self.rotation

        if GuassianID is not None:
            xyz = xyz[GuassianID]
            f_dc = f_dc[GuassianID]
            f_rest = f_rest[GuassianID]
            scaling = scaling[GuassianID]
            opacity = opacity[GuassianID]
            rotation = rotation[GuassianID]

        if fov_x is None:
            fov_x, fov_y = self.FoVx, self.FoVy

        fx = width / (2 * math.tan(fov_x * 0.5))
        fy = height / (2 * math.tan(fov_y * 0.5))
        cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
        K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device=device)

        viewmats = view_matrix.unsqueeze(0).contiguous()
        Ks = K.unsqueeze(0).contiguous()

        if xyz.shape[0] > 0:
            shs = torch.cat([f_dc, f_rest], dim=1)

            # gsplat rasterization
            render_colors_depth, render_alphas, meta = rasterization(
                means=xyz, quats=rotation, scales=scaling,
                opacities=opacity.squeeze(-1), colors=shs,
                viewmats=viewmats, Ks=Ks, width=width, height=height,
                near_plane=0.01, far_plane=100.0,
                sh_degree=self.active_sh_degree, render_mode="RGB+ED"
            )

            color = render_colors_depth[0, ..., :3].permute(2, 0, 1)
            depth = render_colors_depth[0, ..., 3:4].permute(2, 0, 1)
            invdepth = 1.0 / depth.clamp(min=1e-8)
            
            isect_radii = meta["radii"]
            if isect_radii.ndim == 2:
                isect_radii = isect_radii.max(dim=-1)[0]
            
            radii = torch.zeros(xyz.shape[0], device=device, dtype=torch.int32)
            radii.scatter_add_(0, meta["gaussian_ids"].long(), isect_radii)
            
            mainGaussID = torch.full((1, height, width), -1, device=device, dtype=torch.int32)
            
            screenspace_points = meta["means2d"][0]
        else:
            color = torch.zeros(3, height, width, device=device)
            invdepth = torch.zeros(1, height, width, device=device)
            mainGaussID = torch.full((1, height, width), -1, device=device, dtype=torch.int32)
            radii = torch.zeros(0, device=device)
            screenspace_points = torch.zeros(0, 2, device=device)

        return {
            "render": color,
            "invdepth": invdepth,
            "mainGaussID": mainGaussID,
            "radii": radii,
            "visibility_filter": radii > 0,
            "screenspace_points": screenspace_points,
        }

    def render_from_frame(self, frame, pyr_lvl=0, scaling_modifier=1, bg=torch.zeros(3)):
        scale = 2**pyr_lvl
        width, height = self.width // scale, self.height // scale
        fov_x, fov_y = frame.get_fov()
        view_matrix = frame.get_Rt()
        render_pkg = self.render(width, height, view_matrix, scaling_modifier, bg=bg.to("cuda"), fov_x=fov_x, fov_y=fov_y)
        render_pkg["render"] = render_pkg["render"].clamp(0, 1).view(3, height, width)
        return render_pkg

    def init_intrinsics(self, fx, fy):
        self.FoVx = focal2fov(fx, self.width)
        self.FoVy = focal2fov(fy, self.height)
        self.tanfovx = math.tan(self.FoVx * 0.5)
        self.tanfovy = math.tan(self.FoVy * 0.5)
        self.projection_matrix = getProjectionMatrix(0.01, 100.0, self.FoVx, self.FoVy).transpose(0, 1).cuda()

    def process_key_frame(self, image, depth, extrinsic4x4, K, mask=None):
        feat_map = None
        mono_idepth = None
        mono_depth_conf = None
        
        if self.use_guided_mvs:
            feat_map = self.dense_extractor(image)
            # Guided MVS expects mono_idepth as 1/depth
            mono_idepth = 1.0 / depth.clamp(min=1e-6)
            # Add dummy confidence if needed, or use a constant
            mono_depth_conf = torch.ones_like(mono_idepth)
            
        return Keyframe(image, depth, extrinsic4x4, K, mask, 
                        pyr_levels=self.pyr_levels, use_exposure=self.use_exposure,
                        feat_map=feat_map, mono_idepth=mono_idepth, mono_depth_conf=mono_depth_conf)

    def update(self, image, depth, extrinsic4x4, K, mask=None):
        image = torch.from_numpy(image.transpose(2, 0, 1)).float().cuda() / 255.0
        depth = torch.from_numpy(depth).float().cuda()
        extrinsic4x4 = torch.from_numpy(extrinsic4x4).float().cuda()
        if mask is not None: mask = torch.from_numpy(mask).bool().cuda()
        
        self.f = (K[0, 0] + K[1, 1]) / 2.0
        self.init_intrinsics(K[0, 0], K[1, 1])

        keyframe = self.process_key_frame(image, depth, extrinsic4x4, K, mask)
        self.add_new_gaussians(keyframe)
        
        # Track which keyframe belongs to which anchor
        kf_idx = len(self.keyframes)
        self.keyframes.append(keyframe)
        if self.use_anchors and self.active_anchor:
            self.active_anchor.add_keyframe_id(kf_idx)
            
        self.optimization_loop(self.num_steps)

    def save(self, path: str):
        data = {"gaussian_params": self.gaussian_params}
        if self.use_anchors:
            data["anchors"] = [a.gaussian_params for a in self.anchors]
        torch.save(data, path)

    def load(self, path: str):
        data = torch.load(path)
        self.gaussian_params = data["gaussian_params"]
        if "anchors" in data and self.use_anchors:
            pass
