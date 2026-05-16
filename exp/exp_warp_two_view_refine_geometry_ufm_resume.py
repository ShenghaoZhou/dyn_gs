import numpy as np
import rerun as rr
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import io
from PIL import Image
from types import SimpleNamespace

import tyro

# Reuse components from the original script logic
from exp_warp_two_view_refine_geometry_ufm import (
    Config, 
    gs_to_planar_params, 
    render_custom_attribute, 
    optimize_patch_geometry_ufm, 
    manual_forward_homography_warp,
    depth_to_rgb,
    unproject_depth
)
import rerun.blueprint as rrb

def setup_blueprint_resume():
    blueprint = rrb.Blueprint(
        rrb.Tabs(
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.Spatial2DView(name="Init Image (Ref)", contents=["init/image", "init/matches"]),
                    rrb.Spatial2DView(name="Matching Lines (UFM)", contents=["match_vis/**"]),
                    rrb.Spatial2DView(name="Matching Lines (Refined)", contents=["match_vis_refined/**"]),
                    rrb.Spatial2DView(name="GT Image (Target)", contents=["gt/image", "gt/matches"]),
                ),
                rrb.Horizontal(
                    rrb.Spatial2DView(name="Initial Normal", contents=["gs_init/normal"]),
                    rrb.Spatial2DView(name="Refined Normal", contents=["gs_refined/normal"]),
                    rrb.Spatial2DView(name="Warped Refined", contents=["warped/refined"]),
                ),
                rrb.Horizontal(
                    rrb.Spatial2DView(name="Initial Depth", contents=["gs_init/depth"]),
                    rrb.Spatial2DView(name="Refined Depth", contents=["gs_refined/depth"]),
                    rrb.Spatial2DView(name="Refine Loss", contents=["opt/refine_loss_plot"]),
                ),
                name="Refinement Dashboard"
            ),
            rrb.Spatial3DView(
                name="3D Reality",
                contents=["world/**"]
            )
        )
    )
    return blueprint

def main():
    # 1. Load Cache and Config
    cache_path = Path("ufm_refine_cache.pth")
    if not cache_path.exists():
        print(f"Error: Cache file {cache_path} not found. Run the Part 1 script first.")
        return

    print(f"Loading cache from {cache_path}...")
    cache = torch.load(cache_path, weights_only=False)
    cfg_cached = cache["cfg"]

    # 2. Allow command-line overrides using the cached config as default
    cfg = tyro.cli(Config, default=cfg_cached)
    device = torch.device(cfg.device)
    
    # Reconstruct gs_params as a SimpleNamespace
    gs_params = SimpleNamespace(**{k: v.to(device) for k, v in cache["gs_params"].items()})
    
    # Override sampling for denser refinement if needed
    cfg.sample_patches = 20000 
    T_C_O_ref_torch = cache["T_C_O_ref"].to(device)
    T_C_O_est = cache["T_C_O_est"].to(device)
    flow_ufm = cache["flow_ufm"].to(device)
    covis_ufm = cache["covis_ufm"].to(device)
    image_init = cache["image_init"]
    image_target_gt = cache["image_target_gt"]
    flow_vis_image = cache.get("flow_vis_image")
    T_WC_target_gt = cache.get("T_WC_target_gt")
    T_WO_target_gt = cache.get("T_WO_target_gt")
    K_torch = cache["K"].to(device)

    # 1.5. Overwrite Estimation with GT Pose if requested
    if cfg.use_gt_pose:
        if T_WC_target_gt is not None and T_WO_target_gt is not None:
            T_WC_gt = torch.from_numpy(T_WC_target_gt).float().to(device)
            T_WO_gt = torch.from_numpy(T_WO_target_gt).float().to(device)
            T_C_O_est = torch.inverse(T_WC_gt) @ T_WO_gt
            print("[Pose] Using GROUND TRUTH camera-to-object pose for refinement.")
        else:
            print("[Warning] GT Pose requested but T_WC_target_gt or T_WO_target_gt missing in cache.")
    else:
        print("[Pose] Using ESTIMATED camera-to-object pose from Part 1.")
    
    H, W = image_target_gt.shape[:2]
    image_ref_t = torch.from_numpy(image_init).float().to(device).permute(2, 0, 1) / 255.0

    # 2. Re-initialize Rerun
    rr.init("exp_warp_two_view_refine_geometry_ufm_resume", spawn=False)
    rr.connect_grpc("rerun+http://128.175.109.248:9876/proxy")
    rr.send_blueprint(setup_blueprint_resume())

    # Log basic data for visualization
    rr.log("init/image", rr.Image(image_init))
    rr.log("gt/image", rr.Image(image_target_gt))
    if flow_vis_image is not None:
        rr.log("ufm/flow", rr.Image(flow_vis_image))

    if T_WC_target_gt is not None:
        rr.log("world/camera", rr.Pinhole(image_from_camera=K_torch.cpu().numpy(), width=W, height=H))
        rr.log("world/camera", rr.Transform3D(mat3x3=T_WC_target_gt[:3, :3], translation=T_WC_target_gt[:3, 3]))

    # --- Sample Matchings ---
    # Sample points in object area (where alpha is high or just random points if mask is available)
    # Since we have alpha_ref_map, we can use it to find the object area.
    with torch.no_grad():
        n_ref, d_ref = gs_to_planar_params(gs_params, T_C_O_ref_torch)
        _, alpha_ref_map = render_custom_attribute(gs_params, n_ref, T_C_O_ref_torch, K_torch, W, H, cfg.near_plane, cfg.far_plane)
        mask = (alpha_ref_map[0] > 0.5).cpu().numpy()
        
    y, x = np.where(mask)
    if len(x) > 0:
        num_matches = 50
        indices = np.random.choice(len(x), min(len(x), num_matches), replace=False)
        pts_ref = np.stack([x[indices], y[indices]], axis=1) # [N, 2]
        
        # Get target points using flow
        # flow_ufm: [2, H, W]
        pts_ref_torch = torch.from_numpy(pts_ref).float().to(device)
        pts_ref_norm = pts_ref_torch.clone()
        pts_ref_norm[:, 0] = 2.0 * pts_ref_norm[:, 0] / (W - 1) - 1.0
        pts_ref_norm[:, 1] = 2.0 * pts_ref_norm[:, 1] / (H - 1) - 1.0
        
        sampled_flow = F.grid_sample(
            flow_ufm.unsqueeze(0),
            pts_ref_norm.view(1, -1, 1, 2),
            align_corners=True
        ).reshape(2, -1).t().cpu().numpy() # [N, 2]
        
        pts_target = pts_ref + sampled_flow
        
        # Generate colors for matches
        colors = plt.get_cmap("hsv")(np.linspace(0, 1, len(pts_ref)))[:, :3]
        colors = (colors * 255).astype(np.uint8)
        
        # 1. Log to individual images as before
        rr.log("init/matches", rr.Points2D(pts_ref, colors=colors, radii=2))
        rr.log("gt/matches", rr.Points2D(pts_target, colors=colors, radii=2))

        # 2. Side-by-side matching lines
        # Log images with offset
        rr.log("match_vis/ref", rr.Image(image_init))
        rr.log("match_vis/tgt", rr.Image(image_target_gt))
        rr.log("match_vis/tgt", rr.Transform3D(translation=[W, 0, 0]))

        # Create line strips [N, 2, 2]
        pts_tgt_offset = pts_target + np.array([W, 0])
        strips = np.stack([pts_ref, pts_tgt_offset], axis=1) # [N, 2, 2]
        
        rr.log("match_vis/lines", rr.LineStrips2D(strips, colors=colors, radii=0.5))
        print(f"[Matchings] Logged {len(pts_ref)} UFM correspondences with connecting lines.")

    # 3. Resume from Geometry Refinement
    print("\n[Part 2] Resuming Geometry Refinement with UFM Flow...")
    
    # We need initial normal/distance maps at ref frame for the patches
    with torch.no_grad():
        n_ref, d_ref = gs_to_planar_params(gs_params, T_C_O_ref_torch)
        n_ref_map, alpha_ref_map = render_custom_attribute(gs_params, n_ref, T_C_O_ref_torch, K_torch, W, H, cfg.near_plane, cfg.far_plane)
        d_ref_3ch_map, _ = render_custom_attribute(gs_params, d_ref.repeat(1, 3), T_C_O_ref_torch, K_torch, W, H, cfg.near_plane, cfg.far_plane)
        d_ref_map = d_ref_3ch_map[0:1]
        print(f"[Debug] Alpha map max: {alpha_ref_map.max().item():.4f}")
        
    # Log Initial Geometry
    normal_vis_init = (n_ref_map.permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0
    rr.log("gs_init/normal", rr.Image((normal_vis_init.clip(0, 1) * 255).astype(np.uint8)))
    depth_vis_init = depth_to_rgb(d_ref_map[0].cpu().numpy())
    rr.log("gs_init/depth", rr.Image(depth_vis_init))

    n_refined_map, d_refined_map, refine_losses = optimize_patch_geometry_ufm(
        image_ref_t, None, n_ref_map, d_ref_map, alpha_ref_map, T_C_O_ref_torch, T_C_O_est, K_torch, flow_ufm, covis_ufm, cfg
    )
    
    # Log refinement loss plot
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(refine_losses)
    ax.set_title("Geometry Refinement Loss (UFM Flow MSE) - Resumed")
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    rr.log("opt/refine_loss_plot", rr.Image(np.array(Image.open(buf))))
    plt.close(fig)
    
    # 4. Final Refined Visualization
    print("\nGenerating final visualization...")
    with torch.no_grad():
        mask_ref = alpha_ref_map > 0.5
        n_refined_flat = n_refined_map.permute(1, 2, 0).reshape(-1, 3)
        d_refined_flat = d_refined_map.reshape(-1, 1)
        
        T_curr_ref = T_C_O_est @ torch.inverse(T_C_O_ref_torch)
        warped_refined = manual_forward_homography_warp(image_ref_t, n_refined_flat, d_refined_flat, T_curr_ref, K_torch, mask=mask_ref)
        
        # Alpha at optimized pose
        _, alpha_opt = render_custom_attribute(gs_params, n_ref, T_C_O_est, K_torch, W, H, cfg.near_plane, cfg.far_plane)
        warped_refined = warped_refined * alpha_opt

        # --- 3D Visualization ---
        print(f"\n[Part 2] Generating 3D results...")
        # 1. Object Frame Visualization (Always reliable)
        # Transform Camera points to Object frame: P_O = inv(T_CO) @ P_C
        T_OC_est = torch.inverse(T_C_O_est)
        
        # Refined
        pts_C_gs_refined = unproject_depth(d_refined_map, K_torch, H, W).reshape(-1, 3)
        valid_gs = (d_refined_map.reshape(-1) > 0.01) & (alpha_opt.reshape(-1) > 0.5)
        pts_O_gs_refined = (T_OC_est[:3, :3] @ pts_C_gs_refined.t()).t() + T_OC_est[:3, 3]
        
        # Initial (using optimized pose for comparison)
        n_init_tmp, d_init_tmp = gs_to_planar_params(gs_params, T_C_O_est)
        d_init_map_tmp_3ch, alpha_init_tmp = render_custom_attribute(gs_params, d_init_tmp.repeat(1,3), T_C_O_est, K_torch, W, H, cfg.near_plane, cfg.far_plane)
        d_init_map_tmp = d_init_map_tmp_3ch[0:1]
        pts_C_gs_init = unproject_depth(d_init_map_tmp, K_torch, H, W).reshape(-1, 3)
        pts_O_gs_init = (T_OC_est[:3, :3] @ pts_C_gs_init.t()).t() + T_OC_est[:3, 3]

        rr.log("world/object_gs_init", rr.Points3D(pts_O_gs_init[valid_gs].cpu().numpy(), radii=0.001, colors=[200, 50, 50]))
        rr.log("world/object_gs_refined", rr.Points3D(pts_O_gs_refined[valid_gs].cpu().numpy(), radii=0.001, colors=[50, 200, 50]))
        print(f"[3D] Logged {valid_gs.sum().item()} points in Object Frame.")

        # 2. World Frame Visualization (Depends on cache)
        if T_WO_target_gt is not None:
            T_WO = torch.from_numpy(T_WO_target_gt).float().to(device)
            pts_W_gs_init = (T_WO[:3, :3] @ pts_O_gs_init.t()).t() + T_WO[:3, 3]
            pts_W_gs_refined = (T_WO[:3, :3] @ pts_O_gs_refined.t()).t() + T_WO[:3, 3]
            
            rr.log("world/gs_init_pc", rr.Points3D(pts_W_gs_init[valid_gs].cpu().numpy(), radii=0.001, colors=[200, 50, 50]))
            rr.log("world/gs_refined_pc", rr.Points3D(pts_W_gs_refined[valid_gs].cpu().numpy(), radii=0.001, colors=[50, 200, 50]))
            
            # Means
            gs_pts_O = gs_params.means
            pts_W_gs_orig = (T_WO[:3, :3] @ gs_pts_O.t()).t() + T_WO[:3, 3]
            rr.log("world/gs_orig_means", rr.Points3D(pts_W_gs_orig.cpu().numpy(), radii=0.001, colors=[50, 50, 200]))

        # GT Point Cloud
        pts_W_gt = cache.get("pts_W_gt")
        colors_gt = cache.get("colors_gt")
        if pts_W_gt is not None:
            rr.log("world/object_gt", rr.Points3D(pts_W_gt, colors=colors_gt, radii=0.002))

    warped_np = (warped_refined.permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    rr.log("warped/refined", rr.Image(warped_np))
    
    # Log refined normals and depth
    normal_vis_refined = (n_refined_map.permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0
    rr.log("gs_refined/normal", rr.Image((normal_vis_refined.clip(0, 1) * 255).astype(np.uint8)))
    depth_vis_refined = depth_to_rgb(d_refined_map[0].cpu().numpy())
    rr.log("gs_refined/depth", rr.Image(depth_vis_refined))

    # --- Recovered Match (After Optimization) ---
    if 'pts_ref' in locals():
        print("[Matches] Computing refined correspondences...")
        with torch.no_grad():
            # Get optimized n and d for the sampled ref points
            n_samp = n_refined_map[:, pts_ref[:, 1], pts_ref[:, 0]].t() # [N, 3]
            d_samp = d_refined_map[0, pts_ref[:, 1], pts_ref[:, 0]] # [N]
            
            # Homography for these points
            R_rel = T_curr_ref[:3, :3]
            t_rel = T_curr_ref[:3, 3]
            K_inv = torch.inverse(K_torch)
            
            # x_target = K @ (R - t*n^T / d) @ K^-1 @ x_ref
            # We can use our patch_warp logic for single pixels
            uv_ref = torch.from_numpy(pts_ref).float().to(device).unsqueeze(0) # [1, N, 2]
            
            # Single homography per point
            tnT_d = torch.matmul(t_rel.view(3, 1), n_samp.unsqueeze(1)) / d_samp.view(-1, 1, 1).clamp(min=1e-2) # [N, 3, 3]
            H_per_pt = R_rel.unsqueeze(0) - tnT_d
            H_full = K_torch @ H_per_pt @ K_inv
            
            # Project uv_ref [1, N, 2] -> homo_uv [N, 3, 1]
            homo_uv = torch.cat([uv_ref[0], torch.ones_like(uv_ref[0][:, :1])], dim=-1).unsqueeze(-1) # [N, 3, 1]
            pts_target_refined_homo = torch.matmul(H_full, homo_uv).squeeze(-1) # [N, 3]
            pts_target_refined = (pts_target_refined_homo[:, :2] / pts_target_refined_homo[:, 2:]).cpu().numpy()
            
            # Log refined match visualization
            rr.log("match_vis_refined/ref", rr.Image(image_init))
            rr.log("match_vis_refined/tgt", rr.Image(image_target_gt))
            rr.log("match_vis_refined/tgt", rr.Transform3D(translation=[W, 0, 0]))
            
            pts_tgt_ref_offset = pts_target_refined + np.array([W, 0])
            strips_ref = np.stack([pts_ref, pts_tgt_ref_offset], axis=1)
            
            # Draw with a different color (e.g. green) or keep colors to match
            rr.log("match_vis_refined/lines", rr.LineStrips2D(strips_ref, colors=colors, radii=0.5))
            print(f"[Matches] Logged {len(pts_ref)} refined matches.")

    print("Done! Check Rerurn for results.")

if __name__ == "__main__":
    main()
