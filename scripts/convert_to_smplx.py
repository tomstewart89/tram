import sys
import os

sys.path.insert(0, os.path.dirname(__file__) + "/..")

import numpy as np
import numpy as np
import os
import torch

from lib.vis.traj import *
from lib.models.smpl import SMPL
from lib.utils.rotation_conversions import matrix_to_axis_angle

if __name__ == "__main__":

    device = "cuda"
    smpl = SMPL().to(device)

    pred_cam = np.load("results/output/camera.npy", allow_pickle=True).item()
    world_cam_R = torch.tensor(pred_cam["world_cam_R"]).to(device)
    world_cam_T = torch.tensor(pred_cam["world_cam_T"]).to(device)

    pred_smpl = np.load("results/output/hps/hps_track_0.npy", allow_pickle=True).item()

    pred_shape = pred_smpl["pred_shape"].to(device)
    betas = pred_shape.mean(dim=0, keepdim=True).repeat(len(pred_shape), 1)
    translation = pred_smpl["pred_trans"].to(device).squeeze()
    rotation = pred_smpl["pred_rotmat"][:, [0]].to(device)
    body_pose = pred_smpl["pred_rotmat"][:, 1:].to(device)

    out = smpl(
        body_pose=body_pose,
        global_orient=rotation,
        betas=betas,
        transl=translation,
        pose2rot=False,
        default_smpl=True,
    )

    vertices_cam = out.vertices
    vertices_world = torch.zeros_like(out.vertices)

    # transform to world frame from camera frame
    for i in range(world_cam_R.shape[0]):
        vertices_world[i] = (world_cam_R[i] @ out.vertices[i].T).T + world_cam_T[i]

    # compute root joint position (depends only on betas)
    v_shaped = smpl.v_template + torch.einsum("bl,vdl->bvd", betas, smpl.shapedirs)
    J_0 = torch.einsum("jv,bvd->bjd", smpl.J_regressor, v_shaped)[:, 0]  # [N, 3]

    # transform global orient and translation to world frame
    rotation_world = (world_cam_R @ rotation.squeeze(1)).unsqueeze(1)
    translation_world = (
        (world_cam_R @ (translation + J_0).unsqueeze(-1)).squeeze(-1)
        + world_cam_T
        - J_0
    )

    # rotate from y-up to z-up (-90 degrees around x-axis)
    R_y2z = torch.tensor(
        [[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=torch.float32, device=device
    )
    rotation_world = (R_y2z @ rotation_world.squeeze(1)).unsqueeze(1)
    translation_world = (R_y2z @ (translation_world + J_0).unsqueeze(-1)).squeeze(
        -1
    ) - J_0

    # convert rotation matrices to axis-angle
    global_orient_aa = matrix_to_axis_angle(rotation_world.squeeze(1))  # [N, 3]
    body_pose_aa = matrix_to_axis_angle(body_pose)  # [N, 23, 3]

    out_ = smpl(
        body_pose=body_pose_aa.reshape(-1, 69),
        global_orient=global_orient_aa,
        betas=betas,
        transl=translation_world,
        pose2rot=True,
        default_smpl=True,
    )

    vertices_world_ = out_.vertices
    joints_world_ = out_.joints

    # rotate reference vertices to z-up for comparison
    vertices_world_zup = (R_y2z @ vertices_world.permute(0, 2, 1)).permute(0, 2, 1)
    err = (vertices_world_zup - vertices_world_).abs()
    print(f"max error: {err.max().item():.6e}, mean error: {err.mean().item():.6e}")
    assert torch.allclose(vertices_world_zup, vertices_world_, atol=1e-3)

    # center x/y at zero and place lowest vertex at z=0
    offset = torch.zeros(3, device=device)
    offset[0] = joints_world_[:, :, 0].mean()
    offset[1] = joints_world_[:, :, 1].mean()
    offset[2] = joints_world_[:, :, 2].min()
    translation_world -= offset

    np.save(
        "results/output/smplx_params.npy",
        {
            "betas": torch.cat([betas[0].cpu(), torch.zeros(6)]).numpy(),
            "root_orient": global_orient_aa.cpu().numpy(),
            "pose_body": body_pose_aa.cpu().numpy()[:, :21],
            "trans": translation_world.cpu().numpy(),
            "gender": "neutral",
            "mocap_frame_rate": 30,
        },
        allow_pickle=True,
    )
