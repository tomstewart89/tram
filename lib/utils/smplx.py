import numpy as np
import torch

from lib.vis.traj import *
from lib.models.smpl import SMPL
from lib.utils.rotation_conversions import matrix_to_axis_angle


def track_to_smplx(
    pred_trans: torch.Tensor,
    pred_rotmat: torch.Tensor,
    pred_shape: torch.Tensor,
    world_cam_R: torch.Tensor,
    world_cam_T: torch.Tensor,
):
    device = "cuda"
    smpl = SMPL().to(device)

    betas = pred_shape.mean(dim=0, keepdim=True).repeat(len(pred_shape), 1).to(device)
    translation = pred_trans.to(device).squeeze()
    rotation = pred_rotmat[:, [0]].to(device)
    body_pose = pred_rotmat[:, 1:].to(device)
    world_cam_R = world_cam_R.to(device)
    world_cam_T = world_cam_T.to(device)

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

    out = smpl(
        body_pose=body_pose_aa.reshape(-1, 69),
        global_orient=global_orient_aa,
        betas=betas,
        transl=translation_world,
        default_smpl=True,
    )

    # center x/y at zero and place lowest vertex at z=0
    offset = torch.zeros(3, device=device)
    offset[0] = out.joints[:, :, 0].mean()
    offset[1] = out.joints[:, :, 1].mean()
    offset[2] = out.joints[:, :, 2].min()
    translation_world -= offset

    return {
        "betas": torch.cat([betas[0].cpu(), torch.zeros(6)]).numpy(),
        "root_orient": global_orient_aa.cpu().numpy(),
        "pose_body": body_pose_aa.cpu().numpy()[:, :21],
        "trans": translation_world.cpu().numpy(),
        "gender": "neutral",
        "mocap_frame_rate": 30,
    }
