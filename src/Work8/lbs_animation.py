"""
LBS 姿态动画 —— 选做部分
固定 shape 参数，让左肘关节从 0 逐渐旋转到目标角度，导出 GIF。
用法：
    python lbs_animation.py --model-dir ./models --out-dir ./outputs --frames 30
"""

import os
import sys
import types
import argparse
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio.v2 as imageio
import io

import smplx
from smplx.lbs import (
    blend_shapes,
    vertices2joints,
    batch_rodrigues,
    batch_rigid_transform,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ──────────────────────────────────────────────
# chumpy 兼容 shim（与必做部分相同）
# ──────────────────────────────────────────────
class _ChumpyArrayShim:
    def __setstate__(self, state):
        self.__dict__.update(state)

    def _array(self):
        if hasattr(self, "r"):
            return self.r
        if hasattr(self, "x"):
            return self.x
        raise AttributeError("Cannot recover array data from chumpy pickle object")

    def __array__(self, dtype=None):
        return np.asarray(self._array(), dtype=dtype)

    @property
    def shape(self):
        return np.asarray(self).shape

    def __len__(self):
        return len(np.asarray(self))

    def __getitem__(self, item):
        return np.asarray(self)[item]


def install_chumpy_pickle_shim():
    if "chumpy.ch" in sys.modules:
        return
    chumpy_module = types.ModuleType("chumpy")
    chumpy_ch_module = types.ModuleType("chumpy.ch")
    _ChumpyArrayShim.__name__ = "Ch"
    _ChumpyArrayShim.__qualname__ = "Ch"
    _ChumpyArrayShim.__module__ = "chumpy.ch"
    chumpy_ch_module.Ch = _ChumpyArrayShim
    chumpy_module.ch = chumpy_ch_module
    sys.modules["chumpy"] = chumpy_module
    sys.modules["chumpy.ch"] = chumpy_ch_module


# ──────────────────────────────────────────────
# 工具函数（与必做部分相同）
# ──────────────────────────────────────────────
def resolve_script_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(SCRIPT_DIR, path)


def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def set_axes_equal(ax, vertices):
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = 0.5 * np.max(maxs - mins + 1e-8)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def smpl_to_plot_coords(points):
    return points[:, [0, 2, 1]]


def shade_face_colors(vertices, faces, face_colors):
    triangles = vertices[faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0]
    )
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
    light_dir = np.array([-0.25, -0.55, 0.80], dtype=np.float64)
    light_dir /= np.linalg.norm(light_dir)
    intensity = 0.35 + 0.65 * np.clip(normals @ light_dir, 0.0, 1.0)
    shaded = face_colors.copy()
    shaded[:, :3] *= intensity[:, None]
    return shaded


def prepare_posedirs(posedirs, expected_pose_dim):
    if posedirs.dim() != 2:
        posedirs = posedirs.reshape(posedirs.shape[0], -1)
    if posedirs.shape[0] == expected_pose_dim:
        return posedirs
    if posedirs.shape[1] == expected_pose_dim:
        return posedirs.T
    raise RuntimeError(
        f"posedirs 形状与 pose_feature 不匹配，"
        f"posedirs.shape={tuple(posedirs.shape)}, "
        f"expected_pose_dim={expected_pose_dim}"
    )


# ──────────────────────────────────────────────
# 手写 LBS（与必做部分相同，返回最终顶点和变换后关节）
# ──────────────────────────────────────────────
def compute_lbs(model, betas, global_orient, body_pose):
    device = betas.device
    dtype = betas.dtype

    v_template = model.v_template
    if v_template.dim() == 2:
        v_template = v_template.unsqueeze(0)

    shapedirs = model.shapedirs[:, :, :betas.shape[1]]
    v_shaped = v_template + blend_shapes(betas, shapedirs)
    J = vertices2joints(model.J_regressor, v_shaped)

    full_pose = torch.cat([global_orient, body_pose], dim=1)
    rot_mats = batch_rodrigues(full_pose.view(-1, 3)).view(1, -1, 3, 3)
    ident = torch.eye(3, dtype=dtype, device=device)
    pose_feature = (rot_mats[:, 1:, :, :] - ident).view(1, -1)
    posedirs = prepare_posedirs(model.posedirs, expected_pose_dim=pose_feature.shape[1])
    pose_offsets = torch.matmul(pose_feature, posedirs).view(1, -1, 3)
    v_posed = v_shaped + pose_offsets

    J_transformed, A = batch_rigid_transform(rot_mats, J, model.parents, dtype=dtype)

    num_joints = J.shape[1]
    W = model.lbs_weights.unsqueeze(0).expand(1, -1, -1)
    T = torch.matmul(W, A.view(1, num_joints, 16)).view(1, -1, 4, 4)
    homogen_coord = torch.ones((1, v_posed.shape[1], 1), dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, v_posed_homo.unsqueeze(-1))
    verts = v_homo[:, :, :3, 0]

    return to_numpy(verts[0]), to_numpy(J_transformed[0])


# ──────────────────────────────────────────────
# 单帧渲染：左侧为最终姿态网格，右侧为权重热力图
# ──────────────────────────────────────────────
def render_frame(model, betas, global_orient, body_pose, faces,
                 lbs_weights_np, joint_id, angle_deg, frame_idx, total_frames):
    verts, joints = compute_lbs(model, betas, global_orient, body_pose)
    plot_verts = smpl_to_plot_coords(verts)
    plot_joints = smpl_to_plot_coords(joints)

    # 左图：最终姿态，皮肤色
    skin_color = np.tile(np.array([[0.82, 0.67, 0.52, 1.0]]), (faces.shape[0], 1))
    skin_shaded = shade_face_colors(plot_verts, faces, skin_color)

    # 右图：权重热力图（joint_id 关节对每个顶点的权重）
    weight_scalar = lbs_weights_np[:, joint_id]
    w = weight_scalar.astype(np.float64)
    w = (w - w.min()) / (w.max() - w.min() + 1e-8)
    face_w = w[faces].mean(axis=1)
    cmap = plt.get_cmap("plasma")
    weight_colors = cmap(face_w)
    weight_shaded = shade_face_colors(plot_verts, faces, weight_colors)

    fig = plt.figure(figsize=(10, 5.5), facecolor="#1a1a2e")
    fig.suptitle(
        f"LBS Animation  —  Left Elbow  |  Frame {frame_idx+1}/{total_frames}  "
        f"|  Angle = {angle_deg:.1f}°",
        color="white", fontsize=11, y=0.97
    )

    for col, (fc, subtitle) in enumerate([
        (skin_shaded,   "Final Skinned Mesh"),
        (weight_shaded, f"LBS Weight  (joint {joint_id})"),
    ]):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
        ax.set_facecolor("#1a1a2e")
        mesh = Poly3DCollection(
            plot_verts[faces],
            facecolors=fc,
            linewidths=0.02,
            edgecolors=(0, 0, 0, 0.04),
        )
        ax.add_collection3d(mesh)
        ax.scatter(
            plot_joints[:, 0], plot_joints[:, 1], plot_joints[:, 2],
            c="white", s=10, depthshade=False,
            edgecolors="#aaaaaa", linewidths=0.3
        )
        set_axes_equal(ax, plot_verts)
        ax.set_proj_type("persp", focal_length=0.85)
        ax.view_init(elev=12, azim=108)
        ax.set_axis_off()
        ax.set_title(subtitle, color="white", fontsize=9, pad=4)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    img = imageio.imread(buf)
    return img


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────
def main(args):
    device = torch.device("cpu")
    dtype = torch.float32

    model_dir = resolve_script_path(args.model_dir)
    out_dir = resolve_script_path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # 加载模型
    install_chumpy_pickle_shim()
    model = smplx.create(
        model_path=model_dir,
        model_type="smpl",
        gender="neutral",
        ext="pkl",
        num_betas=args.num_betas,
    ).to(device)

    faces = np.asarray(model.faces, dtype=np.int32)
    lbs_weights_np = to_numpy(model.lbs_weights)

    # 固定 shape 参数（与必做部分一致）
    betas = torch.zeros((1, args.num_betas), dtype=dtype, device=device)
    betas[0, 0] = 2.0
    betas[0, 1] = -1.2
    betas[0, 2] = 0.8

    # 固定基础姿态（肩膀略展开，方便看到肘部运动）
    global_orient = torch.zeros((1, 3), dtype=dtype, device=device)

    # left_elbow = joint 18，其 body_pose 索引 = (18-1)*3 = 51
    # 动画：绕 Y 轴从 0° → target_deg → 0°（来回一次）
    target_rad = float(np.radians(args.target_angle))
    n = args.frames

    # 生成角度序列：0 → target → 0（平滑往返，使 GIF 循环自然）
    half = n // 2
    angles_go   = np.linspace(0, target_rad, half, endpoint=False)
    angles_back = np.linspace(target_rad, 0, n - half, endpoint=True)
    angles = np.concatenate([angles_go, angles_back])

    print(f"共生成 {n} 帧，关节 {args.joint_id}，"
          f"目标角度 {args.target_angle}°，导出到 {out_dir}")

    frames_imgs = []
    for i, angle in enumerate(angles):
        body_pose = torch.zeros((1, 23 * 3), dtype=dtype, device=device)

        # 左肩略展开（固定）
        body_pose[0, (16-1)*3 + 2] =  0.45   # left_shoulder z
        body_pose[0, (17-1)*3 + 2] = -0.45   # right_shoulder z
        body_pose[0, (19-1)*3 + 1] =  0.35   # right_elbow y（固定）

        # 左肘绕 Y 轴旋转（动画变量）
        body_pose[0, (18-1)*3 + 1] = -float(angle)  # left_elbow y

        img = render_frame(
            model, betas, global_orient, body_pose,
            faces, lbs_weights_np,
            joint_id=args.joint_id,
            angle_deg=np.degrees(angle),
            frame_idx=i,
            total_frames=n,
        )
        frames_imgs.append(img)
        print(f"  帧 {i+1:03d}/{n}  angle={np.degrees(angle):.1f}°", end="\r")

    print()

    # 导出 GIF
    gif_path = os.path.join(out_dir, "lbs_animation.gif")
    imageio.mimsave(gif_path, frames_imgs, fps=args.fps, loop=0)
    print(f"GIF 已保存：{gif_path}")

    # 同时把第一帧和中间帧保存为 PNG 供报告使用
    for idx, label in [(0, "anim_frame_start"),
                       (half - 1, "anim_frame_mid"),
                       (n - 1, "anim_frame_end")]:
        png_path = os.path.join(out_dir, f"{label}.png")
        imageio.imwrite(png_path, frames_imgs[idx])
    print("关键帧 PNG 已保存。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LBS 姿态动画（选做）")
    parser.add_argument("--model-dir",     type=str,   default="./models")
    parser.add_argument("--out-dir",       type=str,   default="./outputs")
    parser.add_argument("--frames",        type=int,   default=36,
                        help="总帧数（建议 24-48）")
    parser.add_argument("--fps",           type=int,   default=12,
                        help="GIF 帧率")
    parser.add_argument("--target-angle",  type=float, default=110.0,
                        help="左肘最大旋转角度（度）")
    parser.add_argument("--joint-id",      type=int,   default=18,
                        help="右侧面板显示权重的关节编号（默认 18 = 左肘）")
    parser.add_argument("--num-betas",     type=int,   default=10)
    args = parser.parse_args()
    main(args)