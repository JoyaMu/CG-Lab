import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

import pytorch3d
from pytorch3d.io import load_objs_as_meshes, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.renderer import (
    look_at_view_transform, FoVPerspectiveCameras,
    RasterizationSettings, MeshRenderer, MeshRasterizer,
    SoftPhongShader, SoftSilhouetteShader,
    BlendParams, PointLights, TexturesVertex
)
from pytorch3d.loss import mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency

# -------------------------------------------------------
# 0. 配置
# -------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")

os.makedirs("output_texture", exist_ok=True)

# -------------------------------------------------------
# 1. 加载目标奶牛模型（含纹理）
# -------------------------------------------------------
# load_objs_as_meshes 会同时加载 .mtl 纹理
cow_mesh = load_objs_as_meshes(["cow.obj"], device=device)

verts = cow_mesh.verts_packed()
verts = (verts - verts.mean(0)) / max(verts.abs().max(0)[0])
cow_mesh = cow_mesh.update_padded(verts.unsqueeze(0))
print(f"目标网格：{verts.shape[0]} 个顶点")

# -------------------------------------------------------
# 2. 设置多视角摄像机（20个视角）
# -------------------------------------------------------
num_views = 20
elev = torch.zeros(num_views)
azim = torch.linspace(-180, 180, num_views)
R, T = look_at_view_transform(dist=2.7, elev=elev, azim=azim)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

# 点光源，放在摄像机附近
lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])

# -------------------------------------------------------
# 3. 构建两个渲染器：剪影 + RGB
# -------------------------------------------------------
sigma = 1e-4

# 剪影渲染器（和必做一样）
raster_settings_silhouette = RasterizationSettings(
    image_size=256,
    blur_radius=np.log(1.0 / 1e-4 - 1.0) * sigma,
    faces_per_pixel=50,
)
silhouette_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings_silhouette),
    shader=SoftSilhouetteShader(blend_params=BlendParams(sigma=sigma, gamma=1e-4))
)

# RGB 渲染器（SoftPhongShader，用于纹理优化）
raster_settings_rgb = RasterizationSettings(
    image_size=256,
    blur_radius=np.log(1.0 / 1e-4 - 1.0) * sigma,
    faces_per_pixel=50,
    perspective_correct=False,
)
rgb_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings_rgb),
    shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
)

# -------------------------------------------------------
# 4. 渲染目标图像（Ground Truth）
# -------------------------------------------------------
with torch.no_grad():
    target_images = rgb_renderer(cow_mesh.extend(num_views))      # (N,H,W,4) RGBA
    target_rgb = target_images[..., :3]                           # (N,H,W,3)
    target_silhouette = silhouette_renderer(cow_mesh.extend(num_views))[..., 3]  # (N,H,W)

print(f"目标RGB图像: {target_rgb.shape}")
print(f"目标剪影: {target_silhouette.shape}")

# 展示目标图像的4个视角
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle("Target RGB Images (Ground Truth)", fontsize=13)
for col, v in enumerate([0, 5, 10, 15]):
    axes[col].imshow(target_rgb[v].cpu().numpy().clip(0, 1))
    axes[col].set_title(f"azim={azim[v]:.0f}°")
    axes[col].axis("off")
plt.tight_layout()
plt.savefig("output_texture/target_rgb.png", dpi=100)
plt.show()

# -------------------------------------------------------
# 5. 初始化球体 + 顶点颜色 + 优化器
# -------------------------------------------------------
src_mesh = ico_sphere(4, device)
verts_shape = src_mesh.verts_packed().shape

# 可微参数1：顶点位置偏移
deform_verts = torch.zeros(verts_shape, device=device, requires_grad=True)

# 可微参数2：顶点颜色（RGB），初始化为灰色 0.5
sphere_verts_rgb = torch.full([1, verts_shape[0], 3], 0.5, device=device, requires_grad=True)

# Adam 同时优化两个参数
optimizer = torch.optim.Adam([deform_verts, sphere_verts_rgb], lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)

# -------------------------------------------------------
# 6. 损失权重
# -------------------------------------------------------
w_sil    = 1.0    # 剪影损失
w_rgb    = 1.0    # RGB颜色损失
w_lap    = 0.1    # 拉普拉斯平滑
w_edge   = 1.0    # 边长一致性
w_normal = 0.01   # 法线一致性

# -------------------------------------------------------
# 7. 优化循环
# -------------------------------------------------------
epochs = 500
num_views_per_iter = 4  # 每步随机采样4个视角，节省显存

loss_history = []
print(f"\n开始联合优化（形状 + 纹理），共 {epochs} 轮...\n")

for i in range(epochs):
    optimizer.zero_grad()

    # 形变球体
    new_src_mesh = src_mesh.offset_verts(deform_verts)

    # 给球体附上当前顶点颜色作为纹理
    new_src_mesh.textures = TexturesVertex(verts_features=sphere_verts_rgb)

    # 随机采样几个视角计算 Loss（全部20个太慢）
    view_ids = torch.randperm(num_views)[:num_views_per_iter]
    R_batch = R[view_ids].to(device)
    T_batch = T[view_ids].to(device)
    cameras_batch = FoVPerspectiveCameras(device=device, R=R_batch, T=T_batch)

    # 渲染 RGB 图像
    images_pred = rgb_renderer(
        new_src_mesh.extend(num_views_per_iter),
        cameras=cameras_batch,
        lights=lights
    )
    pred_rgb = images_pred[..., :3]
    pred_silhouette = images_pred[..., 3]

    # --- Loss 计算 ---
    loss_sil    = ((pred_silhouette - target_silhouette[view_ids]) ** 2).mean()
    loss_rgb    = ((pred_rgb - target_rgb[view_ids]) ** 2).mean()
    loss_lap    = mesh_laplacian_smoothing(new_src_mesh, method="uniform")
    loss_edge   = mesh_edge_loss(new_src_mesh)
    loss_normal = mesh_normal_consistency(new_src_mesh)

    loss = (w_sil * loss_sil +
            w_rgb * loss_rgb +
            w_lap * loss_lap +
            w_edge * loss_edge +
            w_normal * loss_normal)

    loss.backward()
    optimizer.step()
    scheduler.step()
    loss_history.append(loss.item())

    if i % 50 == 0 or i == epochs - 1:
        clear_output(wait=True)
        print(f"Step [{i:03d}/{epochs}] "
              f"Loss: {loss.item():.4f} | "
              f"Sil: {loss_sil.item():.4f} | "
              f"RGB: {loss_rgb.item():.4f} | "
              f"Lap: {loss_lap.item():.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.5f}")

        # 用全部20个视角做展示
        with torch.no_grad():
            display_mesh = src_mesh.offset_verts(deform_verts)
            display_mesh.textures = TexturesVertex(verts_features=sphere_verts_rgb)
            display_images = rgb_renderer(display_mesh.extend(num_views))
            display_rgb = display_images[..., :3].clamp(0, 1)

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f"Step {i}/{epochs} — Loss: {loss.item():.4f}", fontsize=13)
        for col, v in enumerate([0, 5, 10, 15]):
            axes[0, col].imshow(target_rgb[v].cpu().numpy().clip(0, 1))
            axes[0, col].set_title(f"GT azim={azim[v]:.0f}°")
            axes[0, col].axis("off")
            axes[1, col].imshow(display_rgb[v].cpu().numpy())
            axes[1, col].set_title(f"Pred azim={azim[v]:.0f}°")
            axes[1, col].axis("off")
        plt.tight_layout()
        plt.savefig(f"output_texture/step_{i:03d}.png", dpi=80)
        plt.show()

# -------------------------------------------------------
# 8. Loss 曲线 + 保存结果
# -------------------------------------------------------
plt.figure(figsize=(10, 4))
plt.plot(loss_history, linewidth=1.5)
plt.title("Loss Curve — 联合纹理优化")
plt.xlabel("Step")
plt.ylabel("Total Loss")
plt.grid(True, alpha=0.3)
plt.savefig("output_texture/loss_curve.png", dpi=100)
plt.show()

# 保存最终 mesh
final_verts = new_src_mesh.verts_list()[0].detach()
final_faces = new_src_mesh.faces_list()[0]
save_obj("output_texture/final_cow_textured.obj", final_verts, final_faces)
print("✅ 联合纹理优化完成！结果保存在 output_texture/")