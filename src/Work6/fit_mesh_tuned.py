import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

import pytorch3d
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.loss import mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency
from pytorch3d.renderer import (
    look_at_view_transform, FoVPerspectiveCameras,
    RasterizationSettings, MeshRasterizer, SoftSilhouetteShader, BlendParams
)

# 0. 基本配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"当前运行设备: {device}")
print(f"PyTorch3D 版本: {pytorch3d.__version__}")

# 1. 加载奶牛模型
verts, faces, _ = load_obj("cow.obj")
faces_idx = faces.verts_idx.to(device)
verts = verts.to(device)
verts = (verts - verts.mean(0)) / max(verts.abs().max(0)[0])
cow_mesh = Meshes(verts=[verts], faces=[faces_idx])
print(f"目标网格：{verts.shape[0]} 个顶点，{faces_idx.shape[0]} 个面")

# 2. 渲染管线
num_views = 20
elev = torch.zeros(num_views)
azim = torch.linspace(-180, 180, num_views)
R, T = look_at_view_transform(dist=2.7, elev=elev, azim=azim)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

sigma = 1e-4
raster_settings = RasterizationSettings(
    image_size=256,
    blur_radius=np.log(1.0 / 1e-4 - 1.0) * sigma,
    faces_per_pixel=50,
)
rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
shader = SoftSilhouetteShader(blend_params=BlendParams(sigma=sigma, gamma=1e-4))

with torch.no_grad():
    target_silhouette = shader(
        rasterizer(cow_mesh.extend(num_views)),
        cow_mesh.extend(num_views)
    )[..., 3]
print(f"目标剪影渲染完成：{target_silhouette.shape}")

# 3. 初始化球体与优化器
src_mesh = ico_sphere(4, device)
deform_verts = torch.zeros_like(src_mesh.verts_packed(), requires_grad=True)
optimizer = torch.optim.Adam([deform_verts], lr=0.005)

# 4. 优化循环
epochs = 500
w_lap    = 0.05   # 原来 0.1，降低→允许牛角耳朵更尖锐
w_edge   = 0.5    # 原来 1.0，降低→允许更多形变
w_normal = 0.005

for i in range(epochs):
    optimizer.zero_grad()
    new_src_mesh = src_mesh.offset_verts(deform_verts)
    pred_silhouette = shader(
        rasterizer(new_src_mesh.extend(num_views)),
        new_src_mesh.extend(num_views)
    )[..., 3]

    loss_sil    = ((pred_silhouette - target_silhouette) ** 2).mean()
    loss_lap    = mesh_laplacian_smoothing(new_src_mesh, method="uniform")
    loss_edge   = mesh_edge_loss(new_src_mesh)
    loss_normal = mesh_normal_consistency(new_src_mesh)
    loss = loss_sil + w_lap*loss_lap + w_edge*loss_edge + w_normal*loss_normal

    loss.backward()
    optimizer.step()

    if i % 20 == 0 or i == epochs - 1:
        clear_output(wait=True)
        print(f"Step [{i:03d}/{epochs}] Loss: {loss.item():.4f} | Sil: {loss_sil.item():.4f} | Lap: {loss_lap.item():.4f}")

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle(f"Step {i}/{epochs} — Loss: {loss.item():.4f}", fontsize=13)
        for col, v in enumerate([0, 5, 10, 15]):
            axes[0, col].imshow(target_silhouette[v].cpu().numpy(), cmap='gray')
            axes[0, col].set_title(f"GT azim={azim[v]:.0f}°")
            axes[0, col].axis("off")
            axes[1, col].imshow(pred_silhouette[v].detach().cpu().numpy(), cmap='gray')
            axes[1, col].set_title(f"Pred azim={azim[v]:.0f}°")
            axes[1, col].axis("off")
        plt.tight_layout()
        plt.show()

# 5. 保存最终模型
os.makedirs("output_meshes", exist_ok=True)
final_verts = new_src_mesh.verts_list()[0].detach()
final_faces = new_src_mesh.faces_list()[0]
save_obj("output_meshes/final_cow.obj", final_verts, final_faces)
print("优化完成！最终模型已保存至 output_meshes/final_cow.obj")