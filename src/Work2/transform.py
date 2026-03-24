import taichi as ti
import math

# Model: 绕Z轴旋转
def get_model_matrix(angle):
    angle = angle * math.pi / 180
    c = ti.cos(angle)
    s = ti.sin(angle)

    return ti.Matrix([
        [c, -s, 0, 0],
        [s,  c, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1],
    ])


# View: 相机变换（平移）
def get_view_matrix(eye_pos):
    return ti.Matrix([
        [1, 0, 0, -eye_pos[0]],
        [0, 1, 0, -eye_pos[1]],
        [0, 0, 1, -eye_pos[2]],
        [0, 0, 0, 1],
    ])


# Projection: 透视投影
def get_projection_matrix(eye_fov, aspect_ratio, zNear, zFar):
    fov = eye_fov * math.pi / 180

    n = -zNear
    f = -zFar

    t = math.tan(fov / 2) * abs(n)
    b = -t
    r = aspect_ratio * t
    l = -r

    # 透视 → 正交
    persp_to_ortho = ti.Matrix([
        [n, 0, 0, 0],
        [0, n, 0, 0],
        [0, 0, n + f, -n * f],
        [0, 0, 1, 0],
    ])

    # 正交投影：平移
    translate = ti.Matrix([
        [1, 0, 0, -(l + r) / 2],
        [0, 1, 0, -(t + b) / 2],
        [0, 0, 1, -(n + f) / 2],
        [0, 0, 0, 1],
    ])

    # 正交投影：缩放
    scale = ti.Matrix([
        [2 / (r - l), 0, 0, 0],
        [0, 2 / (t - b), 0, 0],
        [0, 0, 2 / (n - f), 0],
        [0, 0, 0, 1],
    ])

    ortho = scale @ translate

    return ortho @ persp_to_ortho