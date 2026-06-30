import taichi as ti

ti.init(arch=ti.gpu)

# 物理与网格参数
N = 20
mass = 1.0
dt = 5e-4
k_s = 10000.0
k_d = 1.0
gravity = ti.Vector([0.0, -9.8, 0.0])
max_velocity = 50.0

# 球体参数
sphere_center = ti.Vector([0.0, 0.3, 0.0])
sphere_radius = 0.2

# 弹簧类型数量上限
# 结构: N*(N-1)*2，剪切: (N-1)*(N-1)*2，弯曲: N*(N-2)*2
max_springs = N * N * 8

# 数据场
x          = ti.Vector.field(3, dtype=float, shape=N * N)
v          = ti.Vector.field(3, dtype=float, shape=N * N)
f          = ti.Vector.field(3, dtype=float, shape=N * N)
is_fixed   = ti.field(dtype=int,   shape=N * N)

x_next     = ti.Vector.field(3, dtype=float, shape=N * N)
v_next     = ti.Vector.field(3, dtype=float, shape=N * N)
f_next     = ti.Vector.field(3, dtype=float, shape=N * N)

spring_indices  = ti.field(dtype=int,   shape=max_springs * 2)
spring_pairs    = ti.Vector.field(2, dtype=int,   shape=max_springs)
spring_lengths  = ti.field(dtype=float, shape=max_springs)
num_springs     = ti.field(dtype=int,   shape=())

# ============ 初始化 ============

@ti.kernel
def init_positions():
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        x[idx]       = ti.Vector([i * 0.05 - 0.5, 0.8, j * 0.05 - 0.5])
        v[idx]       = ti.Vector([0.0, 0.0, 0.0])
        f[idx]       = ti.Vector([0.0, 0.0, 0.0])
        is_fixed[idx] = 1 if (j == 0 and (i == 0 or i == N - 1)) else 0

@ti.kernel
def init_springs():
    for i, j in ti.ndrange(N, N):
        idx = i * N + j

        # ── 结构弹簧 (Structural) ──────────────────
        # 右邻
        if i < N - 1:
            nb = (i + 1) * N + j
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c]   = ti.Vector([idx, nb])
            spring_lengths[c] = (x[idx] - x[nb]).norm()
        # 下邻
        if j < N - 1:
            nb = i * N + (j + 1)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c]   = ti.Vector([idx, nb])
            spring_lengths[c] = (x[idx] - x[nb]).norm()

        # ── 剪切弹簧 (Shear) ──────────────────────
        # 右下对角
        if i < N - 1 and j < N - 1:
            nb = (i + 1) * N + (j + 1)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c]   = ti.Vector([idx, nb])
            spring_lengths[c] = (x[idx] - x[nb]).norm()
        # 右上对角（等价于从右上点看左下，避免重复只在 j>0 时加）
        if i < N - 1 and j > 0:
            nb = (i + 1) * N + (j - 1)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c]   = ti.Vector([idx, nb])
            spring_lengths[c] = (x[idx] - x[nb]).norm()

        # ── 弯曲弹簧 (Bending) ────────────────────
        # 右方隔一个
        if i < N - 2:
            nb = (i + 2) * N + j
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c]   = ti.Vector([idx, nb])
            spring_lengths[c] = (x[idx] - x[nb]).norm()
        # 下方隔一个
        if j < N - 2:
            nb = i * N + (j + 2)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c]   = ti.Vector([idx, nb])
            spring_lengths[c] = (x[idx] - x[nb]).norm()

@ti.kernel
def init_spring_indices():
    for i in range(num_springs[None]):
        spring_indices[i * 2]     = spring_pairs[i][0]
        spring_indices[i * 2 + 1] = spring_pairs[i][1]

def init_cloth():
    num_springs[None] = 0
    init_positions()
    init_springs()
    init_spring_indices()

# ============ 力与碰撞（ti.func，内联到 kernel）============

@ti.func
def compute_forces_on(pos: ti.template(), vel: ti.template(), force: ti.template()):
    for i in range(N * N):
        force[i] = gravity * mass - k_d * vel[i]
    for i in range(num_springs[None]):
        a   = spring_pairs[i][0]
        b   = spring_pairs[i][1]
        d   = pos[a] - pos[b]
        dist = d.norm()
        if dist > 1e-6:
            f_spring = -k_s * (dist - spring_lengths[i]) * (d / dist)
            ti.atomic_add(force[a],  f_spring)
            ti.atomic_add(force[b], -f_spring)

@ti.func
def clamp_velocity(vel: ti.template(), idx: int):
    vel_norm = vel[idx].norm()
    if vel_norm > max_velocity:
        vel[idx] = vel[idx] / vel_norm * max_velocity

@ti.func
def resolve_sphere_collision(pos: ti.template(), vel: ti.template(), idx: int):
    """
    若质点在球体内部，将其推回球面，并清除法线方向的速度分量。
    """
    p  = pos[idx]
    sc = sphere_center
    diff = p - sc
    dist = diff.norm()
    if dist < sphere_radius:
        # 推回球面
        normal    = diff / (dist + 1e-6)
        pos[idx]  = sc + normal * sphere_radius
        # 消除指向球内的速度分量（完全非弹性）
        vn = vel[idx].dot(normal)
        if vn < 0.0:
            vel[idx] -= vn * normal

# ============ 积分 kernel ============

@ti.kernel
def step_explicit():
    compute_forces_on(x, v, f)
    for i in range(N * N):
        if is_fixed[i] == 0:
            x[i] += v[i] * dt
            v[i] += (f[i] / mass) * dt
            clamp_velocity(v, i)
            resolve_sphere_collision(x, v, i)

@ti.kernel
def step_semi_implicit():
    compute_forces_on(x, v, f)
    for i in range(N * N):
        if is_fixed[i] == 0:
            v[i] += (f[i] / mass) * dt
            clamp_velocity(v, i)
            x[i] += v[i] * dt
            resolve_sphere_collision(x, v, i)

@ti.kernel
def step_implicit_iter():
    for i in range(N * N):
        v_next[i] = v[i]
        x_next[i] = x[i]
    for _ in ti.static(range(3)):
        compute_forces_on(x_next, v_next, f_next)
        for i in range(N * N):
            if is_fixed[i] == 0:
                v_next[i] = v[i] + (f_next[i] / mass) * dt
                clamp_velocity(v_next, i)
                x_next[i] = x[i] + v_next[i] * dt
    for i in range(N * N):
        v[i] = v_next[i]
        x[i] = x_next[i]
        if is_fixed[i] == 0:
            resolve_sphere_collision(x, v, i)

# ============ 球体渲染辅助 ============
# 用一组离散点近似显示球体表面
SPHERE_DRAW_N = 20
sphere_pts = ti.Vector.field(3, dtype=float, shape=SPHERE_DRAW_N * SPHERE_DRAW_N)

@ti.kernel
def build_sphere_mesh():
    for i, j in ti.ndrange(SPHERE_DRAW_N, SPHERE_DRAW_N):
        theta = (i / SPHERE_DRAW_N) * 3.14159 * 2.0
        phi   = (j / SPHERE_DRAW_N) * 3.14159
        sphere_pts[i * SPHERE_DRAW_N + j] = sphere_center + sphere_radius * ti.Vector([
            ti.sin(phi) * ti.cos(theta),
            ti.cos(phi),
            ti.sin(phi) * ti.sin(theta),
        ])

# ============ 主函数 ============

def main():
    init_cloth()
    build_sphere_mesh()

    window  = ti.ui.Window("Mass Spring - with Shear/Bending + Collision", (800, 800))
    canvas  = window.get_canvas()
    scene   = window.get_scene()
    camera  = ti.ui.Camera()
    camera.position(0.0, 0.5, 2.0)
    camera.lookat(0.0, 0.0, 0.0)

    current_method = 1
    paused = False

    while window.running:
        # ── GUI ──────────────────────────────────────
        window.GUI.begin("Control Panel", 0.02, 0.02, 0.40, 0.38)
        window.GUI.text("Integration Method:")
        labels = ["Explicit Euler", "Semi-Implicit Euler", "Implicit Euler"]
        for mi, label in enumerate(labels):
            prefix = "[*] " if current_method == mi else "[ ] "
            if window.GUI.button(prefix + label):
                current_method = mi
                init_cloth()
        window.GUI.text("")
        pause_label = "Resume" if paused else "Pause"
        if window.GUI.button(pause_label):
            paused = not paused
        if window.GUI.button("Reset Cloth"):
            init_cloth()
        window.GUI.end()

        # ── 物理更新 ─────────────────────────────────
        if not paused:
            for _ in range(40):
                if current_method == 0:
                    step_explicit()
                elif current_method == 1:
                    step_semi_implicit()
                else:
                    step_implicit_iter()

        # ── 渲染 ─────────────────────────────────────
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

        scene.particles(x,           radius=0.015, color=(0.2, 0.6, 1.0))
        scene.lines(x, indices=spring_indices, width=1.5, color=(0.8, 0.8, 0.8))
        scene.particles(sphere_pts,  radius=0.008, color=(1.0, 0.4, 0.2))

        canvas.scene(scene)
        window.show()

if __name__ == '__main__':
    main()