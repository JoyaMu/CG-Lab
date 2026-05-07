import taichi as ti

ti.init(arch=ti.cuda)

vec3 = ti.math.vec3

# -------------------------
# 分辨率
# -------------------------
W, H = 800, 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(W, H))

# -------------------------
# 相机 & 场景参数
# -------------------------
camera_pos = vec3(0, 1, 5)

light_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
max_bounces = ti.field(dtype=ti.i32, shape=())

light_pos[None] = vec3(2, 3, 2)
max_bounces[None] = 5   # 玻璃折射需要更多弹射次数（进一次出一次）


# -------------------------
# 相交函数
# -------------------------

@ti.func
def intersect_sphere(ro, rd, center, radius):
    t = -1.0
    oc = ro - center
    b = oc.dot(rd)
    c = oc.dot(oc) - radius * radius
    h = b * b - c

    if h >= 0:
        s = ti.sqrt(h)
        t1 = -b - s
        t2 = -b + s

        if t1 > 0:
            t = t1
        elif t2 > 0:
            t = t2

    return t


@ti.func
def intersect_plane(ro, rd):
    t = -1.0
    if abs(rd.y) >= 1e-6:
        tmp = (-1.0 - ro.y) / rd.y
        if tmp > 0:
            t = tmp
    return t


@ti.func
def intersect_scene(ro, rd):
    t_min = 1e9
    hit_id = -1

    # plane
    t = intersect_plane(ro, rd)
    if t > 0 and t < t_min:
        t_min = t
        hit_id = 0

    # glass sphere（左球）
    t = intersect_sphere(ro, rd, vec3(-1.5, 0, 0), 1.0)
    if t > 0 and t < t_min:
        t_min = t
        hit_id = 1

    # mirror sphere（右球）
    t = intersect_sphere(ro, rd, vec3(1.5, 0, 0), 1.0)
    if t > 0 and t < t_min:
        t_min = t
        hit_id = 2

    return t_min, hit_id


# -------------------------
# 渲染核心
# -------------------------

@ti.kernel
def render():
    for i, j in pixels:

        u = (i + 0.5) / W * 2 - 1
        v = (j + 0.5) / H * 2 - 1
        u *= W / H

        ro = camera_pos
        rd = vec3(u, v, -1).normalized()

        throughput = vec3(1.0)
        final_color = vec3(0.0)

        for bounce in range(8):   # 内部循环上限略大于 max_bounces

            if bounce >= max_bounces[None]:
                break

            t, hit_id = intersect_scene(ro, rd)

            if hit_id == -1:
                break

            hit_point = ro + t * rd

            # ---- 计算法线 ----
            normal = vec3(0.0)
            if hit_id == 0:
                normal = vec3(0, 1, 0)
            elif hit_id == 1:
                normal = (hit_point - vec3(-1.5, 0, 0)).normalized()
            else:
                normal = (hit_point - vec3(1.5, 0, 0)).normalized()

            # =========================================
            # 玻璃球（hit_id == 1）折射处理
            # =========================================
            if hit_id == 1:

                # 判断光线是从外部还是内部射入球体
                # rd.dot(normal) > 0 说明光线与法线同向，即从内部射出
                front_face = rd.dot(normal) < 0.0

                # 折射率：玻璃约 1.5
                # 外->内：eta = n_air / n_glass = 1/1.5
                # 内->外：eta = n_glass / n_air = 1.5
                eta = 1.0 / 1.5
                n = normal
                if not front_face:
                    n = -normal       # 翻转法线，使其始终朝向入射光线一侧
                    eta = 1.5

                cos_theta = -rd.dot(n)                 # 入射角余弦（>0）
                sin2_theta_t = eta * eta * (1.0 - cos_theta * cos_theta)

                # 全内反射判断：sin²θt > 1 时无法折射
                if sin2_theta_t > 1.0:
                    # 全内反射 → 当作镜面处理
                    rd = rd - 2.0 * rd.dot(n) * n
                    ro = hit_point + n * 1e-4           # 沿法线偏移防自相交
                else:
                    # 斯涅尔折射方向公式
                    # T = eta * rd + (eta * cos_theta - sqrt(1 - sin2_theta_t)) * n
                    cos_theta_t = ti.sqrt(1.0 - sin2_theta_t)
                    refr_dir = eta * rd + (eta * cos_theta - cos_theta_t) * n
                    rd = refr_dir.normalized()
                    # 折射时沿 -n（即进入球体方向）偏移，避免与入射面自相交
                    ro = hit_point - n * 1e-4

                throughput *= 0.96   # 玻璃轻微吸收（可设为 1.0 理想透明）
                continue             # 继续追踪折射/反射光线

            # =========================================
            # 漫反射（地面 & 其他漫反射物体）和镜面球
            # =========================================

            # 防自相交偏移（漫反射和镜面在此统一处理）
            hit_point_offset = hit_point + normal * 1e-4

            # ---- 阴影检测 ----
            lp = light_pos[None]
            light_dir = (lp - hit_point_offset).normalized()
            t_shadow, shadow_id = intersect_scene(hit_point_offset, light_dir)
            dist_light = (lp - hit_point_offset).norm()

            in_shadow = shadow_id != -1 and t_shadow < dist_light

            # ---- 漫反射着色 ----
            if hit_id == 0:
                checker = (int(ti.floor(hit_point.x)) +
                           int(ti.floor(hit_point.z))) % 2
                color = vec3(1.0) if checker == 0 else vec3(0.0)

                diff = max(normal.dot(light_dir), 0.0)
                if in_shadow:
                    diff = 0.1

                final_color += throughput * color * diff
                break

            # ---- 镜面球 ----
            elif hit_id == 2:
                rd = rd - 2.0 * rd.dot(normal) * normal
                ro = hit_point_offset
                throughput *= 0.8
                # continue 到下一次 bounce

        pixels[i, j] = final_color


# -------------------------
# UI
# -------------------------

window = ti.ui.Window("Ray Tracing - Glass & Mirror", (W, H))
canvas = window.get_canvas()
gui = window.get_gui()

while window.running:

    lp = light_pos[None]
    lp[0] = gui.slider_float("Light X", lp[0], -5, 5)
    lp[1] = gui.slider_float("Light Y", lp[1], 0, 5)
    lp[2] = gui.slider_float("Light Z", lp[2], -5, 5)
    light_pos[None] = lp

    max_bounces[None] = gui.slider_int("Max Bounces", max_bounces[None], 1, 8)

    render()
    canvas.set_image(pixels)
    window.show()