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
num_samples = ti.field(dtype=ti.i32, shape=())   # MSAA 采样数

light_pos[None] = vec3(2, 3, 2)
max_bounces[None] = 5
num_samples[None] = 4   # 默认 4 次采样


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

    t = intersect_plane(ro, rd)
    if t > 0 and t < t_min:
        t_min = t
        hit_id = 0

    t = intersect_sphere(ro, rd, vec3(-1.5, 0, 0), 1.0)
    if t > 0 and t < t_min:
        t_min = t
        hit_id = 1

    t = intersect_sphere(ro, rd, vec3(1.5, 0, 0), 1.0)
    if t > 0 and t < t_min:
        t_min = t
        hit_id = 2

    return t_min, hit_id


# -------------------------
# 单条光线追踪（抽出为函数，方便多次调用）
# -------------------------

@ti.func
def trace_ray(ro, rd, max_b):
    throughput = vec3(1.0)
    final_color = vec3(0.0)

    for bounce in range(8):

        if bounce >= max_b:
            break

        t, hit_id = intersect_scene(ro, rd)

        if hit_id == -1:
            break

        hit_point = ro + t * rd

        normal = vec3(0.0)
        if hit_id == 0:
            normal = vec3(0, 1, 0)
        elif hit_id == 1:
            normal = (hit_point - vec3(-1.5, 0, 0)).normalized()
        else:
            normal = (hit_point - vec3(1.5, 0, 0)).normalized()

        # ---- 玻璃折射 ----
        if hit_id == 1:
            front_face = rd.dot(normal) < 0.0
            eta = 1.0 / 1.5
            n = normal
            if not front_face:
                n = -normal
                eta = 1.5

            cos_theta = -rd.dot(n)
            sin2_theta_t = eta * eta * (1.0 - cos_theta * cos_theta)

            if sin2_theta_t > 1.0:
                rd = rd - 2.0 * rd.dot(n) * n
                ro = hit_point + n * 1e-4
            else:
                cos_theta_t = ti.sqrt(1.0 - sin2_theta_t)
                refr_dir = eta * rd + (eta * cos_theta - cos_theta_t) * n
                rd = refr_dir.normalized()
                ro = hit_point - n * 1e-4

            throughput *= 0.96
            continue

        hit_point_offset = hit_point + normal * 1e-4

        # ---- 阴影检测 ----
        lp = light_pos[None]
        light_dir = (lp - hit_point_offset).normalized()
        t_shadow, shadow_id = intersect_scene(hit_point_offset, light_dir)
        dist_light = (lp - hit_point_offset).norm()
        in_shadow = shadow_id != -1 and t_shadow < dist_light

        # ---- 漫反射地面 ----
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

    return final_color


# -------------------------
# 渲染核心（MSAA）
# -------------------------

@ti.kernel
def render():
    for i, j in pixels:

        color_sum = vec3(0.0)
        n_s = num_samples[None]

        # 每个像素内发射 n_s 条随机偏移的射线
        for s in range(16):   # Taichi 要求循环上限为编译期常量，用 16 作上限
            if s >= n_s:
                break

            # 在像素内随机抖动：offset ∈ (-0.5, 0.5)
            # ti.random() 返回 [0, 1) 均匀随机数
            ox = ti.random(ti.f32) - 0.5
            oy = ti.random(ti.f32) - 0.5

            u = (i + 0.5 + ox) / W * 2 - 1
            v = (j + 0.5 + oy) / H * 2 - 1
            u *= W / H

            ro = camera_pos
            rd = vec3(u, v, -1).normalized()

            color_sum += trace_ray(ro, rd, max_bounces[None])

        # 取平均
        pixels[i, j] = color_sum / float(n_s)


# -------------------------
# UI
# -------------------------

window = ti.ui.Window("Ray Tracing - MSAA", (W, H))
canvas = window.get_canvas()
gui = window.get_gui()

while window.running:

    lp = light_pos[None]
    lp[0] = gui.slider_float("Light X", lp[0], -5, 5)
    lp[1] = gui.slider_float("Light Y", lp[1], 0, 5)
    lp[2] = gui.slider_float("Light Z", lp[2], -5, 5)
    light_pos[None] = lp

    max_bounces[None] = gui.slider_int("Max Bounces", max_bounces[None], 1, 8)
    num_samples[None] = gui.slider_int("MSAA Samples", num_samples[None], 1, 16)

    render()
    canvas.set_image(pixels)
    window.show()