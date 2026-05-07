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
max_bounces[None] = 3


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

    # red sphere
    t = intersect_sphere(ro, rd, vec3(-1.5, 0, 0), 1.0)
    if t > 0 and t < t_min:
        t_min = t
        hit_id = 1

    # mirror sphere
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

        # ✅ 修复：不再翻转图像（关键修复点）
        u = (i + 0.5) / W * 2 - 1
        v = (j + 0.5) / H * 2 - 1
        u *= W / H

        ro = camera_pos
        rd = vec3(u, v, -1).normalized()

        throughput = vec3(1.0)
        final_color = vec3(0.0)

        for bounce in range(5):

            if bounce >= max_bounces[None]:
                break

            t, hit_id = intersect_scene(ro, rd)

            if hit_id == -1:
                break

            hit_point = ro + t * rd

            # normal
            normal = vec3(0)

            if hit_id == 0:
                normal = vec3(0, 1, 0)
            elif hit_id == 1:
                normal = (hit_point - vec3(-1.5, 0, 0)).normalized()
            else:
                normal = (hit_point - vec3(1.5, 0, 0)).normalized()

            # 防自相交
            hit_point = hit_point + normal * 1e-4

            # light
            lp = light_pos[None]
            light_dir = (lp - hit_point).normalized()

            # shadow ray
            t_shadow, shadow_id = intersect_scene(hit_point, light_dir)
            dist_light = (lp - hit_point).norm()

            in_shadow = False
            if shadow_id != -1 and t_shadow < dist_light:
                in_shadow = True

            # ---------------------
            # diffuse
            # ---------------------
            if hit_id == 0 or hit_id == 1:

                color = vec3(0.0)  # ✅ 修复：避免未定义

                # ground checker
                if hit_id == 0:
                    checker = (int(ti.floor(hit_point.x)) +
                               int(ti.floor(hit_point.z))) % 2
                    if checker == 0:
                        color = vec3(1.0)
                    else:
                        color = vec3(0.0)

                # red sphere
                elif hit_id == 1:
                    color = vec3(1.0, 0.2, 0.2)

                diff = max(normal.dot(light_dir), 0.0)

                if in_shadow:
                    diff = 0.1

                final_color += throughput * color * diff
                break

            # ---------------------
            # mirror
            # ---------------------
            elif hit_id == 2:
                rd = rd - 2 * rd.dot(normal) * normal
                ro = hit_point
                throughput *= 0.8

        pixels[i, j] = final_color


# -------------------------
# UI（✔ 已修复关键BUG）
# -------------------------

window = ti.ui.Window("Ray Tracing", (W, H))
canvas = window.get_canvas()
gui = window.get_gui()

while window.running:

    # ✔ 正确 UI 写法（关键修复）
    lp = light_pos[None]

    lp[0] = gui.slider_float("Light X", lp[0], -5, 5)
    lp[1] = gui.slider_float("Light Y", lp[1], 0, 5)
    lp[2] = gui.slider_float("Light Z", lp[2], -5, 5)

    light_pos[None] = lp

    max_bounces[None] = gui.slider_int(
        "Max Bounces", max_bounces[None], 1, 5
    )

    render()

    canvas.set_image(pixels)
    window.show()