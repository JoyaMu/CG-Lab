import taichi as ti
import taichi.math as tm

ti.init(arch=ti.cpu)

WIDTH, HEIGHT = 800, 600

pixels = ti.Vector.field(3, dtype=float, shape=(WIDTH, HEIGHT))

Ka        = ti.field(dtype=float, shape=())
Kd        = ti.field(dtype=float, shape=())
Ks        = ti.field(dtype=float, shape=())
Shininess = ti.field(dtype=float, shape=())

Ka[None]        = 0.2
Kd[None]        = 0.7
Ks[None]        = 0.5
Shininess[None] = 32.0

# ────────────────────────────────────────────────
# 场景常量（与 phong_lighting.py 完全相同）
# ────────────────────────────────────────────────
CAM_POS   = tm.vec3(0.0, 0.0, 5.0)
LIGHT_POS = tm.vec3(2.0, 3.0, 4.0)
LIGHT_COL = tm.vec3(1.0, 1.0, 1.0)
BG_COLOR  = tm.vec3(0.05, 0.2, 0.25)

SPHERE_CENTER = tm.vec3(-1.2, -0.2, 0.0)
SPHERE_RADIUS = 1.2
SPHERE_COLOR  = tm.vec3(0.8, 0.1, 0.1)

CONE_APEX   = tm.vec3(1.2,  1.2, 0.0)
CONE_BASE_Y = -1.4
CONE_RADIUS = 1.2
CONE_HEIGHT = CONE_APEX[1] - CONE_BASE_Y  # 2.6
CONE_COLOR  = tm.vec3(0.6, 0.2, 0.8)

# ────────────────────────────────────────────────
# 求交函数
# ────────────────────────────────────────────────

@ti.func
def intersect_sphere(ray_o: tm.vec3, ray_d: tm.vec3) -> float:
    oc = ray_o - SPHERE_CENTER
    a  = tm.dot(ray_d, ray_d)
    b  = 2.0 * tm.dot(oc, ray_d)
    c  = tm.dot(oc, oc) - SPHERE_RADIUS * SPHERE_RADIUS
    disc = b * b - 4.0 * a * c
    t = -1.0
    if disc >= 0.0:
        sqrt_d = tm.sqrt(disc)
        t1 = (-b - sqrt_d) / (2.0 * a)
        t2 = (-b + sqrt_d) / (2.0 * a)
        if t1 > 1e-4:
            t = t1
        elif t2 > 1e-4:
            t = t2
    return t


@ti.func
def intersect_cone(ray_o: tm.vec3, ray_d: tm.vec3) -> float:
    k  = CONE_RADIUS / CONE_HEIGHT
    k2 = k * k

    ax = CONE_APEX[0]; ay = CONE_APEX[1]; az = CONE_APEX[2]
    ox = ray_o[0] - ax; oy = ray_o[1] - ay; oz = ray_o[2] - az
    dx = ray_d[0];      dy = ray_d[1];      dz = ray_d[2]

    A = dx*dx + dz*dz - k2*dy*dy
    B = 2.0*(ox*dx + oz*dz - k2*oy*dy)
    C = ox*ox + oz*oz - k2*oy*oy

    best_t = -1.0

    if ti.abs(A) < 1e-8:
        if ti.abs(B) > 1e-8:
            tc = -C / B
            if tc > 1e-4:
                hit_y = ray_o[1] + tc * ray_d[1]
                if CONE_BASE_Y <= hit_y <= ay:
                    best_t = tc
    else:
        disc = B*B - 4.0*A*C
        if disc >= 0.0:
            sqrt_d = tm.sqrt(disc)
            for sign in ti.static([-1.0, 1.0]):
                tc = (-B + sign * sqrt_d) / (2.0 * A)
                if tc > 1e-4:
                    hit_y = ray_o[1] + tc * ray_d[1]
                    if CONE_BASE_Y <= hit_y <= ay:
                        if best_t < 0.0 or tc < best_t:
                            best_t = tc

    if ti.abs(ray_d[1]) > 1e-6:
        t_cap = (CONE_BASE_Y - ray_o[1]) / ray_d[1]
        if t_cap > 1e-4:
            hx = ray_o[0] + t_cap * ray_d[0] - ax
            hz = ray_o[2] + t_cap * ray_d[2] - az
            if hx*hx + hz*hz <= CONE_RADIUS * CONE_RADIUS:
                if best_t < 0.0 or t_cap < best_t:
                    best_t = t_cap

    return best_t


# ────────────────────────────────────────────────
# 法向量
# ────────────────────────────────────────────────

@ti.func
def sphere_normal(hit: tm.vec3) -> tm.vec3:
    return (hit - SPHERE_CENTER).normalized()


@ti.func
def cone_normal(hit: tm.vec3) -> tm.vec3:
    n = tm.vec3(0.0, -1.0, 0.0)
    if ti.abs(hit[1] - CONE_BASE_Y) >= 1e-4:
        dx = hit[0] - CONE_APEX[0]
        dz = hit[2] - CONE_APEX[2]
        r  = tm.sqrt(dx*dx + dz*dz)
        k  = CONE_RADIUS / CONE_HEIGHT
        n  = tm.vec3(dx / (r + 1e-8), k, dz / (r + 1e-8)).normalized()
    return n


# ────────────────────────────────────────────────
# Blinn-Phong 着色器
# 唯一改动：用半程向量 H = normalize(L+V) 替代反射向量 R
# 高光项变为 max(0, N·H)^n，其余与 Phong 完全相同
# ────────────────────────────────────────────────

@ti.func
def blinn_phong_shade(hit: tm.vec3, N: tm.vec3, obj_color: tm.vec3) -> tm.vec3:
    ka   = Ka[None]
    kd   = Kd[None]
    ks   = Ks[None]
    shin = Shininess[None]

    L = (LIGHT_POS - hit).normalized()   # 指向光源
    V = (CAM_POS   - hit).normalized()   # 指向摄像机
    H = (L + V).normalized()             # 半程向量（Blinn-Phong 核心改动）

    ambient  = ka * LIGHT_COL * obj_color
    diffuse  = kd * ti.max(0.0, tm.dot(N, L)) * LIGHT_COL * obj_color
    specular = ks * tm.pow(ti.max(0.0, tm.dot(N, H)), shin) * LIGHT_COL

    return tm.clamp(ambient + diffuse + specular, 0.0, 1.0)


# ────────────────────────────────────────────────
# 渲染 Kernel
# ────────────────────────────────────────────────

@ti.kernel
def render():
    for i, j in pixels:
        aspect = float(WIDTH) / float(HEIGHT)
        u = (float(i) / float(WIDTH)  - 0.5) * 2.0 * aspect
        v = (float(j) / float(HEIGHT) - 0.5) * 2.0

        ray_o = CAM_POS
        ray_d = tm.vec3(u, v, -1.0).normalized()

        t_sphere = intersect_sphere(ray_o, ray_d)
        t_cone   = intersect_cone  (ray_o, ray_d)

        color   = BG_COLOR
        best_t  = -1.0
        hit_obj = 0

        if t_sphere > 0.0:
            best_t  = t_sphere
            hit_obj = 1
        if t_cone > 0.0:
            if best_t < 0.0 or t_cone < best_t:
                best_t  = t_cone
                hit_obj = 2

        if hit_obj == 1:
            hit   = ray_o + best_t * ray_d
            N     = sphere_normal(hit)
            color = blinn_phong_shade(hit, N, SPHERE_COLOR)
        elif hit_obj == 2:
            hit   = ray_o + best_t * ray_d
            N     = cone_normal(hit)
            color = blinn_phong_shade(hit, N, CONE_COLOR)

        pixels[i, j] = color


# ────────────────────────────────────────────────
# 主循环
# ────────────────────────────────────────────────

def main():
    window = ti.ui.Window("Blinn-Phong Lighting Model", (WIDTH, HEIGHT), vsync=True)
    canvas = window.get_canvas()
    gui    = window.get_gui()

    while window.running:
        with gui.sub_window("Blinn-Phong Parameters", 0.02, 0.02, 0.30, 0.22):
            Ka[None]        = gui.slider_float("Ka  (Ambient)",  Ka[None],        0.0,   1.0)
            Kd[None]        = gui.slider_float("Kd  (Diffuse)",  Kd[None],        0.0,   1.0)
            Ks[None]        = gui.slider_float("Ks  (Specular)", Ks[None],        0.0,   1.0)
            Shininess[None] = gui.slider_float("Shininess",      Shininess[None], 1.0, 128.0)

        render()
        canvas.set_image(pixels)
        window.show()


if __name__ == "__main__":
    main()