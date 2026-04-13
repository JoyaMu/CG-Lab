import taichi as ti
import numpy as np

# ─────────────────────────────────────────────
# 初始化与常量
# ─────────────────────────────────────────────
ti.init(arch=ti.gpu)

WIDTH, HEIGHT      = 800, 800
NUM_SEGMENTS       = 1000
MAX_CONTROL_POINTS = 100

pixels             = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=(NUM_SEGMENTS + 1,))
gui_points         = ti.Vector.field(2, dtype=ti.f32, shape=(MAX_CONTROL_POINTS,))

# ─────────────────────────────────────────────
# De Casteljau 算法（CPU）
# ─────────────────────────────────────────────
def de_casteljau(points: list, t: float) -> list:
    pts = [list(p) for p in points]
    n = len(pts)
    while n > 1:
        new_pts = []
        for i in range(n - 1):
            x = (1 - t) * pts[i][0] + t * pts[i + 1][0]
            y = (1 - t) * pts[i][1] + t * pts[i + 1][1]
            new_pts.append([x, y])
        pts = new_pts
        n -= 1
    return pts[0]

# ─────────────────────────────────────────────
# GPU 内核
# ─────────────────────────────────────────────
@ti.kernel
def clear_pixels():
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.0, 0.0, 0.0])

@ti.kernel
def draw_curve_aa_kernel(n: ti.i32):
    """
    反走样光栅化：
    对每个采样点周围 3×3 邻域，计算各像素中心与精确浮点坐标的欧氏距离，
    用高斯衰减（sigma=0.8）分配亮度权重，通过 ti.atomic_max 叠加到绿色通道。
    """
    for i in range(n):
        fx = curve_points_field[i][0] * WIDTH
        fy = curve_points_field[i][1] * HEIGHT
        cx = int(fx)
        cy = int(fy)
        for dx in ti.static(range(-1, 2)):
            for dy in ti.static(range(-1, 2)):
                nx = cx + dx
                ny = cy + dy
                if 0 <= nx < WIDTH and 0 <= ny < HEIGHT:
                    dist2 = (float(nx) + 0.5 - fx) ** 2 + (float(ny) + 0.5 - fy) ** 2
                    w = ti.exp(-dist2 / (2.0 * 0.8 * 0.8))
                    ti.atomic_max(pixels[nx, ny][1], w)

# ─────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────
def compute_and_upload_curve(control_points: list):
    arr = np.zeros((NUM_SEGMENTS + 1, 2), dtype=np.float32)
    for i in range(NUM_SEGMENTS + 1):
        t = i / NUM_SEGMENTS
        pt = de_casteljau(control_points, t)
        arr[i, 0] = pt[0]
        arr[i, 1] = pt[1]
    curve_points_field.from_numpy(arr)

def upload_gui_points(control_points: list):
    arr = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)
    for i, p in enumerate(control_points):
        arr[i, 0] = p[0]
        arr[i, 1] = p[1]
    gui_points.from_numpy(arr)

# ─────────────────────────────────────────────
# 主循环
# ─────────────────────────────────────────────
def main():
    window = ti.ui.Window(
        "Bézier Curve (反走样) — 左键添加控制点 | C 键清空",
        (WIDTH, HEIGHT)
    )
    canvas = window.get_canvas()

    control_points: list = []
    prev_mouse_pressed   = False

    clear_pixels()

    while window.running:
        if window.get_event(ti.ui.PRESS):
            if window.event.key == 'c':
                control_points.clear()
                clear_pixels()

        cur_pressed = window.is_pressed(ti.ui.LMB)
        if cur_pressed and not prev_mouse_pressed:
            if len(control_points) < MAX_CONTROL_POINTS:
                pos = window.get_cursor_pos()
                control_points.append([pos[0], pos[1]])
        prev_mouse_pressed = cur_pressed

        clear_pixels()

        if len(control_points) >= 2:
            compute_and_upload_curve(control_points)
            draw_curve_aa_kernel(NUM_SEGMENTS + 1)

        canvas.set_image(pixels)

        if len(control_points) >= 2:
            n_pts      = len(control_points)
            line_field = ti.Vector.field(2, dtype=ti.f32, shape=(n_pts,))
            line_field.from_numpy(np.array(control_points, dtype=np.float32))
            idx = []
            for k in range(n_pts - 1):
                idx.extend([k, k + 1])
            indices_field = ti.field(dtype=ti.i32, shape=(len(idx),))
            indices_field.from_numpy(np.array(idx, dtype=np.int32))
            canvas.lines(line_field, width=0.002,
                         indices=indices_field,
                         color=(0.5, 0.5, 0.5))

        if len(control_points) >= 1:
            upload_gui_points(control_points)
            canvas.circles(gui_points, radius=0.008, color=(1.0, 0.2, 0.2))

        window.show()

if __name__ == "__main__":
    main()