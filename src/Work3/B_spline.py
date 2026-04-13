import taichi as ti
import numpy as np

# ─────────────────────────────────────────────
# 初始化与常量
# ─────────────────────────────────────────────
ti.init(arch=ti.gpu)

WIDTH, HEIGHT      = 800, 800
NUM_SEGMENTS       = 1000          # 每段采样数
MAX_CURVE_POINTS   = 8000          # 分段多，预留更大缓冲
MAX_CONTROL_POINTS = 100

# 均匀三次 B 样条基矩阵（已除以 6）
B_BASIS = np.array([
    [-1,  3, -3,  1],
    [ 3, -6,  3,  0],
    [-3,  0,  3,  0],
    [ 1,  4,  1,  0],
], dtype=np.float64) / 6.0

pixels             = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=(MAX_CURVE_POINTS,))
gui_points         = ti.Vector.field(2, dtype=ti.f32, shape=(MAX_CONTROL_POINTS,))

# ─────────────────────────────────────────────
# 均匀三次 B 样条（CPU）
# ─────────────────────────────────────────────
def b_spline_point(p0, p1, p2, p3, t: float):
    """
    用基矩阵计算均匀三次 B 样条一段上参数 t 处的点。
    P(t) = [t³ t² t 1] · M_basis · [P0 P1 P2 P3]ᵀ
    """
    T  = np.array([t**3, t**2, t, 1.0])
    Px = np.array([p0[0], p1[0], p2[0], p3[0]], dtype=np.float64)
    Py = np.array([p0[1], p1[1], p2[1], p3[1]], dtype=np.float64)
    return [float(T @ (B_BASIS @ Px)), float(T @ (B_BASIS @ Py))]

def compute_b_spline(control_points: list) -> list:
    """
    采样所有 B 样条分段并拼接。
    n 个控制点 → n-3 段，每段均匀采样 seg_samples 个点。
    """
    n   = len(control_points)
    num_segs    = n - 3                              # 段数
    seg_samples = max(NUM_SEGMENTS // num_segs, 20)  # 每段采样数，至少 20
    pts = []
    for i in range(num_segs):
        p0, p1 = control_points[i],     control_points[i + 1]
        p2, p3 = control_points[i + 2], control_points[i + 3]
        # 最后一段包含 t=1 的端点，其余段不重复采样端点
        count = seg_samples + (1 if i == num_segs - 1 else 0)
        for j in range(count):
            t = j / seg_samples
            pts.append(b_spline_point(p0, p1, p2, p3, t))
    return pts

# ─────────────────────────────────────────────
# GPU 内核
# ─────────────────────────────────────────────
@ti.kernel
def clear_pixels():
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.0, 0.0, 0.0])

@ti.kernel
def draw_curve_kernel(n: ti.i32):
    for i in range(n):
        fx = curve_points_field[i][0]
        fy = curve_points_field[i][1]
        px = int(fx * WIDTH)
        py = int(fy * HEIGHT)
        if 0 <= px < WIDTH and 0 <= py < HEIGHT:
            pixels[px, py] = ti.Vector([0.0, 1.0, 0.0])

# ─────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────
def upload_curve_points(pts: list) -> int:
    actual = min(len(pts), MAX_CURVE_POINTS)
    arr = np.zeros((MAX_CURVE_POINTS, 2), dtype=np.float32)
    for i in range(actual):
        arr[i, 0] = pts[i][0]
        arr[i, 1] = pts[i][1]
    curve_points_field.from_numpy(arr)
    return actual

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
        "B-Spline Curve — 左键添加控制点（至少 4 个）| C 键清空",
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

        # B 样条至少需要 4 个控制点
        if len(control_points) >= 4:
            pts      = compute_b_spline(control_points)
            actual_n = upload_curve_points(pts)
            draw_curve_kernel(actual_n)

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