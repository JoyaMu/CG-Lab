import taichi as ti
import numpy as np

# ─────────────────────────────────────────────
# 任务 1：初始化与常量定义
# ─────────────────────────────────────────────
ti.init(arch=ti.gpu)

WIDTH, HEIGHT = 800, 800
NUM_SEGMENTS = 1000          # 曲线采样段数，共 1001 个点
MAX_CONTROL_POINTS = 100

# GPU 缓冲区
pixels            = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=(NUM_SEGMENTS + 1))
gui_points         = ti.Vector.field(2, dtype=ti.f32, shape=(MAX_CONTROL_POINTS,))

# ─────────────────────────────────────────────
# 任务 2：De Casteljau 算法（纯 Python / CPU）
# ─────────────────────────────────────────────
def de_casteljau(points: list, t: float) -> list:
    """
    递归线性插值求贝塞尔曲线上参数 t 处的点。
    points : [[x0,y0], [x1,y1], ...] 归一化坐标 [0,1]
    t      : 参数，范围 [0, 1]
    返回   : [x, y]
    """
    pts = [list(p) for p in points]          # 深拷贝，避免修改原始数据
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
# 任务 3：GPU 绘制内核
# ─────────────────────────────────────────────
@ti.kernel
def clear_pixels():
    """将画布清空为黑色。"""
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.0, 0.0, 0.0])

@ti.kernel
def draw_curve_kernel(n: ti.i32):
    """
    从 curve_points_field 读取 n 个归一化坐标，
    映射到像素索引后点亮绿色。GPU 并行执行。
    """
    for i in range(n):
        fx = curve_points_field[i][0]
        fy = curve_points_field[i][1]
        px = int(fx * WIDTH)
        py = int(fy * HEIGHT)
        if 0 <= px < WIDTH and 0 <= py < HEIGHT:
            pixels[px, py] = ti.Vector([0.0, 1.0, 0.0])   # 绿色

# ─────────────────────────────────────────────
# 辅助：把 CPU 上算好的曲线点批量传给 GPU
# ─────────────────────────────────────────────
def compute_and_upload_curve(control_points: list):
    """
    在 CPU 端采样 NUM_SEGMENTS+1 个点，
    一次性 from_numpy 上传到 GPU。
    """
    n = NUM_SEGMENTS + 1
    arr = np.zeros((n, 2), dtype=np.float32)
    for i in range(n):
        t = i / NUM_SEGMENTS
        pt = de_casteljau(control_points, t)
        arr[i, 0] = pt[0]
        arr[i, 1] = pt[1]
    curve_points_field.from_numpy(arr)

# ─────────────────────────────────────────────
# 辅助：对象池技巧——把控制点塞进定长 Field
# ─────────────────────────────────────────────
def upload_gui_points(control_points: list):
    """
    将真实控制点写入前 k 个位置，
    其余位置填 -10（藏到屏幕外）。
    """
    arr = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)
    for i, p in enumerate(control_points):
        arr[i, 0] = p[0]
        arr[i, 1] = p[1]
    gui_points.from_numpy(arr)

# ─────────────────────────────────────────────
# 任务 4 & 5：主循环
# ─────────────────────────────────────────────
def main():
    window = ti.ui.Window("Bézier Curve — 左键添加控制点 | C 键清空", (WIDTH, HEIGHT))
    canvas = window.get_canvas()

    control_points: list = []   # 存储归一化坐标 [[x,y], ...]
    prev_mouse_pressed = False  # 用于边沿检测，避免长按重复添加

    clear_pixels()

    while window.running:
        # ── 事件处理 ──────────────────────────────
        # 1. 键盘 C 键：清空控制点
        if window.get_event(ti.ui.PRESS):
            if window.event.key == 'c':
                control_points.clear()
                clear_pixels()

        # 2. 鼠标左键点击（边沿检测：按下瞬间触发一次）
        cur_pressed = window.is_pressed(ti.ui.LMB)
        if cur_pressed and not prev_mouse_pressed:
            if len(control_points) < MAX_CONTROL_POINTS:
                pos = window.get_cursor_pos()   # 归一化坐标 (x, y) ∈ [0,1]
                control_points.append([pos[0], pos[1]])
        prev_mouse_pressed = cur_pressed

        # ── 绘制逻辑 ──────────────────────────────
        clear_pixels()

        if len(control_points) >= 2:
            # CPU 计算 → 批量上传 → GPU 并行点亮
            compute_and_upload_curve(control_points)
            draw_curve_kernel(NUM_SEGMENTS + 1)

        # 传给 canvas 显示
        canvas.set_image(pixels)

        # ── 绘制控制多边形（灰色折线，连续不断开）──────────────────
        if len(control_points) >= 2:
            n_pts = len(control_points)
            lines_np = np.array(control_points, dtype=np.float32)
            line_field = ti.Vector.field(2, dtype=ti.f32, shape=(n_pts,))
            line_field.from_numpy(lines_np)

            # indices 格式：每条线段用两个端点索引表示
            # 折线 0-1, 1-2, 2-3, ... → [0,1, 1,2, 2,3, ...]
            idx = []
            for k in range(n_pts - 1):
                idx.extend([k, k + 1])
            indices_field = ti.field(dtype=ti.i32, shape=(len(idx),))
            indices_field.from_numpy(np.array(idx, dtype=np.int32))

            canvas.lines(line_field, width=0.002,
                         indices=indices_field,
                         color=(0.5, 0.5, 0.5))

        # ── 绘制控制点红圆 ────────────────────────
        if len(control_points) >= 1:
            upload_gui_points(control_points)
            canvas.circles(gui_points, radius=0.008, color=(1.0, 0.2, 0.2))

        window.show()

if __name__ == "__main__":
    main()