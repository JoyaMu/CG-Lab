import taichi as ti
from transform import get_model_matrix, get_view_matrix, get_projection_matrix

ti.init(arch=ti.cpu)

gui = ti.GUI("MVP Transform", res=(700, 700))

# 三角形顶点（齐次坐标）
v0 = [2.0, 0.0, -2.0, 1.0]
v1 = [0.0, 2.0, -2.0, 1.0]
v2 = [-2.0, 0.0, -2.0, 1.0]

eye_pos = [0, 0, 5]

angle = 0


# MVP 变换 + 透视除法 + 映射到屏幕
def mvp_transform(v, mvp):
    vec = ti.Vector(v)
    res = mvp @ vec

    # 透视除法
    res /= res[3]

    # 映射到屏幕 ([-1,1] → [0,1])
    x = (res[0] + 1) / 2
    y = (res[1] + 1) / 2

    return (x, y)


while gui.running:
    gui.clear(0x0)

    # 键盘控制旋转
    for e in gui.get_events():
        if e.key == 'a':
            angle += 1
        elif e.key == 'd':
            angle -= 1
        elif e.key == ti.GUI.ESCAPE:
            gui.running = False

    # 三个矩阵
    model = get_model_matrix(angle)
    view = get_view_matrix(eye_pos)
    projection = get_projection_matrix(45, 1, 0.1, 50)

    mvp = projection @ view @ model

    # 变换顶点
    p0 = mvp_transform(v0, mvp)
    p1 = mvp_transform(v1, mvp)
    p2 = mvp_transform(v2, mvp)

    # 画线框三角形
    gui.line(p0, p1, radius=2, color=0xff0000)
    gui.line(p1, p2, radius=2, color=0x00ff00)
    gui.line(p2, p0, radius=2, color=0x0000ff)

    gui.show()