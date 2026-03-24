import taichi as ti
import time
from . import config
from . import physics


def main():
    ti.init(arch=ti.gpu)

    # 初始化粒子
    physics.setup()

    window = ti.ui.Window("CG Lab Work0 - Particles", (config.WINDOW_W, config.WINDOW_H))
    canvas = window.get_canvas()
    gui = window.get_gui()

    paused = False
    show_help = True

    zoom = 1.0

    last_t = time.perf_counter()
    fps = 0.0

    kick_strength = 80.0      # 鼠标扰动强度
    center_mass = getattr(config, "CENTER_MASS", 30.0) 
    center_soft = getattr(config, "CENTER_SOFT", 1.5e-2)

    while window.running:
        # ----------------------------
        # 1) 键盘事件（按键触发）
        # ----------------------------
        if window.is_pressed('r'):
            physics.init_particles()

        if window.is_pressed(ti.ui.SPACE):
            paused = not paused

        if window.is_pressed('h'):
            show_help = not show_help

        if window.is_pressed('z'):
            zoom *= 1.01

        if window.is_pressed('x'):
            zoom *= 0.99

        zoom = max(0.5, min(5.0, zoom))

        # ----------------------------
        # 2) 鼠标事件
        # ----------------------------
        mx, my = window.get_cursor_pos()

        if window.is_pressed(ti.ui.LMB):
            physics.kick(mx, my, kick_strength * config.DT)

        if window.is_pressed('q'):
            center_mass *= 0.998
        if window.is_pressed('e'):
            center_mass *= 1.002
        center_mass = max(1.0, min(500.0, center_mass))

        # ----------------------------
        # 3) 仿真推进
        # ----------------------------
        if not paused:
            physics.step(center_mass, center_soft)

        # ----------------------------
        # 4) 渲染
        # ----------------------------
        canvas.set_background_color((0.05, 0.05, 0.08))
        r = 2.5 / min(config.WINDOW_W, config.WINDOW_H)
        physics.update_draw_pos(zoom)
        canvas.circles(physics.draw_pos, radius=r, color=(0.2, 0.7, 1.0))

        # ----------------------------
        # 5) GUI 文本
        # ----------------------------
        if show_help:
            gui.text(f"FPS: {fps:.1f}")
            gui.text("Controls:")
            gui.text("  R     : reset")
            gui.text("  SPACE : pause/resume")
            gui.text("  H     : toggle help")
            gui.text("  LMB   : stir particles")
            gui.text("  Q / E : adjust center mass")
            gui.text("  Z / X : zoom in / out")
            gui.text(f"zoom: {zoom:.2f}")
            gui.text(f"center_mass: {center_mass:.2f}")
            gui.text(f"Paused: {paused}")

        now = time.perf_counter()
        dt_real = now - last_t
        if dt_real > 0:
            fps = 1.0 / dt_real
        last_t = now

        window.show()


if __name__ == "__main__":
    main()