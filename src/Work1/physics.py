import taichi as ti
from . import config

pos = None
vel = None
draw_pos = None


def setup():
    global pos, vel, draw_pos
    pos = ti.Vector.field(2, dtype=ti.f32, shape=config.N_PARTICLES)
    vel = ti.Vector.field(2, dtype=ti.f32, shape=config.N_PARTICLES)
    draw_pos = ti.Vector.field(2, dtype=ti.f32, shape=config.N_PARTICLES)
    init_particles()

@ti.kernel
def init_particles():
    for i in range(config.N_PARTICLES):
        p = ti.Vector([ti.random(), ti.random()]) - 0.5
        p = 0.6 * p  
        pos[i] = p + 0.5

        v = ti.Vector([-p.y, p.x])
        vel[i] = 0.5 * v


@ti.kernel
def step(center_mass: ti.f32, center_soft: ti.f32):
    center = ti.Vector([0.5, 0.5])
    for i in range(config.N_PARTICLES):
        r = center - pos[i]
        dist2 = r.dot(r) + center_soft * center_soft
        inv_dist = ti.rsqrt(dist2)
        inv_dist3 = inv_dist * inv_dist * inv_dist

        acc = config.G * center_mass * r * inv_dist3
        vel[i] += acc * config.DT
        pos[i] += vel[i] * config.DT

@ti.kernel
def kick(center_x: ti.f32, center_y: ti.f32, strength: ti.f32):
    """
    在鼠标位置附近给粒子一个速度扰动（外力脉冲）。
    center_x, center_y 是 [0,1] 归一化坐标
    strength 控制扰动强度
    """
    c = ti.Vector([center_x, center_y])
    for i in range(config.N_PARTICLES):
        r = pos[i] - c
        dist2 = r.dot(r) + 1e-4
        w = ti.exp(-dist2 * 200.0)
        tang = ti.Vector([-r.y, r.x])
        vel[i] += strength * w * tang

@ti.kernel
def update_draw_pos(zoom: ti.f32):
    center = ti.Vector([0.5, 0.5])
    for i in range(config.N_PARTICLES):
        draw_pos[i] = (pos[i] - center) * zoom + center