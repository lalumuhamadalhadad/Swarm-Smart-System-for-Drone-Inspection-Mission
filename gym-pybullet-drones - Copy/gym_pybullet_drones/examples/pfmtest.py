import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

class Obstacle:
    DEFAULT_DIAM = 10
    DEFAULT_CHARGE = -1
    DEFAULT_MASS = 10

    def __init__(self, position, charge=DEFAULT_CHARGE, diam=DEFAULT_DIAM):
        self.p = np.array(position, dtype=float)
        self.charge = charge
        self.diam = diam

    def distance(self, robot):
        d = np.linalg.norm(self.p - np.array([robot.x, robot.y])) - (self.diam + robot.diam) / 2
        return max(d, 0.0000001)  # avoid division by zero

    def distance_sq(self, robot):
        d = self.distance(robot)
        return d * d

class Robot:
    def __init__(self, position, obstacles, dt=0.1, mass=40, f_max=4000, diam=15):
        self.x, self.y = position
        self.vx, self.vy = 0.0, 0.0
        self.dt = dt
        self.mass = mass
        self.f_max = f_max
        self.diam = diam
        self.obstacles = obstacles
        self.target = obstacles[0]  # first obstacle is the goal
        self.virtualforce = False

    def update_position(self):
        dir_x, dir_y = 0.0, 0.0
        min_safety = 200

        for ob in self.obstacles:
            dist_sq = ob.distance_sq(self)
            dx = ob.charge * (ob.p[0] - self.x) / dist_sq
            dy = ob.charge * (ob.p[1] - self.y) / dist_sq
            dir_x += dx
            dir_y += dy

        norm = np.linalg.norm([dir_x, dir_y])
        if norm != 0:
            dir_x /= norm
            dir_y /= norm

        for ob in self.obstacles:
            if not self.in_range(ob, 1200):
                continue
            dist_sq = ob.distance_sq(self)
            dx = (ob.p[0] - self.x) + np.random.normal(0, 1)
            dy = (ob.p[1] - self.y) + np.random.normal(0, 1)
            safety = dist_sq / (dx * dir_x + dy * dir_y + 0.000001)
            if 0 < safety < min_safety:
                min_safety = safety

        if min_safety < 5:
            self.target.charge *= min_safety / 5
        elif min_safety > 100:
            self.target.charge *= min_safety / 100

        vt_norm = min_safety / 2
        vtx = vt_norm * dir_x
        vty = vt_norm * dir_y

        fx = self.mass * (vtx - self.vx) / self.dt
        fy = self.mass * (vty - self.vy) / self.dt

        f_norm = np.linalg.norm([fx, fy])
        if f_norm > self.f_max:
            fx *= self.f_max / f_norm
            fy *= self.f_max / f_norm

        self.vx += (fx * self.dt) / self.mass
        self.vy += (fy * self.dt) / self.mass

        # Virtual force component
        if self.virtualforce and (self.target.charge < 1000) and (self.x > 25) and (self.y > 25):
            self.target.charge *= min_safety / 100
            self.vx += 5

        self.x += self.vx * self.dt
        self.y += self.vy * self.dt

    def in_range(self, ob, range_val):
        return ob.distance_sq(self) < range_val * range_val

# Setup simulation
obstacles = [
    Obstacle((0, 100), +100, 3),  # Target (goal)
    Obstacle((400, 300), -100, 20),  # Obstacle 1
    Obstacle((600, 500), -100, 20),  # Obstacle 2
    Obstacle((200, 200), -100, 20)   # Obstacle 3
]

robot = Robot((800, 600), obstacles)

# Visualization
fig, ax = plt.subplots()
ax.set_xlim(-100, 1000)
ax.set_ylim(-100, 800)
robot_dot, = ax.plot([], [], 'ro', markersize=10)
trace, = ax.plot([], [], 'b.', markersize=2)
obstacle_circles = []

for ob in obstacles:
    circle = plt.Circle((ob.p[0], ob.p[1]), ob.diam / 2, color='black')
    ax.add_artist(circle)
    obstacle_circles.append(circle)

path_x, path_y = [], []

def init():
    robot_dot.set_data([], [])
    trace.set_data([], [])
    return robot_dot, trace

def animate(frame):
    robot.update_position()
    path_x.append(robot.x)
    path_y.append(robot.y)

    robot_dot.set_data(robot.x, robot.y)
    trace.set_data(path_x, path_y)
    return robot_dot, trace

ani = animation.FuncAnimation(fig, animate, init_func=init, frames=500, interval=50, blit=True)
plt.title('Virtual Force Robot Simulation')
plt.show()
