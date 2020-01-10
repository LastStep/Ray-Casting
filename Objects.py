import math

import pygame as py


class Circle:

  def __init__(self, pos, radius, color):
    self.center = pos
    self.radius = radius
    self.color = color

  def show(self, screen):
    py.draw.circle(screen, self.color, self.center, self.radius)

  def intersect(self, point):
    return self.center[0]**2 - point.pos.x**2 + self.center[1]**2 - point.pos.y**2 <= self.radius**2


class Boundary:

  def __init__(self, x1, y1, x2, y2, color = (255,255,255)):
    self.startpoint, self.endpoint = py.math.Vector2(x1, y1), py.math.Vector2(x2, y2)
    self.color = color

  def line(self, screen):
    py.draw.line(screen, self.color, self.startpoint, self.endpoint, 2)


class Ray:

  def __init__(self, startpoint, endpoint, color):
    self.startpoint, self.endpoint = startpoint, endpoint
    self.color = color

  def show(self, screen):
    py.draw.line(screen, self.color, self.startpoint, self.endpoint, 2)


class Particle:

  def __init__(self, number_of_rays, ray_length, ray_color, particle_size, *pos):
    self.number_of_rays = number_of_rays
    self.ray_length = ray_length
    self.ray_color = ray_color
    self.size = particle_size
    self.pos = py.math.Vector2(pos)
    self.vel = py.math.Vector2()
    self.acc = py.math.Vector2()
    self.brain = None
    self.inputs = [math.inf]
    self.dead = False
    self.fitness = 0

    self.rays = get_rays(self.pos, self.number_of_rays, self.ray_length, self.ray_color)

  def show(self, screen):
    py.draw.circle(screen, (255, 255, 255), (int(self.pos[0]), int(self.pos[1])), self.size)

  def apply_force(self, force, max_speed):
    if not self.dead:
      self.acc += force
      self.update(max_speed)

  def update(self, max_speed):
    self.vel += self.acc
    self.pos += self.vel

    self.vel = py.math.Vector2.normalize(self.vel)
    self.vel *= max_speed

    self.acc = py.math.Vector2()
    self.rays = get_rays(self.pos, self.number_of_rays, self.ray_length, self.ray_color)

  def copy(self, number_of_rays, ray_length, ray_color, particle_size, *start_pos):
    new_particle = Particle(number_of_rays, ray_length, ray_color, particle_size, *start_pos)
    new_particle.brain = self.brain
    return  new_particle

  def check_death(self, dist, target):
    if dist < self.size or self.pos.distance_to(py.math.Vector2(target)) < self.size:
      self.dead = True

  def calculate_fitness(self, target):
    self.fitness = 1/self.pos.distance_to(py.math.Vector2(target))


def get_rays(startpoint, n, length, color):
  angle, rays = 0, list()
  for _ in range(n):
    angle += 360 / n
    x = math.cos(math.radians(angle)) * length
    y = math.sin(math.radians(angle)) * length
    endpoint = startpoint + py.math.Vector2(x, y)
    rays.append(Ray(startpoint, endpoint, color))
  return rays

def get_particles(start_pos, number_of_particles, number_of_rays, ray_length, ray_color, particle_size):
  particles = list()
  for _ in range(number_of_particles):
    particles.append(Particle(number_of_rays, ray_length, ray_color, particle_size, *start_pos))
  return particles

def get_boundaries(w, h, gap = 150):
  boundary = list()
  boundary.append(Boundary(w//2 - gap//2, 400, w//2 - gap//2, 800))
  boundary.append(Boundary(w//2 - gap//2, 400, w//2 + gap//2, 100))
  boundary.append(Boundary(w//2 + gap//2, 100, w, 100))

  boundary.append(Boundary(w//2 + gap//2, 450, w//2 + gap//2, 800))
  boundary.append(Boundary(w//2 + gap//2, 450, w//2 + 2*gap, 200))
  boundary.append(Boundary(w//2 + 2*gap, 200, w, 200))

  boundary.append(Boundary(w//2 - gap//2, h, w//2 + gap//2, h))
  boundary.append(Boundary(w, 100, w, 200))
  return boundary
