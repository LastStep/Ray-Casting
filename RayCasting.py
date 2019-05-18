import pygame as py
from pygame.locals import *
import numpy as np
import math, os, random
py.init()
os.environ['SDL_VIDEO_CENTERED'] = '1'


class NeuralNetwork:
  def __init__(self, x, y, brain = None):
    if y == 0:
      self.input = np.array(x)
      self.weights1 = brain.weights1.copy()
      self.weights2 = brain.weights2.copy()
      self.y = brain.y.copy()
      self.output = np.zeros(self.y.shape)

    else:
      self.input = np.array(x)
      self.y = np.array(y)
      self.weights1 = np.random.rand(self.input.shape[1], brain)
      self.weights2 = np.random.rand(brain, self.y.shape[1])
      self.output = np.zeros(self.y.shape)

  def feedforward(self):
    self.layer1 = self.sigmoid(np.dot(self.input, self.weights1))
    self.output = self.sigmoid(np.dot(self.layer1, self.weights2))

  def backprop(self):
    delY = self.y - self.output
    d_weights2 = np.dot(self.layer1.T, (2 * delY * self.sigmoid_derivative(self.output)))
    d_weights1 = np.dot(self.input.T, (
          np.dot(2 * delY * self.sigmoid_derivative(self.output), self.weights2.T) * self.sigmoid_derivative(
        self.layer1)))
    self.weights1 += d_weights1
    self.weights2 += d_weights2

  @staticmethod
  def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))
  @staticmethod
  def sigmoid_derivative(x):
    return x * (1.0 - x)

  def mutate(self, mutation_rate):
    if np.random.random() < mutation_rate:
      rand = np.random.rand(2, self.weights1.shape[1])
      self.weights1 = np.vstack((self.weights1, rand))
      self.weights1 = np.delete(self.weights1, np.random.randint(0, 9), 0)
      self.weights1 = np.delete(self.weights1, np.random.randint(0, 8), 0)

  def run(self, n):
    for _ in range(n):
      self.feedforward()
      # self.backprop()


class Circle:

  def __init__(self, pos, radius, color):
    self.center = pos
    self.radius = radius
    self.color = color

  def show(self):
    py.draw.circle(screen, self.color, self.center, self.radius)

  def intersect(self, point):
    return self.center[0]**2 - point.pos.x**2 + self.center[1]**2 - point.pos.y**2 <= self.radius**2


class Boundary:

  def __init__(self, x1, y1, x2, y2, color = (255,255,255)):
    self.startpoint, self.endpoint = py.math.Vector2(x1, y1), py.math.Vector2(x2, y2)
    self.color = color

  def line(self):
    py.draw.line(screen, self.color, self.startpoint, self.endpoint, 2)


class Ray:

  def __init__(self, startpoint, endpoint, color):
    self.startpoint, self.endpoint = startpoint, endpoint
    self.color = color

  def show(self):
    py.draw.line(screen, self.color, self.startpoint, self.endpoint, 2)


class Particle:

  def __init__(self, *pos):
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

  def show(self):
    py.draw.circle(screen, (255, 255, 255), (int(self.pos[0]), int(self.pos[1])), self.size)

  def apply_force(self, force):
    if not self.dead:
      self.acc += force
      self.update()

  def update(self):
    self.vel += self.acc
    self.pos += self.vel
    self.vel = py.math.Vector2.normalize(self.vel)
    self.acc = py.math.Vector2()
    self.rays = get_rays(self.pos, self.number_of_rays, self.ray_length, self.ray_color)

  def copy(self):
    new_particle = Particle(*start_pos)
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

def get_boundaries():
  boundary, gap = list(), 150
  boundary.append(Boundary(w//2 - gap//2, 400, w//2 - gap//2, 800))
  boundary.append(Boundary(w//2 - gap//2, 400, w//2 + gap//2, 100))
  boundary.append(Boundary(w//2 + gap//2, 100, w, 100))

  boundary.append(Boundary(w//2 + gap//2, 450, w//2 + gap//2, 800))
  boundary.append(Boundary(w//2 + gap//2, 450, w//2 + 2*gap, 200))
  boundary.append(Boundary(w//2 + 2*gap, 200, w, 200))

  boundary.append(Boundary(w//2 - gap//2, h, w//2 + gap//2, h))
  boundary.append(Boundary(w, 100, w, 200))
  return boundary

def intersect(ray, boundary):
  p, r = ray.startpoint, ray.endpoint - ray.startpoint
  q, s = boundary.startpoint, boundary.endpoint - boundary.startpoint

  if r.cross(s) == 0:
    return False, None

  t = (q - p).cross(s) / (r.cross(s))
  u = (q - p).cross(r) / (r.cross(s))

  if r.cross(s) != 0 and 0 <= t <= 1 and 0 <= u <= 1:
    return True, p + t*r
  return False, None

def pick_one(fitness):
  particle = np.random.choice(particles, 1, p = fitness)[0]
  return particle.copy()

def calculate_fitness():
  total_fitness = 0
  for p in particles:
    p.calculate_fitness(end_pos)
    total_fitness += p.fitness

  for p in particles:
    p.fitness /= total_fitness

def next_generation():
  calculate_fitness()
  fitness = [p.fitness for p in particles]
  new_particles = []
  for i in range(number_of_particles):
    new_particles.append(pick_one(fitness))
  return new_particles

def run():
  # particles = [Particle(*py.mouse.get_pos())]
  population = particles.copy()
  while len(population) > 0:
    screen.fill((0, 0, 0))
    clock.tick(fps)
    for k,p in enumerate(population):
      if p.dead:
        population.remove(p)

      p.inputs.clear()

      for ray in p.rays:
        dist = p.ray_length
        temp = (False, p.ray_length)

        for boundary in boundaries:
          boundary.line()
          check = intersect(ray, boundary)
          if check[0]:
            tmp = dist
            dist = min(dist, ray.startpoint.distance_to(check[1]))
            if tmp != dist:
              temp = check

        if temp[0]:
          ray.endpoint = temp[1]
          if not p.dead:
            ray.show()

        p.check_death(dist, end_pos)
        p.inputs.append(np.interp(dist, [0, p.ray_length], [1, 0]))

      if p.brain is None:
        p.brain = NeuralNetwork([p.inputs], [[1]], 4)
      else:
        p.brain = NeuralNetwork([p.inputs], 0, p.brain)

      p.brain.mutate(mutation_rate)
      p.brain.run(1)
      force = py.math.Vector2(particle_speed) - p.vel
      # force = py.math.Vector2.normalize(force)
      output = np.interp(p.brain.output[0], [0, 1], [0, 360*4])
      # print(p.inputs)
      # print(k+1, output)
      p.apply_force(force.rotate(output))

    for p in particles:
      p.show()

    start.show()
    end.show()
    py.display.update()

def get_particles():
  particles = list()
  for i in range(number_of_particles):
    particles.append(Particle(*start_pos))
  return particles

if __name__ == '__main__':
  w, h = 1400, 800
  # flags = FULLSCREEN | DOUBLEBUF
  screen = py.display.set_mode((w+1, h+1))
  screen.set_alpha(None)
  fps = 60

  particle_speed = 2
  particle_size = 6
  number_of_particles = 50
  number_of_rays = 8
  ray_length = 80
  ray_color = (0, 0, 255)

  mutation_rate = 1

  start_pos = (700, 625)
  end_pos = (1300, 150)
  start = Circle(start_pos, 10, (255, 0, 0))
  end = Circle(end_pos, 10, (0, 255, 0))

  generation = 0
  boundaries = get_boundaries()
  particles = get_particles()

  clock = py.time.Clock()
  game = True
  while game:
    screen.fill((0, 0, 0))
    for event in py.event.get():
      if event.type == py.QUIT:
        game = False
        py.quit()
        quit()
      if event.type == py.MOUSEBUTTONDOWN:
        x, y = py.mouse.get_pos()
        print(x, y)

    generation += 1
    print('Generation {}'.format(generation))

    run()

    particles = next_generation()

    py.display.update()
    clock.tick(fps)
