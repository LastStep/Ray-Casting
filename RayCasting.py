import math
import os

import numpy as np
import pygame as py
from pygame.locals import *

import NeuralNetwork as NN
import Objects as obj

py.init()
os.environ['SDL_VIDEO_CENTERED'] = '1'



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


def run(screen):
  # particles = [Particle(*py.mouse.get_pos())]
  population = particles.copy()
  while len(population) > 0:
    screen.fill((0, 0, 0))
    clock.tick(fps)
    for _,p in enumerate(population):
      if p.dead:
        population.remove(p)

      p.inputs.clear()

      for ray in p.rays:
        dist = p.ray_length
        temp = (False, p.ray_length)

        for boundary in boundaries:
          boundary.line(screen)
          check = intersect(ray, boundary)
          if check[0]:
            tmp = dist
            dist = min(dist, ray.startpoint.distance_to(check[1]))
            if tmp != dist:
              temp = check

        if temp[0]:
          ray.endpoint = temp[1]
          if not p.dead:
            ray.show(screen)

        p.check_death(dist, end_pos)
        p.inputs.append(np.interp(dist, [0, p.ray_length], [1, 0]))

      if p.brain is None:
        p.brain = NN.NeuralNetwork([p.inputs], [[1]], 4)
      else:
        p.brain = NN.NeuralNetwork([p.inputs], 0, p.brain)

      p.brain.run(1)
      angle = np.interp(p.brain.output[0], [0, 1], [0, 360*4])
      force = py.math.Vector2(max_speed).rotate(angle) - p.vel
      # print(p.inputs)
      # print(k+1, output)
      p.apply_force(force, max_speed)

    for p in particles:
      p.show(screen)
      ximg, yimg = p.pos
      ximg, yimg = ximg - 20, yimg - 20
      dispImg(screen, particleImg, ximg, yimg)


    start.show(screen)
    end.show(screen)
    py.display.update()


def dispImg(screen, img, x, y):
    screen.blit(img, (x, y))

if __name__ == '__main__':
  particleImg = py.image.load('particle.png')
  particleImg = py.transform.scale(particleImg, (40, 40))

  w, h = 1400, 800
  # flags = FULLSCREEN | DOUBLEBUF
  screen = py.display.set_mode((w+1, h+1))
  screen.set_alpha(None)
  fps = 60

  max_speed = 4
  particle_size = 6
  number_of_particles = 50
  number_of_rays = 8
  ray_length = 80
  ray_color = (0, 0, 255)

  mutation_rate = 1

  start_pos = (700, 625)
  end_pos = (1300, 150)
  start = obj.Circle(start_pos, 10, (255, 0, 0))
  end = obj.Circle(end_pos, 10, (0, 255, 0))

  generation = 0
  boundaries = obj.get_boundaries(w, h)
  particles = obj.get_particles(start_pos, number_of_particles, number_of_rays, ray_length, ray_color, particle_size)

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

    run(screen)

    particles = NN.next_generation(particles, number_of_particles, mutation_rate, number_of_rays, ray_length, ray_color, particle_size, end_pos, *start_pos)

    py.display.update()
    clock.tick(fps)
