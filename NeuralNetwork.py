import numpy as np


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
    #   self.backprop()


def pick_one(fitness, particles, mutation_rate, number_of_rays, ray_length, ray_color, particle_size, *start_pos):
  particle = np.random.choice(particles, 1, p = fitness)[0]
  particle =  particle.copy(number_of_rays, ray_length, ray_color, particle_size, *start_pos)
  particle.brain.mutate(mutation_rate)
  return particle

def calculate_fitness(particles, end_pos):
  total_fitness = 0
  for p in particles:
    p.calculate_fitness(end_pos)
    total_fitness += p.fitness

  for p in particles:
    p.fitness /= total_fitness

def next_generation(particles, number_of_particles, mutation_rate, number_of_rays, ray_length, ray_color, particle_size, end_pos, *start_pos):
  calculate_fitness(particles, end_pos)
  fitness = [p.fitness for p in particles]
  new_particles = []
  for _ in range(number_of_particles):
    new_particles.append(pick_one(fitness, particles, mutation_rate, number_of_rays, ray_length, ray_color, particle_size, *start_pos))
  return new_particles
