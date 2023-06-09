import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from sklearn.metrics import mean_squared_error

LENGTH_INPUT = 300
POPULATION_SIZE = 100
MAX_GENERATIONS = 100


# define the standalone discriminator model
class Discriminator(nn.Module):
    def __init__(self, n_inputs=LENGTH_INPUT):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_inputs, LENGTH_INPUT),
            nn.ReLU(),
            nn.Linear(LENGTH_INPUT, 250),
            nn.ReLU(),
            nn.Linear(250, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# define the standalone generator model
class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.ReLU(),
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


# generate n real samples with class labels
def generate_faulty_samples(n, fault_type):
    amps = np.arange(0.1, 10, 0.1)
    bias = np.arange(0.1, 10, 0.1)
    freqs = np.linspace(1, 2, 1000)
    X2 = np.linspace(-5, 5, LENGTH_INPUT)
    X1 = []

    for x in range(n):
        noise = np.random.normal(size=len(X2))

        if fault_type == 'ball':
            fault_frequency = 500
        elif fault_type == 'outer_race':
            fault_frequency = 1000
        elif fault_type == 'inner_race':
            fault_frequency = 1500
        elif fault_type == 'roller':
            fault_frequency = 2000
        else:
            raise ValueError("Invalid fault type. Supported types: 'ball', 'outer_race', 'inner_race', 'roller'")

        X1.append(
            np.random.choice(amps) * np.sin(X2 * fault_frequency) + np.random.choice(bias) + 0.3 * noise)
    X1 = np.array(X1).reshape(n, LENGTH_INPUT)
    # generate class labels
    y = np.ones((n, 1))
    return X1, y


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n):
    # generate points in the latent space
    x_input = np.random.randn(n, latent_dim)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n, latent_dim)
    return x_input


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n):
    # Generate points in latent space
    x_input = generate_latent_points(latent_dim, n)
    # Convert to PyTorch tensor
    x_input = torch.from_numpy(x_input).float().to(device)
    # Generate output
    X = generator(x_input)
    # Create fake labels
    y = torch.zeros((n, 1), device=device)
    return X, y


# calculate the fitness of each chromosome
def calculate_fitness(population, discriminator, latent_dim):
    fitness_scores = []

    for chromosome in population:
        generator = Generator(latent_dim, LENGTH_INPUT).to(device)
        state_dict = generator.state_dict()
        state_dict.update({"model.0.weight": chromosome})

        generator.load_state_dict(state_dict)

        # Training the GAN
        g_optimizer = optim.Adam(generator.parameters(), lr=0.001)
        d_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)
        real_samples, _ = generate_faulty_samples(POPULATION_SIZE, fault_type='ball')
        real_samples = torch.from_numpy(real_samples).float().to(device)

        for _ in range(10):
            fake_samples, _ = generate_fake_samples(generator, latent_dim, POPULATION_SIZE)

            d_optimizer.zero_grad()
            d_real = discriminator(real_samples)
            d_fake = discriminator(fake_samples.detach())
            d_loss = -(torch.mean(torch.log(d_real)) + torch.mean(torch.log(1 - d_fake)))
            d_loss.backward()
            d_optimizer.step()

            g_optimizer.zero_grad()
            g_fake = discriminator(fake_samples)
            g_loss = -torch.mean(torch.log(g_fake))
            g_loss.backward()
            g_optimizer.step()

        generated_samples, _ = generate_fake_samples(generator, latent_dim, POPULATION_SIZE)
        generated_samples = torch.where(torch.isnan(generated_samples), torch.zeros_like(generated_samples),
                                        generated_samples)
        real_samples = torch.where(torch.isnan(real_samples), torch.zeros_like(real_samples), real_samples)
        mse = mean_squared_error(real_samples.cpu().detach().numpy(), generated_samples.cpu().detach().numpy())
        fitness_scores.append(1 / (mse + 1e-8))

    return fitness_scores

# perform crossover between parents to produce offspring
def crossover(parent1, parent2):
    child1 = parent1.clone()  # Use clone() instead of copy()
    child2 = parent2.clone()  # Use clone() instead of copy()
    crossover_point = np.random.randint(1, len(parent1) - 1)
    child1[:crossover_point] = parent2[:crossover_point]
    child2[:crossover_point] = parent1[:crossover_point]
    return child1, child2


# perform mutation on the offspring
def mutation(chromosome, mutation_rate):
    mutated_chromosome = chromosome.clone()  # Use clone() instead of copy()
    for i in range(len(mutated_chromosome)):
        if np.random.rand() < mutation_rate:
            mutated_chromosome[i] = torch.randn(1)  # Example mutation operation
    return mutated_chromosome


# select parents based on their fitness scores
def selection(population, fitness_scores, tournament_size):
    selected_parents = []
    parents_array = np.array(population)  # Convert parents to a NumPy array
    for _ in range(len(population)):
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        selected_parent_index = tournament_indices[np.argmax(tournament_fitness)]
        selected_parents.append(parents_array[selected_parent_index])
    return selected_parents



# initialize population with random chromosomes
def initialize_population(population_size, latent_dim):
    population = []
    for _ in range(population_size):
        chromosome = torch.randn(latent_dim, 100)  # Set the size of the chromosome to [latent_dim, 100]
        population.append(chromosome)
    return population


# main genetic algorithm
def genetic_algorithm(discriminator, latent_dim):
    population = initialize_population(POPULATION_SIZE, latent_dim)
    best_fitness = None
    best_chromosome = None

    for generation in range(MAX_GENERATIONS):
        fitness_scores = calculate_fitness(population, discriminator, latent_dim)

        max_fitness = max(fitness_scores)
        max_index = fitness_scores.index(max_fitness)
        if best_fitness is None or max_fitness > best_fitness:
            best_fitness = max_fitness
            best_chromosome = population[max_index]

        print(f"Generation {generation + 1}: Best Fitness = {best_fitness}")

        parents = selection(population, fitness_scores, tournament_size=2)

        offspring = []
        while len(offspring) < len(population):
            parent1, parent2 = np.random.choice(parents, size=2, replace=False)
            child1, child2 = crossover(parent1, parent2)
            mutated_child1 = mutation(child1, mutation_rate=0.1)
            mutated_child2 = mutation(child2, mutation_rate=0.1)
            offspring.extend([mutated_child1, mutated_child2])

        population = offspring

    best_chromosome = best_chromosome.reshape(100, 100)

    return best_chromosome



# Set device
device = torch.device("mps")

# Create discriminator and generator instances
discriminator = Discriminator().to(device)
latent_dim = 100

# Train the GAN using the genetic algorithm
best_chromosome = genetic_algorithm(discriminator, latent_dim)

# Generate samples using the best chromosome
best_generator = Generator(latent_dim, LENGTH_INPUT).to(device)
state_dict = {
    "model.0.weight": best_chromosome,
    "model.0.bias": torch.zeros(LENGTH_INPUT),
    "model.2.weight": torch.zeros(250, LENGTH_INPUT),
    "model.2.bias": torch.zeros(250),
    "model.4.weight": torch.zeros(100, 250),
    "model.4.bias": torch.zeros(100),
    "model.6.weight": torch.zeros(1, 100),
    "model.6.bias": torch.zeros(1)
}
# 저장할 경로 설정
save_path = "./best_generator_model.pth"

# 상태 사전(state_dict) 저장
torch.save(best_generator.state_dict(), save_path)

# Generate a signal using the best chromosome
generated_signal, _ = generate_fake_samples(best_generator, latent_dim=100, n=1)
generated_signal = generated_signal.cpu().detach().numpy()[0]
print(generated_signal)
fft_fake_samples = np.fft.fft(generated_signal)
# Plot the generated signal

plt.subplot(1, 2, 2)
plt.plot(np.abs(fft_fake_samples), '-', label='Random Fake FFT Sample', color='firebrick')
plt.title('FFT signal')
plt.legend(fontsize=10)

plt.subplot(1, 2, 1)
plt.plot(generated_signal, '-', label='Random Fake Sample', color='firebrick')
plt.title('Signal')
plt.legend(fontsize=10)
plt.show()


