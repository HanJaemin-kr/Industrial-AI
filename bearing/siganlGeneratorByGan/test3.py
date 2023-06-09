import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from sklearn.metrics import mean_squared_error

LENGTH_INPUT = 300
POPULATION_SIZE = 100
MAX_GENERATIONS = 1000


# Define hyperparameters
LATENT_DIM = 500
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 2000

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
            nn.Linear(input_size, LENGTH_INPUT),
            nn.ReLU(),
            nn.Linear(LENGTH_INPUT, 250),
            nn.ReLU(),
            nn.Linear(250, 100),
            nn.ReLU(),
            nn.Linear(100, output_size),
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
    return torch.from_numpy(X1).float(), torch.from_numpy(y).float()

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
def calculate_fitness(chromosomes, generator, discriminator):
    fitness_scores = []
    for chromosome in chromosomes:
        # Set the generator parameters
        generator.load_state_dict(chromosome['state_dict'])  # Load the generator's state dictionary
        # Generate faulty samples
        real_samples, _ = generate_faulty_samples(POPULATION_SIZE, 'ball')
        real_samples = real_samples.to(device)

        # Generate fake samples
        fake_samples, _ = generate_fake_samples(generator, LATENT_DIM, POPULATION_SIZE)
        fake_samples = fake_samples.to(device)

        # Evaluate discriminator on real samples
        real_labels = torch.ones((POPULATION_SIZE, 1)).to(device)
        real_predictions = discriminator(real_samples)
        real_loss = criterion(real_predictions, real_labels)

        # Evaluate discriminator on fake samples
        fake_labels = torch.zeros((POPULATION_SIZE, 1)).to(device)
        fake_predictions = discriminator(fake_samples)
        fake_loss = criterion(fake_predictions, fake_labels)

        # Calculate fitness score
        fitness = (real_loss.item() + fake_loss.item()) / 2.0
        fitness_scores.append(fitness)
    return fitness_scores

# update the population of chromosomes
def update_population(population, scores):
    # select the top performing chromosomes
    selected = [x for _, x in sorted(zip(scores, population), key=lambda pair: pair[0])]
    # select top performing chromosomes for generating next generation
    selected = selected[:int(POPULATION_SIZE/2)]
    return selected

# crossover two parents to create two children
def crossover(parent1, parent2):
    # Get the model parameters from the parents
    parent1_params = parent1['state_dict']
    parent2_params = parent2['state_dict']
    child1_params = {}
    child2_params = {}
    # Perform crossover at the parameter level
    for param_name in parent1_params.keys():
        p1 = parent1_params[param_name]
        p2 = parent2_params[param_name]
        # Randomly select a crossover point
        cutoff = np.random.randint(0, len(p1.view(-1)))
        # Create child parameters using crossover
        child1_params[param_name] = torch.cat((p1.view(-1)[:cutoff], p2.view(-1)[cutoff:])).reshape(p1.shape)
        child2_params[param_name] = torch.cat((p2.view(-1)[:cutoff], p1.view(-1)[cutoff:])).reshape(p1.shape)
    # Create children chromosomes
    child1 = {'state_dict': child1_params}
    child2 = {'state_dict': child2_params}
    return child1, child2

# mutate a chromosome
def mutate(chromosome):
    # Get the model parameters from the chromosome
    chromosome_params = chromosome['state_dict']
    mutated_params = {}
    # Perform mutation at the parameter level
    for param_name in chromosome_params.keys():
        p = chromosome_params[param_name]
        # Randomly select a mutation point
        mutation_index = np.random.randint(0, len(p.view(-1)))
        # Apply mutation
        p.view(-1)[mutation_index] += np.random.normal()
        mutated_params[param_name] = p
    # Create mutated chromosome
    mutated_chromosome = {'state_dict': mutated_params}
    return mutated_chromosome

# define the main genetic algorithm function
def genetic_algorithm(generator, discriminator):
    # Initialize population
    population = []
    for _ in range(POPULATION_SIZE):
        chromosome = {'state_dict': generator.state_dict()}
        population.append(chromosome)

    best_fitness = float('inf')
    best_generation = 0

    # Start the evolution process
    for generation in range(MAX_GENERATIONS):
        print(f'Generation {generation + 1}/{MAX_GENERATIONS}')

        # Calculate fitness of each chromosome
        fitness_scores = calculate_fitness(population, generator, discriminator)

        # Update the best fitness and generation
        min_fitness = min(fitness_scores)
        if min_fitness < best_fitness:
            best_fitness = min_fitness
            best_generation = generation

        print(f'Best Fitness: {best_fitness:.4f} (Generation {best_generation + 1})')

        # Update the population
        population = update_population(population, fitness_scores)

        # Create next generation
        next_generation = []
        while len(next_generation) < POPULATION_SIZE:
            # Select parents for crossover
            parent1 = np.random.choice(population)
            parent2 = np.random.choice(population)
            # Perform crossover
            child1, child2 = crossover(parent1, parent2)
            # Perform mutation
            child1 = mutate(child1)
            child2 = mutate(child2)
            # Add children to next generation
            next_generation.append(child1)
            next_generation.append(child2)
        population = next_generation

    # Return the best generator model
    best_chromosome = population[0]
    best_generator = Generator(LATENT_DIM, LENGTH_INPUT).to(device)
    best_generator.load_state_dict(best_chromosome['state_dict'])
    return best_generator


# Set random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Set device
device = torch.device("mps")


# Initialize generator and discriminator
generator = Generator(LATENT_DIM, LENGTH_INPUT).to(device)
discriminator = Discriminator().to(device)

# Define loss function and optimizer
criterion = nn.BCELoss()
generator_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)

# Train the GAN using genetic algorithm
best_generator = genetic_algorithm(generator, discriminator)

# Generate fake samples using the best generator
num_fake_samples = 10
fake_samples, _ = generate_fake_samples(best_generator, LATENT_DIM, num_fake_samples)

# Move the fake_samples tensor to CPU and convert it to a NumPy array
fake_samples_np = fake_samples.cpu().detach().numpy()

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot the fake samples in the left subplot
for fake_sample in range(num_fake_samples):
    #fft_real_samples = np.fft.fft(fake_samples_np[fake_sample])
    ax1.plot(fake_samples_np[fake_sample])

ax1.set_title("Fake Samples")
ax1.set_xlabel("Time")
ax1.set_ylabel("Amplitude")

# Generate faulty samples
faulty_samples, _ = generate_faulty_samples(num_fake_samples, 'ball')

# Plot the faulty samples in the right subplot
for faulty_sample in range(num_fake_samples):
    ax2.plot(faulty_samples[faulty_sample])

ax2.set_title("Faulty Samples")
ax2.set_xlabel("Time")
ax2.set_ylabel("Amplitude")

plt.tight_layout()
plt.show()

# 저장할 모델 파일 경로
model_path = "/Users/hanjaemin/Desktop/Industrial-AI/bearing/siganlGeneratorByGan/gan_model2.pth"

# GAN 모델 저장
torch.save(best_generator.state_dict(), model_path)