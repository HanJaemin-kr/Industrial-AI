import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from sklearn.metrics import mean_squared_error

LENGTH_INPUT = 300
POPULATION_SIZE = 50
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
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
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
    x_input = torch.from_numpy(x_input).float()
    # Generate output
    X = generator(x_input)
    # Create fake labels
    y = torch.zeros((n, 1))
    return X, y


# calculate the fitness of each chromosome
def calculate_fitness(chromosomes, generator, discriminator, generation):
    fitness_scores = []
    for chromosome in chromosomes:
        generation += 1

        # Set the generator parameters
        generator.load_state_dict(chromosome)
        # Train the GAN with the current generator parameters
        train(generator, discriminator, latent_dim, generation, n_epochs=1000, n_batch=128, n_eval=200)
        # Calculate the fitness score based on GAN's performance
        fitness = evaluate_gan_performance(generator)
        fitness_scores.append(fitness)
    return fitness_scores


def evaluate_gan_performance(generator):
    real_samples, _ = generate_faulty_samples(1000, 'ball')
    real_samples = torch.from_numpy(real_samples).float()

    fake_samples, _ = generate_fake_samples(generator, latent_dim, 1000)
    fake_samples = fake_samples.detach().numpy()

    # Measure the performance using MSE
    performance = mean_squared_error(real_samples, fake_samples)
    return performance


# train the generator and discriminator
def train(g_model, d_model, latent_dim, cur_generation, n_epochs=10000, n_batch=128, n_eval=200):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g_model.to(device)
    d_model.to(device)

    # determine the size of one batch, for updating the discriminator
    half_batch = int(n_batch / 2)
    # define loss function and optimizers
    adversarial_loss = nn.BCELoss()
    d_optimizer = optim.Adam(d_model.parameters(), lr=0.0002)
    g_optimizer = optim.Adam(g_model.parameters(), lr=0.0002)

    # manually enumerate epochs
    for i in range(n_epochs):
        # train discriminator
        for _ in range(2):
            # prepare real samples
            x_real, y_real = generate_faulty_samples(half_batch, 'ball')
            real_samples = torch.Tensor(x_real).to(device)
            real_labels = torch.Tensor(y_real).to(device)
            # prepare fake examples
            x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            fake_samples = torch.Tensor(x_fake).to(device)
            fake_labels = torch.Tensor(y_fake).to(device)
            # train discriminator
            d_optimizer.zero_grad()
            real_output = d_model(real_samples)
            fake_output = d_model(fake_samples)
            real_loss = adversarial_loss(real_output, real_labels[:half_batch])
            fake_loss = adversarial_loss(fake_output, fake_labels[:half_batch])
            d_loss = real_loss + fake_loss
            d_loss.backward()
            d_optimizer.step()

        # train generator
        g_optimizer.zero_grad()
        gen_samples, _ = generate_fake_samples(g_model, latent_dim, n_batch)
        gen_samples = torch.Tensor(gen_samples).to(device)
        gan_output = d_model(gen_samples)
        g_loss = adversarial_loss(gan_output[:half_batch], real_labels[:half_batch])
        g_loss.backward()
        g_optimizer.step()

        # evaluate the model every n_eval epochs
        if (i + 1) % n_eval == 0:
            plt.figure(figsize=(12, 5))
            plt.title('Number of epochs = %i' % (i + 1))
            with torch.no_grad():
                gen_samples, _ = generate_fake_samples(g_model, latent_dim, latent_dim)
                real_samples, _ = generate_faulty_samples(latent_dim, 'ball')

                real_samples_tensor = torch.Tensor(real_samples)  # numpy.ndarray를 torch.Tensor로 변환
                fft_real_samples = torch.fft.fft(real_samples_tensor)

                gen_samples_tensor = torch.Tensor(gen_samples)  # numpy.ndarray를 torch.Tensor로 변환
                fft_fake_samples = torch.fft.fft(gen_samples_tensor)

                plt.subplot(1, 2, 2)
                plt.plot(np.abs(fft_fake_samples[0]), '-', label='Random Fake FFT Sample', color='firebrick')
                plt.plot(np.abs(fft_real_samples[0]), '-', label='Random Real FFT Sample', color='navy')
                plt.title('FFT signal')
                plt.legend(fontsize=10)

                plt.subplot(1, 2, 1)
                plt.plot(gen_samples[0], '-', label='Random Fake Sample ', color='firebrick')
                plt.plot(real_samples[0], '-', label='Random Real Sample ', color='navy')
                plt.title('signal')
                plt.legend(fontsize=10)

            plt.legend(fontsize=10)
            plt.tight_layout()
            plt.savefig(f'img/ball-graph_g-{generation}_epoch-{i + 1}.png')
            plt.close()

# perform turnament selection to choose parents for crossover
def tournament_selection(chromosomes, fitness_scores, tournament_size):
    selected_parents = []
    for _ in range(len(chromosomes)):
        # Randomly select individuals for the tournament
        tournament_indices = np.random.choice(len(chromosomes), tournament_size, replace=False)
        tournament_fitness_scores = [fitness_scores[i] for i in tournament_indices]
        # Select the fittest individual from the tournament
        selected_index = tournament_indices[np.argmax(tournament_fitness_scores)]
        selected_parents.append(chromosomes[selected_index])
    return selected_parents

# perform crossover between parents to create offspring
def crossover(parents):
    offspring = []
    for i in range(0, len(parents), 2):
        parent1 = parents[i]
        parent2 = parents[i+1]
        # Perform crossover operation (e.g., blend crossover, uniform crossover, etc.)
        child = {}
        for key in parent1.keys():
            if np.random.rand() < 0.5:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        offspring.append(child)
    return offspring

# perform mutation on the offspring
def mutate(offspring, mutation_rate):
    mutated_offspring = []
    for chromosome in offspring:
        if np.random.random() < mutation_rate:
            # Perform mutation operation (e.g., random perturbation, flip bit, etc.)
            mutated_chromosome = {}
            for key in chromosome.keys():
                mutated_chromosome[key] = chromosome[key] + np.random.normal(0, 0.1, size=chromosome[key].shape)
            mutated_offspring.append(mutated_chromosome)
        else:
            mutated_offspring.append(chromosome)
    return mutated_offspring

device = torch.device("mps")

latent_dim = 100  # Update with your desired latent dimension
output_size = LENGTH_INPUT
generator = Generator(latent_dim, output_size)

discriminator = Discriminator(n_inputs=LENGTH_INPUT)


gan_model = nn.Sequential(generator, discriminator)

# initialize the population
population = [generator.state_dict() for _ in range(POPULATION_SIZE)]

# evolution loop
for generation in range(MAX_GENERATIONS):
    print(' == 에인 반복문에서 ',generation, '번째 실행 ===')
    # calculate fitness scores
    fitness_scores = calculate_fitness(population, generator, discriminator, generation)
    # perform tournament selection to choose parents
    parents = tournament_selection(population, fitness_scores, tournament_size=5)
    # perform crossover to create offspring
    offspring = crossover(parents)
    # perform mutation on the offspring
    mutated_offspring = mutate(offspring, mutation_rate=0.1)
    # replace the population with the new generation (offspring + mutated offspring)
    population = offspring + mutated_offspring

    print(f"Generation: {generation + 1}")

    # train the generator and discriminator with the current generation information
    train(generator, discriminator, latent_dim, MAX_GENERATIONS + 1, n_epochs=1000, n_batch=128, n_eval=200)
