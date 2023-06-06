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


# calculate the fitness score for a set of chromosomes
def calculate_fitness(chromosomes, gan_model, generation):
    fitness_scores = []
    for chromosome in chromosomes:
        latent_dim = chromosome.shape[0]
        generator, discriminator = gan_model
        # Resize or reshape the chromosome to match the expected shape
        resized_chromosome = np.tile(chromosome, (128, 1))

        # Create a new state dictionary
        state_dict = {
            'model.0.weight': torch.from_numpy(resized_chromosome).float(),
            'model.0.bias': generator.model[0].bias,
            'model.2.weight': generator.model[2].weight,
            'model.2.bias': generator.model[2].bias,
            'model.4.weight': generator.model[4].weight,
            'model.4.bias': generator.model[4].bias,
            'model.6.weight': generator.model[6].weight,
            'model.6.bias': generator.model[6].bias
        }

        # Update generator's weights
        generator.load_state_dict(state_dict)

        # Generate fake samples
        X, y = generate_fake_samples(generator, latent_dim, POPULATION_SIZE)
        # Evaluate GAN performance
        mse = evaluate_gan_performance(X, discriminator)
        fitness_scores.append(1 / mse)
    return fitness_scores


# evaluate the GAN performance
def evaluate_gan_performance(samples, discriminator):
    y_pred = discriminator(samples)
    y_true = torch.zeros(samples.size(0), 1)
    mse = mean_squared_error(y_true.detach().numpy(), y_pred.detach().numpy())
    return mse


# create a random sample of input noise for the generator
def generate_noise(latent_dim, n):
    x_input = np.random.randn(latent_dim * n)
    x_input = x_input.reshape(n, latent_dim)
    return x_input


# train the generator and discriminator
def train(gan_model, n_epochs, n_eval=10):
    generator, discriminator = gan_model
    latent_dim = generator.model[0].in_features
    population = [generate_noise(latent_dim, 1) for _ in range(POPULATION_SIZE)]
    best_chromosome = None
    best_fitness = float('-inf')
    all_fitness = []

    for generation in range(n_epochs):
        fitness_scores = calculate_fitness(population, gan_model, generation)
        all_fitness.append(max(fitness_scores))

        if max(fitness_scores) > best_fitness:
            best_fitness = max(fitness_scores)
            best_chromosome = population[np.argmax(fitness_scores)]

        if (generation + 1) % n_eval == 0:
            print(f"Generation {generation + 1}, Best Fitness: {best_fitness}")

            # Generate and save the graph
            plt.figure(figsize=(12, 5))
            plt.title(f'Generation {generation + 1}')
            with torch.no_grad():
                gen_samples, _ = generate_fake_samples(generator, latent_dim, 1)
                real_samples, _ = generate_faulty_samples(latent_dim, 'ball')

                real_samples_tensor = torch.Tensor(real_samples)
                fft_real_samples = torch.fft.fft(real_samples_tensor)

                gen_samples_tensor = torch.Tensor(gen_samples)
                fft_fake_samples = torch.fft.fft(gen_samples_tensor)

                plt.subplot(1, 2, 2)
                plt.plot(np.abs(fft_fake_samples[0]), '-', label='Random Fake FFT Sample', color='firebrick')
                plt.plot(np.abs(fft_real_samples[0]), '-', label='Random Real FFT Sample', color='navy')
                plt.title('FFT signal')
                plt.legend(fontsize=10)

                plt.subplot(1, 2, 1)
                plt.plot(gen_samples[0], '-', label='Random Fake Sample', color='firebrick')
                plt.plot(real_samples[0], '-', label='Random Real Sample', color='navy')
                plt.title('Signal')
                plt.legend(fontsize=10)

            plt.legend(fontsize=10)
            plt.tight_layout()
            plt.savefig(f'img/Ver2-ball-graph_g-{generation + 1}.png')
            plt.close()

        # Select parents for reproduction
        parents = np.random.choice(population, size=POPULATION_SIZE // 2, replace=False)
        # Perform crossover
        offspring = []
        for parent in parents:
            index = np.random.randint(0, latent_dim)
            offspring.append(np.concatenate((best_chromosome[:index], parent[index:])))
        offspring = parents + offspring
        # Update population with offspring
        population = offspring

    # Save the generator model
    torch.save(generator.state_dict(), 'generator_model.pth')

    return best_chromosome

# define the GAN model
def define_gan(latent_dim, output_size):
    generator = Generator(latent_dim, output_size)
    discriminator = Discriminator()
    gan_model = (generator, discriminator)
    return gan_model


# define the number of dimensions in the latent space
latent_dim = 10
output_size = LENGTH_INPUT

# define the GAN model
gan_model = define_gan(latent_dim, output_size)

# set the device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator, discriminator = gan_model
generator = generator.to(device)
discriminator = discriminator.to(device)

# train the GAN
best_chromosome = train(gan_model, MAX_GENERATIONS)

print("Best chromosome:")
print(best_chromosome)

# Save the generator model
torch.save(generator.state_dict(), 'generator_model.pth')
