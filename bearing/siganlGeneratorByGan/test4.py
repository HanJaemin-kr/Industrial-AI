import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from sklearn.metrics import mean_squared_error

LENGTH_INPUT = 300
POPULATION_SIZE = 1000
MAX_GENERATIONS = 50

# Define hyperparameters
LATENT_DIM = 1000
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

# calculate the discriminator loss for real samples
def real_loss(discriminator, real_samples):
    # Predict probabilities for real samples
    y_pred = discriminator(real_samples)
    # Calculate binary cross-entropy loss
    loss = nn.BCELoss()(y_pred, torch.ones_like(y_pred))
    return loss

# calculate the discriminator loss for fake samples
def fake_loss(discriminator, fake_samples):
    # Predict probabilities for fake samples
    y_pred = discriminator(fake_samples)
    # Calculate binary cross-entropy loss
    loss = nn.BCELoss()(y_pred, torch.zeros_like(y_pred))
    return loss

# train the generator and discriminator
def train(generator, discriminator, real_samples, latent_dim, n_epochs, batch_size):

    # Move models to the specified device
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    # Create optimizer for generator and discriminator
    generator_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)

    # Lists for storing losses
    gen_losses = []
    disc_losses = []

    # Training loop
    for epoch in range(n_epochs):
        # Shuffle the real samples
        indices = torch.randperm(real_samples.shape[0])
        real_samples = real_samples[indices]

        # Number of batches
        num_batches = real_samples.shape[0] // batch_size

        for batch_idx in range(num_batches):
            # Sample real batch
            real_batch = real_samples[batch_idx * batch_size: (batch_idx + 1) * batch_size].to(device)

            # Generate fake samples
            fake_samples, _ = generate_fake_samples(generator, latent_dim, batch_size)

            # Reset gradients
            generator_optimizer.zero_grad()
            discriminator_optimizer.zero_grad()

            # Calculate discriminator loss for real and fake samples
            real_loss_val = real_loss(discriminator, real_batch)
            fake_loss_val = fake_loss(discriminator, fake_samples)

            # Calculate total discriminator loss
            discriminator_loss = real_loss_val + fake_loss_val

            # Update discriminator weights
            discriminator_loss.backward()
            discriminator_optimizer.step()

            # Generate new fake samples
            fake_samples, _ = generate_fake_samples(generator, latent_dim, batch_size)

            # Calculate generator loss
            target = torch.ones_like(fake_samples[:, :1])  # Create target tensor with the same shape as the output tensor
            generator_loss = nn.BCELoss()(discriminator(fake_samples[:, :1]), target)

            # Update generator weights
            generator_optimizer.zero_grad()
            generator_loss.backward()
            generator_optimizer.step()

        # Calculate average losses for the epoch
        avg_gen_loss = generator_loss.item()
        avg_disc_loss = discriminator_loss.item()

        gen_losses.append(avg_gen_loss)
        disc_losses.append(avg_disc_loss)

        # Print progress
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}], Generator Loss: {avg_gen_loss:.4f}, Discriminator Loss: {avg_disc_loss:.4f}")

    return generator, discriminator, gen_losses, disc_losses

# Set device
device = torch.device("mps:0")

# Create discriminator and generator instances
generator = Generator(LATENT_DIM, LENGTH_INPUT).to(device)
discriminator = Discriminator().to(device)

# Generate real samples
real_samples, _ = generate_faulty_samples(POPULATION_SIZE, fault_type='ball')
real_samples = real_samples.to(device)

# Train generator and discriminator
generator, discriminator, gen_losses, disc_losses = train(generator, discriminator, real_samples, LATENT_DIM, NUM_EPOCHS, BATCH_SIZE)

# Plot loss curves
plt.figure(figsize=(10, 5))
plt.plot(gen_losses, label='Generator Loss')
plt.plot(disc_losses, label='Discriminator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
