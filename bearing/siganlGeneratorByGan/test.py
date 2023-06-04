from numpy import hstack
import numpy as np
from numpy import zeros
from numpy import ones
from numpy.random import rand
from numpy.random import randn
from keras.models import Sequential
from keras import Input
from keras.layers import Dense, LSTM
from matplotlib import pyplot
import matplotlib.pyplot as plt

LENGTH_INPUT = 300
POPULATION_SIZE = 50
MAX_GENERATIONS = 100

# define the standalone discriminator model
def define_discriminator(n_inputs=LENGTH_INPUT):
    model = Sequential()
    model.add(Dense(LENGTH_INPUT, activation='relu', input_dim=n_inputs))
    model.add(Dense(250, activation='relu', input_dim=n_inputs))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# define the standalone generator model
def define_generator(latent_dim, n_outputs=LENGTH_INPUT):
    model = Sequential()
    model.add(Input(shape=(latent_dim, 1)))
    model.add(LSTM(150))
    model.add(Dense(LENGTH_INPUT, activation='linear'))
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

    return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(generator, discriminator):
    # make weights in the discriminator not trainable
    discriminator.trainable = False
    # connect them
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

# generate n real samples with class labels
def generate_real_samples(n):
    amps = np.arange(0.1, 10, 0.1)
    bias = np.arange(0.1, 10, 0.1)
    freqs = np.linspace(1, 2, 1000)
    X2 = np.linspace(-5, 5, LENGTH_INPUT)
    X1 = []
    for x in range(n):
        noise = np.random.normal(size=len(X2))
        X1.append(
            np.random.choice(amps) * np.sin(X2 * np.random.choice(freqs)) + np.random.choice(bias) 
+ 0.3 * noise)
    X1 = np.array(X1).reshape(n, LENGTH_INPUT)
    # generate class labels
    y = ones((n, 1))
    return X1, y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n):
    # generate points in the latent space
    x_input = randn(latent_dim * n)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n, latent_dim)
    return x_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n):
    # generate points in latent space
    x_input = generate_latent_points(latent_dim, n)
    # predict outputs
    X = generator.predict(x_input, verbose=0)
    # create class labels
    y = zeros((n, 1))
    return X, y

# calculate the fitness of each chromosome
def calculate_fitness(chromosomes, generator, discriminator):
    fitness_scores = []
    for chromosome in chromosomes:
        # Set the generator parameters
        generator.set_weights(chromosome)
        # Train the GAN with the current generator parameters
        train(generator, discriminator, gan_model, latent_dim, n_epochs=1000, n_batch=128, 
n_eval=200)
        # Calculate the fitness score based on GAN's performance
        fitness = evaluate_gan_performance(generator)
        fitness_scores.append(fitness)
    return fitness_scores

# evaluate the performance of the GAN
def evaluate_gan_performance(generator):
    # Generate fake samples
    fake_samples, _ = generate_fake_samples(generator, latent_dim, 1000)
    # Measure the performance (e.g., MSE, accuracy, etc.)
    performance = ...
    return performance

# perform tournament selection to choose parents for crossover
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
        child = ...
        offspring.append(child)
    return offspring

# perform mutation on the offspring
def mutate(offspring, mutation_rate):
    mutated_offspring = []
    for chromosome in offspring:
        if np.random.random() < mutation_rate:
            # Perform mutation operation (e.g., random perturbation, flip bit, etc.)
            mutated_chromosome = ...
            mutated_offspring.append(mutated_chromosome)
        else:
            mutated_offspring.append(chromosome)
    return mutated_offspring

# train the generator and discriminator
def train(g_model, d_model, gan_model, latent_dim, n_epochs=10000, n_batch=128, n_eval=200):
    # determine half the size of one batch, for updating the discriminator
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_epochs):
        # prepare real samples
        x_real, y_real = generate_real_samples(half_batch)
        # prepare fake examples
        x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        # update discriminator
        d_model.train_on_batch(x_real, y_real)
        d_model.train_on_batch(x_fake, y_fake)
        # prepare points in latent space as input for the generator
        x_gan = generate_latent_points(latent_dim, n_batch)
        # create inverted labels for the fake samples
        y_gan = ones((n_batch, 1))
        # update the generator via the discriminator's error
        gan_model.train_on_batch(x_gan, y_gan)
        # evaluate the model every n_eval epochs
        if (i+1) % n_eval == 0:
            plt.title('Number of epochs = %i'%(i+1))
            pred_data = generate_fake_samples(generator,latent_dim,latent_dim)[0]
            real_data  = generate_real_samples(latent_dim)[0]
            plt.plot(pred_data[0],'.',label='Random Fake Sample',color='firebrick')
            plt.plot(real_data[0],'.',label = 'Random Real Sample',color='navy')
            plt.legend(fontsize=10)
            plt.savefig(f'img/graph_{i+1}.png')
            plt.close()

# size of the latent space
latent_dim = 3
# create the discriminator
discriminator = define_discriminator()
# create the generator
generator = define_generator(latent_dim)
# create the gan
gan_model = define_gan(generator, discriminator)

# initialize the population
population = [generator.get_weights() for _ in range(POPULATION_SIZE)]

# evolution loop
for generation in range(MAX_GENERATIONS):
    # calculate fitness scores
    fitness_scores = calculate_fitness(population, generator, discriminator)
    # perform tournament selection to choose parents
    parents = tournament_selection(population, fitness_scores, tournament_size=5)
    # perform crossover to create offspring
    offspring = crossover(parents)
    # perform mutation on the offspring
    mutated_offspring = mutate(offspring, mutation_rate=0.1)
    # replace the population with the new generation (offspring + mutated offspring)
    population = offspring + mutated_offspring
    # Print generation information
    print(f"Generation: {generation+1}")
