import random
import abc

import numpy as np

class GeneticAlgorithm:
    def __init__(self, gene_length, population_size, use_mutations, mutation_rate, random_variation, keep_best_gene, maximize_score):
        self.gene_length = gene_length
        self.population_size = population_size
        self.use_mutations = use_mutations
        self.mutation_rate = mutation_rate
        self.random_variation = random_variation
        self.keep_best_gene = keep_best_gene
        self.maximize_score = maximize_score

        self.population = []
        self.configure_next_generation()

    @abc.abstractmethod
    def score_gene(self, gene):
        pass

    def configure_next_generation(self):
        pass

    def add_gene_to_pool(self, gene):
        score = self.score_gene(gene)
        self.population.append([gene, score])

    def complete_population_with_random_genes(self):
        for _i in range(self.population_size - len(self.population)):
            gene = np.random.rand(self.gene_length) * 2 - 1
            score = self.score_gene(gene)
            self.population.append([gene, score])

    def evolve(self, generations):
        # Genetic algorithm
        for generation in range(generations):
            # Do any setup for the new generation
            self.configure_next_generation()

            # Get stdev of chromosomes - to provide a scale for random variation
            chromosome_range = 0
            for gene, score in self.population:
                chromosome_range += np.std(gene)
            chromosome_range /= self.population_size

            # Create and evaluate the next generation - first create new genes by mutation and replace parent if better
            new_genes = []
            if self.use_mutations:
                for gene, score in self.population:
                    new_gene = [c + (random.random() - 0.5) * chromosome_range * self.random_variation for c in gene]
                    new_genes.append(new_gene)

            # Now breed genes
            new_genes = []
            for idx in range(2, self.population_size):
                breeding_pair = random.sample(self.population[:idx], 2)
                parent1 = breeding_pair[0][0]
                parent2 = breeding_pair[1][0]
                # Create one child gene where the individual genes are mixed and one which is an interpolation
                child_gene_interleave = [0] * self.gene_length
                child_gene_scale = [0] * self.gene_length
                child_gene_splice = [0] * self.gene_length
                splice_point = random.randint(0, self.gene_length)
                splice_factor = 1.2 * random.random() - 0.1
                for j in range(self.gene_length):
                    random_variation = (1 if random.random() < self.mutation_rate else 0) * 2 * self.random_variation * chromosome_range * random.random() * (random.random() - 0.5)
                    child_gene_interleave[j] = (parent1[j] if random.random() > 0.5 else parent2[j]) + random_variation
                    random_variation_for_scale = (1 if random.random() < self.mutation_rate else 0) * 2 * self.random_variation * chromosome_range * random.random() * (random.random() - 0.5)
                    child_gene_scale[j] = (parent1[j] * splice_factor + parent2[j] * (1 - splice_factor)) + random_variation_for_scale
                    random_variation_for_splice = (1 if random.random() < self.mutation_rate else 0) * 2 * self.random_variation * chromosome_range * random.random() * (random.random() - 0.5)
                    child_gene_splice[j] = (parent1[j] if j < splice_point else parent2[j]) + random_variation_for_splice
                new_genes.append(child_gene_interleave)
                new_genes.append(child_gene_scale)
                new_genes.append(child_gene_splice)

            # Score the new population of genes
            new_population = []
            for new_gene in new_genes:
                score = self.score_gene(new_gene)
                new_population.append([new_gene, score])

            # Add the best gene from the previous generation if required
            if self.keep_best_gene:
                best_gene = self.population[0]
                score = self.score_gene(best_gene[0])
                emwa_score = best_gene[1] * 0.8 + 0.2 * score
                new_population.append([best_gene[0], emwa_score])

            # Filter and select best of population
            self.population = sorted(new_population, key=lambda x: x[1], reverse=self.maximize_score)[:self.population_size]
            print(f'Generation {generation}. Best score: {self.population[0][1]}')
            print(f'All scores: {np.around(np.array([g[1] for g in self.population]), 2).tolist()}')

    def get_population(self):
        return self.population
