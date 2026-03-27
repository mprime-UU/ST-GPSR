import logging
import numpy as np
import os

from ..evaluation.fitness_function import VectorBasedFunction

LOGGER = logging.getLogger(__name__)

class CustomFitness(VectorBasedFunction):
    def __init__(self, training_data, stress, d):
        super().__init__(training_data)
        self.training_data = training_data
        self.stress = stress
        self.d = d
        
    def evaluate_fitness_vector(self, individual):
        self.eval_count += 1
        delta_S = individual.evaluate_equation_at(self.training_data.x)
        
        E1 = 240e3
        E2 = 240e3
        v12 = 0.17
        G12 = 90e3

        
        # folder = r'C:\Users\Will\Documents\bingo_david\bingo\Data'
        # os.chdir(folder)

        # strain_real = np.loadtxt('0_90s-20C-1-2098-01-0001-TD3.csv',usecols=(0),skiprows=2,delimiter=',')[::10]
        # stress_real = np.loadtxt('0_90s-20C-1-2098-01-0001-TD3.csv',usecols=(1),skiprows=2,delimiter=',')[::10]

        # strain = np.zeros((len(strain_real),3,1))
        # stress = np.zeros((len(stress_real),3,1))

        # for i in range(len(strain_real)):
        #     strain[i,0,0] = strain_real[i]
        #     strain[i,1,0] = strain_real[i]*-0.1
            
        #     stress[i,0,0] = stress_real[i]
            
        S = np.array([[1/E1, -v12/E1, 0],
                      [-v12/E2, 1/E2, 0],
                      [0, 0, 1/G12]])
        
        # C = np.linalg.inv(S)
        # Ceff = np.linalg.inv(S + self.d*delta_S)
        
        delta_S[:,0,1] = 0
        delta_S[:,1,0] = 0

        strain = (S + delta_S) @ self.stress
        
        # stress = Ceff @ self.training_data.x[0]
        # print(stress)
        
        custom_fitness_vector = abs(self.training_data.x[0] - strain)
        
        # error = strain - self.training_data.y
        # rel_err = 2.0 * error / (np.abs(strain) + np.abs(self.training_data.y))  # RPD
        # both_zero = (strain == 0) & (self.training_data.y == 0)
        # rel_err[both_zero] = 0

        return custom_fitness_vector.flatten() 
        