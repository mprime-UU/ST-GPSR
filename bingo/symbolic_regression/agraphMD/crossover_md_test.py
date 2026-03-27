"""Definition of crossover between two acyclic graph individuals

This module contains the implementation of single point crossover between
acyclic graph individuals.
"""
import numpy as np
import random 

from .validation_backend.validation_backend import validate_individual, get_stack_command_dimensions

from ...chromosomes.crossover import Crossover


class AGraphCrossoverMD(Crossover):
    """Crossover between acyclic graph individuals
    """

    def __call__(self, parent_1, parent_2):
        """Single point crossover.

        Parameters
        ----------
        parent_1 : `AGraph`
            The first parent individual
        parent_2 : `AGraph`
            The second parent individual

        Returns
        -------
        tuple(`AGraph`, `AGraph`) :
            The two children from the crossover.
        """
        attempts = 0
        while attempts == 0 or not validate_individual(child_1) or not validate_individual(child_2):
            if attempts >= 100:
                print("crossover failed")  # TODO turn into warning
                return parent_1.copy(), parent_2.copy()
            child_1, child_2 = self._single_point_crossover(parent_1, parent_2)
            child_1._update(), child_2._update()  # TODO testing for this
            attempts += 1
            # print(attempts)
        return child_1, child_2

    def _single_point_crossover(self, parent_1, parent_2):
        child_1 = parent_1.copy()
        child_2 = parent_2.copy()

        dim_list_1 = get_stack_command_dimensions(parent_1.command_array, parent_1.input_dims, parent_1.constant_shapes)
        dim_list_2 = get_stack_command_dimensions(parent_2.command_array, parent_2.input_dims, parent_2.constant_shapes)

        cross_matches = [i for i, (a, b) in enumerate(zip(dim_list_1, dim_list_2)) if a == b]
        
        # cross_matches = [(i, j) for i, x in enumerate(dim_list_1) for j, y in enumerate(dim_list_2) if x == y]

        if cross_matches:
            cross_point = random.choice(cross_matches)
        else:
            return parent_1, parent_2
        
        # child_1.command_array = np.zeros((len(parent_1.command_array) - cross_point[1] + cross_point[0], 4), dtype=int)
        # child_2.command_array = np.zeros((len(parent_2.command_array) - cross_point[0] + cross_point[1], 4), dtype=int)

        # child_1.mutable_command_array[:cross_point[0]] = \
        #     parent_1.command_array[:cross_point[0]]
        # child_2.mutable_command_array[:cross_point[1]] = \
        #     parent_2.command_array[:cross_point[1]]
        
        child_1.mutable_command_array[cross_point:] = \
            parent_2.command_array[cross_point:]
        child_2.mutable_command_array[cross_point:] = \
            parent_1.command_array[cross_point:]

        # for i in range(cross_point[0], len(child_1.command_array)):
        #     print(1, i)
        #     if child_1.mutable_command_array[i, 3] == 0 and child_1.mutable_command_array[i, 1] >= cross_point[0]:
        #         if cross_point[0] < cross_point[1]:
        #             child_1.mutable_command_array[i, 1] += cross_point[1] - cross_point[0]
        #         elif cross_point[0] > cross_point[1]:
        #             child_1.mutable_command_array[i, 1] += cross_point[0] - cross_point[1]
        #         print('change')

        #     if child_1.mutable_command_array[i, 3] == 0 and child_1.mutable_command_array[i, 2] >= cross_point[0]:
        #         if cross_point[0] < cross_point[1]:
        #             child_1.mutable_command_array[i, 2] += cross_point[1] - cross_point[0]
        #         elif cross_point[0] > cross_point[1]:
        #             child_1.mutable_command_array[i, 2] += cross_point[0] - cross_point[1]
        #         print('change')

        # for i in range(cross_point[1], len(child_2.command_array)):
        #     print(2,i, child_2.mutable_command_array[i, 1])
        #     if child_2.mutable_command_array[i, 3] == 0 and child_2.mutable_command_array[i, 1] >= cross_point[1]:
        #         if cross_point[0] < cross_point[1]:
        #             child_2.mutable_command_array[i, 1] += cross_point[0] - cross_point[1]
        #         elif cross_point[0] > cross_point[1]:
        #             child_2.mutable_command_array[i, 1] += cross_point[1] - cross_point[0]       
        #         print('change')

        #     if child_2.mutable_command_array[i, 3] == 0 and child_2.mutable_command_array[i, 2] >= cross_point[1]:
        #         if cross_point[0] < cross_point[1]:
        #             child_2.mutable_command_array[i, 2] += cross_point[0] - cross_point[1]
        #         elif cross_point[0] > cross_point[1]:
        #             child_2.mutable_command_array[i, 2] += cross_point[1] - cross_point[0] 
        #         print('change')
        # if cross_point[0] < cross_point[1]:
        #     print('if')
        #     child_1.mutable_command_array[cross_point[0]:][child_1.mutable_command_array[cross_point[0]:, 3] == 0][] += cross_point[0] - cross_point[1]
        #     child_2.mutable_command_array[cross_point[1]:][child_2.mutable_command_array[cross_point[1]:, 3] == 0, 1:3] += cross_point[1] - cross_point[0]
        # elif cross_point[0] > cross_point[1]:
        #     print('elif')
        #     child_1.mutable_command_array[cross_point[0]:][child_1.mutable_command_array[cross_point[0]:, 3] == 0, 1:3] += cross_point[1] - cross_point[0]
        #     child_2.mutable_command_array[cross_point[1]:][child_2.mutable_command_array[cross_point[1]:, 3] == 0, 1:3] += cross_point[0] - cross_point[1]

        child_age = max(parent_1.genetic_age, parent_2.genetic_age)
        child_1.genetic_age = child_age
        child_2.genetic_age = child_age
        # print(len(child_1.command_array),child_1.command_array, parent_1.command_array)
        # print(len(child_2.command_array),child_2.command_array, parent_2.command_array)
        return child_1, child_2
