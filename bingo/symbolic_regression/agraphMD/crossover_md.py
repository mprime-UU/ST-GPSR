"""Definition of crossover between two acyclic graph individuals

This module contains the implementation of single point crossover between
acyclic graph individuals.
"""
import numpy as np

from .validation_backend.validation_backend import validate_individual
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

        ag_size = parent_1.command_array.shape[0]
        cross_point = np.random.randint(1, ag_size-1)
        child_1.mutable_command_array[cross_point:] = \
            parent_2.command_array[cross_point:]
        child_2.mutable_command_array[cross_point:] = \
            parent_1.command_array[cross_point:]

        child_age = max(parent_1.genetic_age, parent_2.genetic_age)
        child_1.genetic_age = child_age
        child_2.genetic_age = child_age

        return child_1, child_2
