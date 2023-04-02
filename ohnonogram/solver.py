"""
Defines solver interface and solver implementations.
"""
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Dict, List

from ohnonogram.nonogram import Nonogram, NonogramPuzzle


class NonogramSolver(ABC):
    """
    Class defining the interface for nonogram solvers.
    """

    puzzle: NonogramPuzzle
    """
    Puzzle to be solved.
    """
    attempted_states: List[Nonogram] = []
    """
    A list of puzzle states attempted during solving for later analysis.
    """

    def __init__(self, puzzle: NonogramPuzzle):
        self.puzzle = deepcopy(puzzle)

    @abstractmethod
    def solve(self) -> Nonogram:
        """
        Solve the puzzle and return a solved `Nonogram` object.
        """
        ...


class PermutationDepthFirstSolver(NonogramSolver):
    """
    Defines a permutation-based depth-first search nonogram solver.
    """

    def solve(self) -> Nonogram:
        """
        Solve the puzzle using a permutation-based depth-first search.

        Returns
        -------
        Nonogram
            An object representing the solved puzzle state.

        Notes
        -----
        First, we find all sequence permutations for all clues. In a smarter (later)
        implementation, maybe later change this so permutations are produced lazily and
        memoized using a generator. Next, use depth-first search to fill in puzzle from
        the collection of permutations. Fill in a permutation for the first row, then
        verify all column clues are still satisfiable. If not, back track, choose
        another permutation, and continue. Repeat until solved.
        """
        row_cursor = 0
        permutation_index_map: Dict[int, int] = {
            i: 0 for i in range(self.puzzle.row_count)
        }
        while not self.puzzle.evaluate_all_clues():
            current_row_permutation = self.puzzle.row_clue_permutations[row_cursor][
                permutation_index_map[row_cursor]
            ]
            self.puzzle.current_state[0, :] = current_row_permutation
            # evaluate all column clues
        return self.puzzle.current_state
