"""
Defines solver interface and solver implementations.
"""
from abc import ABC, abstractmethod
from typing import List

from ohnonogram.nonogram import Nonogram, NonogramPuzzle


class NonogramSolver(ABC):
    """
    Class defining the interface for nonogram solvers.
    """

    puzzle: NonogramPuzzle
    """
    Puzzle to be solved.
    """
    attempted_states: List[Nonogram]
    """
    A list of puzzle states attempted during solving for later analysis.
    """

    def __init__(self, puzzle: NonogramPuzzle):
        self.puzzle = puzzle

    @abstractmethod
    def solve(self) -> Nonogram:
        ...
