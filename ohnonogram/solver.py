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


class SimpleDepthFirstSearchSolver(NonogramSolver):
    """
    Defines a simple depth-first search nonogram solver.
    """

    def solve(self) -> Nonogram:
        """
        Solve the puzzle using a brute-force depth-first search strategy.

        Returns
        -------
        Nonogram
            An object representing the solved puzzle state.

        Raises
        ------
        RuntimeError
            If the solution loop ends an no solution has been found.
        """
        row_cursor = 0
        permutation_index_list: List[int] = [0 for i in range(self.puzzle.row_count)]
        permutation_list_lengths: List[int] = [
            len(self.puzzle.row_clue_permutations[i])
            for i in range(self.puzzle.row_count)
        ]
        while row_cursor < self.puzzle.row_count:
            print("Row cursor:", row_cursor)
            print("Permutation index:", permutation_index_list[row_cursor])
            self.attempted_states.append(self.puzzle.current_state)
            assert row_cursor >= 0, "Negative row cursor found. Undefined state."

            self.puzzle.current_state[
                row_cursor, :
            ] = self.puzzle.row_clue_permutations[row_cursor][
                permutation_index_list[row_cursor]
            ]
            if row_cursor >= self.puzzle.row_count - 1:
                if not self.puzzle.evaluate_all_clues():
                    keep_backtracking = True
                    while keep_backtracking:
                        if (
                            permutation_index_list[row_cursor]
                            >= permutation_list_lengths[row_cursor] - 1
                        ):
                            permutation_index_list[row_cursor] = 0
                            row_cursor -= 1
                        else:
                            permutation_index_list[row_cursor] += 1
                            keep_backtracking = False
                else:
                    print("solved!")
            else:
                row_cursor += 1
        if not self.puzzle.evaluate_all_clues():
            raise RuntimeError("Solution loop ended, but puzzle is not solved.")
        return self.puzzle.current_state


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
        It turns out this solution method is extremely slow for large puzzles, because
        checking the satisfiability of a column requires a number of operations
        proportional to the number of permutations that exist for that column. It's
        probably much faster just to fill the whole puzzle in, then check its
        correctness, in order to check the correctness of a branch of the DFS tree.

        Original note:

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
        permutation_list_length: Dict[int, int] = {
            i: len(self.puzzle.row_clue_permutations[i])
            for i in range(self.puzzle.row_count)
        }
        while row_cursor < self.puzzle.row_count:
            print("Row cursor:", row_cursor)
            print("Permutation index:", permutation_index_map[row_cursor])
            self.attempted_states.append(self.puzzle.current_state)
            assert row_cursor >= 0, "Negative row cursor found. Undefined state."
            current_row_permutation = self.puzzle.row_clue_permutations[row_cursor][
                permutation_index_map[row_cursor]
            ]
            self.puzzle.current_state[row_cursor, :] = current_row_permutation
            all_columns_good = True
            for column_index in range(self.puzzle.column_count):
                print("column index:", column_index)
                if not self.puzzle.clue_is_satisfiable("column", column_index):
                    all_columns_good = False
                    break
            if not all_columns_good:
                keep_backtracking = True
                while keep_backtracking:
                    if (
                        permutation_index_map[row_cursor]
                        >= permutation_list_length[row_cursor] - 1
                    ):
                        row_cursor -= 1
                    else:
                        permutation_index_map[row_cursor] += 1
                        keep_backtracking = False
            else:
                row_cursor += 1
        if not self.puzzle.evaluate_all_clues():
            raise RuntimeError("Solution loop ended, but puzzle is not solved.")
        return self.puzzle.current_state
