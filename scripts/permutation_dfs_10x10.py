from pathlib import Path

from ohnonogram.nonogram import NonogramPuzzle
from ohnonogram.solver import PermutationDepthFirstSolver

if __name__ == "__main__":
    example_file = (
        Path(__file__).parents[1] / "example_puzzles"
    ) / "example_2_10x10.txt"
    puzzle = NonogramPuzzle.load_from_text_file(example_file)
    solver = PermutationDepthFirstSolver(puzzle)
    solution = solver.solve()
    print(str(solution))
