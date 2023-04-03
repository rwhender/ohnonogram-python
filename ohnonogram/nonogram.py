"""
Defines fundamental classes for nonogram representation.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt

STATES = {"unknown": "0", "filled": "#", "unfilled": "x"}
SPLIT_PATTERN = re.compile(rf"[{STATES['unfilled']}{STATES['unknown']}]+")


@dataclass
class Nonogram:
    """
    Dataclass representing nonogram puzzle clues.

    0 = Unknown
    # = Filled
    x = Unfilled
    """

    grid: npt.NDArray

    def __getitem__(self, key) -> "Nonogram":
        return type(self)(self.grid[key])

    def __setitem__(self, key, value):
        if isinstance(value, Nonogram):
            self.grid[key] = value.grid
        elif isinstance(value, np.ndarray):
            self.grid[key] = value
        else:
            raise TypeError("Unsupported type.")

    def __str__(self) -> str:
        return "\n".join(
            [
                "".join(list(self.grid[row_index]))
                for row_index in range(np.shape(self.grid)[0])
            ]
        )

    @property
    def row_count(self) -> Optional[int]:
        return self.grid.shape[0] if len(self.grid.shape) >= 1 else None

    @property
    def column_count(self) -> Optional[int]:
        return self.grid.shape[1] if len(self.grid.shape) >= 2 else None

    @classmethod
    def load_grid_from_text(cls, grid_text: List[List[str]]) -> "Nonogram":
        grid: List[List[str]] = []
        for row in grid_text:
            column_list: List[str] = []
            for column in row:
                assert len(column) == 1, f"Actual column value: {column}"
                column_list.append(column)
            grid.append(column_list)
        return cls(np.array(grid))

    @classmethod
    def from_sequence(cls, sequence: Sequence) -> "Nonogram":
        return cls(np.array(sequence))

    def intersection(self, other: "Nonogram") -> bool:
        return bool(
            np.all(
                ((self.grid == STATES["filled"]) & (other.grid == STATES["filled"]))
                | (
                    (self.grid == STATES["unfilled"])
                    & (other.grid == STATES["unfilled"])
                )
                | (self.grid == STATES["unknown"])
                | (other.grid == STATES["unknown"])
            )
        )


@dataclass
class NonogramPuzzle:
    """
    Dataclass collecting nonogram clues.
    """

    row_count: int
    column_count: int
    row_clues: List[List[int]]
    column_clues: List[List[int]]
    current_state: Nonogram
    row_clue_permutations: List[List[Nonogram]] = field(default_factory=list)
    column_clue_permutations: List[List[Nonogram]] = field(default_factory=list)

    def __post_init__(self):
        self.row_clue_permutations = [
            self.get_all_permutations("row", i) for i in range(len(self.row_clues))
        ]
        self.column_clue_permutations = [
            self.get_all_permutations("column", i)
            for i in range(len(self.column_clues))
        ]

    @classmethod
    def load_from_text_file(cls, text_file: Union[str, Path]) -> "NonogramPuzzle":
        with open(str(text_file), "rt") as f:
            row_count = 0
            column_count = 0
            row_clues: List[List[int]] = []
            column_clues: List[List[int]] = []
            grid_text: List[List[str]] = []
            for line_index, line in enumerate(f):
                if line_index == 0:
                    row_count, column_count = [int(item) for item in line.split()]
                elif line_index >= 1 and line_index < row_count + 1:
                    row_clues.append([int(value) for value in line.split()])
                elif (
                    line_index >= row_count + 1
                    and line_index < row_count + column_count + 1
                ):
                    column_clues.append([int(value) for value in line.split()])
                else:
                    # Read the remaining lines as initial puzzle state
                    grid_text.append(list(line.strip()))
            return cls(
                row_count,
                column_count,
                row_clues,
                column_clues,
                Nonogram.load_grid_from_text(grid_text),
            )

    def evaluate_clue(self, row_or_col: str, index: int) -> bool:
        sequence, clue = self.get_sequence_and_clue(row_or_col, index)
        sequence_split = SPLIT_PATTERN.split("".join(list(sequence)))
        if (
            len(clue) == 0
            and len(sequence_split) == 1
            and all(
                item == STATES["unknown"] or item == STATES["unfilled"]
                for item in sequence_split
            )
        ):
            return True
        if not sequence_split[0]:
            sequence_split.pop(0)
        if not sequence_split[-1]:
            sequence_split.pop(-1)
        elif len(clue) != len(sequence_split):
            return False
        for clue_item, sequence_item in zip(clue, sequence_split):
            if len(sequence_item) != clue_item:
                return False
        return True

    def evaluate_all_clues(self) -> bool:
        # Row clues
        for row_index in range(self.row_count):
            if not self.evaluate_clue("row", row_index):
                return False
        # Column clues
        for column_index in range(self.column_count):
            if not self.evaluate_clue("column", column_index):
                return False
        return True

    def get_sequence_and_clue(
        self,
        row_or_col: str,
        index: int,
    ) -> Tuple[np.ndarray, List[int]]:
        if row_or_col.lower() == "row":
            sequence = self.current_state.grid[index]
            clue = self.row_clues[index]
            assert isinstance(sequence, np.ndarray)
            assert sequence.shape == (self.column_count,)
        elif row_or_col.lower() == "column":
            sequence = self.current_state.grid[:, index]
            clue = self.column_clues[index]
            assert isinstance(sequence, np.ndarray)
            assert sequence.shape == (self.row_count,)
        else:
            raise ValueError("row_or_col argument must be either 'row' or 'column'.")
        return sequence, clue

    def get_all_permutations(self, row_or_col: str, index: int) -> List[Nonogram]:
        """
        Generate all possible permutations for the clue given by the arguments.

        Parameters
        ----------
        row_or_col : str
            "row" or "column". If "row", `index` refers to a row index. If "column",
            `index` refers to a column index.
        index : int
            Row or column index

        Returns
        -------
        List[Nonogram]
            The list of all sequence permutations for the given clue.
        """
        sequence, clue = self.get_sequence_and_clue(row_or_col, index)
        return [Nonogram(array) for array in get_permutations(clue, len(sequence))]

    def clue_is_satisfiable(self, row_or_col: str, index: int) -> bool:
        sequence, _ = self.get_sequence_and_clue(row_or_col, index)
        permutations = (
            self.row_clue_permutations[index]
            if row_or_col == "row"
            else self.column_clue_permutations[index]
        )
        for permutation in permutations:
            if permutation.intersection(Nonogram(sequence)):
                return True
        return False


def get_permutations(clue: List[int], line_length: int) -> List[np.ndarray]:
    if not clue:
        return [np.array([STATES["unfilled"] for _ in range(line_length)])]
    permutations: List[np.ndarray] = []
    for start_index in range(line_length - sum(clue) - len(clue) + 2):
        permutation = np.zeros((line_length,), dtype=">U1")
        permutation[0:start_index] = STATES["unfilled"]
        permutation[start_index : start_index + clue[0]] = STATES["filled"]
        cursor = start_index + clue[0]
        if cursor < line_length:
            permutation[cursor] = STATES["unfilled"]
            cursor += 1
        sub_permutations = get_permutations(clue[1:], line_length - cursor)
        for sub_permutation in sub_permutations:
            new_permutation = np.copy(permutation)
            new_permutation[cursor:] = sub_permutation
            permutations.append(new_permutation)
    return permutations
