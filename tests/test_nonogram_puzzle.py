"""
Tests for NonogramPuzzle.
"""
from pathlib import Path
from typing import List

import pytest
from pytest_unordered import unordered

from ohnonogram.nonogram import Nonogram, NonogramPuzzle, get_permutations

expected_str = """0000000000000000000000000
0000000000000000000000000
0000000000000000000000000
000##0000000##0000000#000
0000000000000000000000000
0000000000000000000000000
0000000000000000000000000
0000000000000000000000000
000000##00#000##00#000000
0000000000000000000000000
0000000000000000000000000
0000000000000000000000000
0000000000000000000000000
0000000000000000000000000
0000000000000000000000000
0000000000000000000000000
000000#0000#0000#000#0000
0000000000000000000000000
0000000000000000000000000
0000000000000000000000000
0000000000000000000000000
000##0000##0000#0000##000
0000000000000000000000000
0000000000000000000000000
0000000000000000000000000"""

expected_solution = """#######0###000#0#0#######
#00000#0##0##00000#00000#
#0###0#00000###0#0#0###0#
#0###0#0#00######0#0###0#
#0###0#00#####0##0#0###0#
#00000#00##0000000#00000#
#######0#0#0#0#0#0#######
00000000###000###00000000
#0##0###00#0#0###0#00#0##
#0#000000###0##0000#000#0
0####0#0####0##0#0000##00
0#0#000#000#0#0####0#0###
00##00#0#0#000000##0#####
000###0##0##0######0###0#
#0#########0#0#00##0000#0
0##0#00##000##0###00000#0
###0#0#0#00#0000#####0#00
00000000#000##0##000#####
#######0#00##000#0#0#0###
#00000#0##00#00##000##0#0
#0###0#000####00#####00#0
#0###0#0###0##########0##
#0###0#0#00######0######0
#00000#00##000000#0#0##00
#######0##000#0##000#####"""


def test_nonogram_puzzle_load():
    example_file = (
        Path(__file__).parents[1] / "example_puzzles"
    ) / "example_1_25x25 with initial state.txt"
    puzzle = NonogramPuzzle.load_from_text_file(example_file)
    assert str(puzzle.current_state) == expected_str


def test_nonogram_puzzle_eval():
    example_file = (
        Path(__file__).parents[1] / "example_puzzles"
    ) / "example_1_25x25 with initial state.txt"
    puzzle = NonogramPuzzle.load_from_text_file(example_file)
    puzzle.current_state = Nonogram.load_grid_from_text(
        [list(row) for row in expected_solution.split("\n")]
    )
    assert puzzle.evaluate_all_clues()


@pytest.mark.parametrize(
    "clue,line_length,expected",
    [
        ([], 1, [["x"]]),
        ([1], 1, [["#"]]),
        ([2], 3, [["#", "#", "x"], ["x", "#", "#"]]),
        (
            [2],
            4,
            [
                ["#", "#", "x", "x"],
                ["x", "#", "#", "x"],
                ["x", "x", "#", "#"],
            ],
        ),
        (
            [1, 1],
            4,
            [
                ["#", "x", "#", "x"],
                ["#", "x", "x", "#"],
                ["x", "#", "x", "#"],
            ],
        ),
        (
            [1, 2],
            5,
            [
                ["#", "x", "#", "#", "x"],
                ["#", "x", "x", "#", "#"],
                ["x", "#", "x", "#", "#"],
            ],
        ),
        (
            [1, 1, 2],
            7,
            [
                ["#", "x", "#", "x", "#", "#", "x"],
                ["#", "x", "#", "x", "x", "#", "#"],
                ["#", "x", "x", "#", "x", "#", "#"],
                ["x", "#", "x", "#", "x", "#", "#"],
            ],
        ),
    ],
)
def test_get_permutations(clue: List[int], line_length: int, expected: List[List[str]]):
    result = get_permutations(clue, line_length)
    result_list = [item.tolist() for item in result]
    assert result_list == unordered(expected)
