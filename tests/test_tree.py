import operator
import tempfile
from pathlib import Path
from typing import Callable

import pyarrow.parquet as pq
import pytest

from emevo.analysis import Tree
from emevo.analysis.tree import (
    load_split_nodes,
    number_split_nodes,
    save_split_nodes,
    traverse_split_nodes,
)

ASSET_DIR = Path(__file__).parent.joinpath("assets")


@pytest.fixture
def treedef() -> list[tuple[int, int]]:
    #     0
    #    / \
    #   1   2
    #  /|\  |\
    # 3 4 5 6 7
    #     |\
    #     8 9
    return [(1, 0), (4, 1), (3, 1), (5, 1), (9, 5), (8, 5), (2, 0), (6, 2), (7, 2)]


def rd(value: float) -> dict[str, float]:
    return {"reward": value}


@pytest.fixture
def treedef_with_rewards() -> list[tuple[int, int, dict[str, float]]]:
    #     0
    #    / \
    #   1   2
    #  /|\  |\
    # 3 4 5 6 7
    #     |\
    #     8 9
    #     |
    #     10
    #    /  \
    #   11  12
    return [
        (1, 0, rd(1.0)),
        (4, 1, rd(3.0)),
        (3, 1, rd(2.0)),
        (5, 1, rd(-1.0)),
        (9, 5, rd(-2.0)),
        (8, 5, rd(-3.0)),
        (2, 0, rd(4.0)),
        (6, 2, rd(4.0)),
        (7, 2, rd(6.0)),
        (10, 8, rd(4.0)),
        (11, 10, rd(10.0)),
        (12, 10, rd(1.0)),
    ]


def test_from_iter(treedef: list[tuple[int, int]]) -> None:
    tree = Tree.from_iter(treedef, root_idx=-1)
    preorder = list(map(operator.attrgetter("index"), tree.traverse(preorder=True)))
    assert preorder == [0, 1, 3, 4, 5, 8, 9, 2, 6, 7]
    postorder = list(map(operator.attrgetter("index"), tree.traverse(preorder=False)))
    assert postorder == [3, 4, 8, 9, 5, 1, 6, 7, 2, 0]
    assert tree.root.n_descendants == 10


def test_split(treedef: list[tuple[int, int]]) -> None:
    tree = Tree.from_iter(treedef)
    sp1 = tree.split(min_group_size=3)
    assert len(sp1) == 4
    parents = [sn for sn in sp1.values() if sn.parent is None]
    assert len(parents) == 1
    assert sp1[0] == parents[0]
    assert sp1[0].size == 1
    assert list(sp1[0].children) == [1, 2]
    assert sp1[1].size == 3
    assert sp1[2].size == 3
    assert sp1[5].size == 3

    sp2 = tree.split(min_group_size=4)
    assert len(sp2) == 2
    assert sp2[0].size == 4, sp2
    assert sp2[1].size == 6


def test_split_traverse(treedef: list[tuple[int, int]]) -> None:
    tree = Tree.from_iter(treedef)
    sp = tree.split(min_group_size=3)
    #     0             0
    #    / \           / \
    #   1   2         1   2
    #  /|\  |\         \
    # 3 4 5 6 7         5
    #     |\
    #     8 9

    def make_visit_fn() -> tuple[dict[int, int], Callable[[int], None]]:
        number = {}

        def visit(nodeid: int) -> None:
            length = len(number)
            number[nodeid] = length

        return number, visit

    d1, visit1 = make_visit_fn()
    traverse_split_nodes(sp, visit1, preorder=True)
    assert d1[0] == 0
    assert d1[1] == 1
    assert d1[5] == 2
    assert d1[2] == 3

    d2, visit2 = make_visit_fn()
    traverse_split_nodes(sp, visit2, preorder=False)
    assert d2[0] == 3
    assert d2[1] == 1
    assert d2[5] == 0
    assert d2[2] == 2


def test_number_split_nodes(treedef: list[tuple[int, int]]) -> None:
    tree = Tree.from_iter(treedef)
    sp = tree.split(min_group_size=3)
    d1 = number_split_nodes(sp, True)
    assert d1[0] == 0
    assert d1[1] == 1
    assert d1[5] == 2
    assert d1[2] == 3

    d2 = number_split_nodes(sp, False)
    assert d2[0] == 3
    assert d2[1] == 1
    assert d2[5] == 0
    assert d2[2] == 2


def test_split_saveload(treedef: list[tuple[int, int]]) -> None:
    tree = Tree.from_iter(treedef)
    sp1 = tree.split(min_group_size=3)
    path = Path(tempfile.NamedTemporaryFile(suffix=".json", delete=True).name)
    save_split_nodes(sp1, path)
    sp2 = load_split_nodes(path)
    assert len(sp1) == 4
    assert len(sp2) == 4
    for key, value in sp1.items():
        assert value == sp2[key]

    path.unlink()


def test_split_by_rewards(treedef_with_rewards: list[tuple[int, int, dict]]) -> None:
    #     0
    #    /
    #   1   2
    #  /|   |\
    # 3 4 5 6 7
    #     |\
    #     8 9
    #
    #     10
    #    /  \
    #   11  12
    tree = Tree.from_iter(treedef_with_rewards, root_info=rd(0.0))
    sp = tree.split(
        min_group_size=3,
        method="reward-mean",
        n_trial=3,
        reward_keys=["reward"],
    )
    assert len(sp) == 4
    assert sp[0].size == 4
    assert sp[2].size == 3
    assert sp[5].size == 3
    assert sp[10].size == 3
    assert list(sp[0].children) == [2, 5]
    assert len(sp[2].children) == 0
    assert list(sp[5].children) == [10]
    assert len(sp[10].children) == 0

    sp2 = tree.split(
        min_group_size=3,
        method="reward-sum",
        n_trial=3,
        reward_keys=["reward"],
    )
    assert len(sp2) == 4
    assert sp2[0].size == 4
    assert sp2[1].size == 3
    assert sp2[5].size == 3
    assert sp2[10].size == 3
    assert list(sp2[0].children) == [1]
    assert list(sp2[1].children) == [5]
    assert list(sp2[5].children) == [10]
    assert len(sp2[10].children) == 0


def test_from_table() -> None:
    table = pq.read_table(ASSET_DIR.joinpath("profile_and_rewards.parquet"))
    tree = Tree.from_table(table, 20)
    for root in tree.root.children:
        assert root.index <= 20
        assert root.birth_time is not None
        for node in root.traverse():
            assert node.birth_time is not None

    split = tree.split(min_group_size=10)
    data_dict = tree.as_datadict(split)
    for key in ["unique_id", "label"]:
        assert key in data_dict
