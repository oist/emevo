import operator
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from emevo.analysis import Tree

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


@pytest.fixture
def treedef_with_reward() -> list[tuple[int, int]]:
    #     0
    #    / \
    #   1   2
    #  /|\  |\
    # 3 4 5 6 7
    #     |\
    #     8 9
    return [(1, 0), (4, 1), (3, 1), (5, 1), (9, 5), (8, 5), (2, 0), (6, 2), (7, 2)]


def test_from_iter(treedef: list[tuple[int, int]]) -> None:
    tree = Tree.from_iter(treedef, root_idx=-1)
    preorder = list(map(operator.attrgetter("index"), tree.traverse(preorder=True)))
    assert preorder == [0, 1, 3, 4, 5, 8, 9, 2, 6, 7]
    postorder = list(map(operator.attrgetter("index"), tree.traverse(preorder=False)))
    assert postorder == [3, 4, 8, 9, 5, 1, 6, 7, 2, 0]
    assert tree.root.n_total_children == 10


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
