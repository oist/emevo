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


def test_from_iter(treedef: list[tuple[int, int]]) -> None:
    tree = Tree.from_iter(treedef)
    preorder = list(map(operator.attrgetter("index"), tree.traverse(preorder=True)))
    assert preorder == [0, 1, 3, 4, 5, 8, 9, 2, 6, 7]
    postorder = list(map(operator.attrgetter("index"), tree.traverse(preorder=False)))
    assert postorder == [3, 4, 8, 9, 5, 1, 6, 7, 2, 0]
    assert tree.root.n_total_children == 10


def test_split(treedef: list[tuple[int, int]]) -> None:
    tree = Tree.from_iter(treedef)
    sp1 = tree.split(min_group_size=3)
    assert len(sp1) == 10
    assert sp1[0] == 0
    for idx in [1, 3, 4]:
        assert sp1[idx] == 1
    for idx in [2, 6, 7]:
        assert sp1[idx] == 2
    for idx in [5, 8, 9]:
        assert sp1[idx] == 3

    sp2 = tree.split(min_group_size=4)
    assert len(sp2) == 10
    for idx in [0, 2, 6, 7]:
        assert sp2[idx] == 0
    for idx in [1, 3, 4, 5, 8, 9]:
        assert sp2[idx] == 1


def test_multilabel_split(treedef: list[tuple[int, int]]) -> None:
    tree = Tree.from_iter(treedef)
    lb1 = tree.multilabel_split(min_group_size=3)
    assert len(lb1) == 4
    assert list(sorted(lb1[0])) == [0]
    assert list(sorted(lb1[1])) == [0, 1, 3, 4]
    assert list(sorted(lb1[2])) == [0, 2, 6, 7]
    assert list(sorted(lb1[3])) == [0, 1, 5, 8, 9]


def test_from_table() -> None:
    table = pq.read_table(ASSET_DIR.joinpath("profile_and_rewards.parquet"))
    tree = Tree.from_table(table)
    for root, _ in tree.root.children:
        assert root.index < 10
        assert root.birth_time is not None
        for node in root.traverse():
            assert node.birth_time is not None

    data_dict = tree.as_datadict(split=10)
    for key in ["index", "birth-step", "label", "in-label-0", "in-label-1"]:
        assert key in data_dict
