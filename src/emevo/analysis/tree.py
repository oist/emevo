"""Tree data implementation used for analyzing agent's phylogeny."""

from __future__ import annotations

import dataclasses
import functools
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal
from weakref import ReferenceType
from weakref import ref as make_weakref

import networkx as nx
import numpy as np
import serde
from networkx.drawing.nx_agraph import graphviz_layout
from numpy.typing import NDArray
from pyarrow import Table
from serde.json import from_json, to_json

datafield = functools.partial(dataclasses.field, compare=False, hash=False, repr=False)


@functools.total_ordering
@dataclasses.dataclass
class Node:
    index: int
    is_root: dataclasses.InitVar[bool] = dataclasses.field(default=False)
    birth_time: int | None = None
    parent_ref: ReferenceType[Node] | None = datafield(default=None)
    children: list[Node] = datafield(default_factory=list)
    info: dict[str, Any] = datafield(default_factory=dict)

    def __post_init__(self, is_root: bool) -> None:
        if not is_root and self.index < 0:
            raise ValueError(f"Negative index {self.index} is not allowed as an index")

    def __hash__(self) -> int:
        return self.index

    def __eq__(self, other: Node) -> bool:
        return self.index == other.index

    def add_child(self, child: Node, **kwargs) -> None:
        if child.parent_ref is not None:
            raise RuntimeError(f"Child {child.index} already has a parent")
        child.info = kwargs
        self.children.append(child)
        child.parent_ref = make_weakref(self)

    def sort_children(self) -> None:
        self.children.sort(key=lambda child: child.index)

    @property
    def n_children(self) -> int:
        return len(self.children)

    @property
    def parent(self) -> Node | None:
        if self.parent_ref is None:
            return None
        else:
            return self.parent_ref()

    @functools.cached_property
    def n_descendants(self) -> int:
        total = 0
        for child in self.children:
            total += 1 + child.n_descendants
        return total

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, Node):
            return self.index < other.index
        else:
            return True

    def ancestors(self, include_self: bool = True) -> Iterable[Node]:
        if include_self:
            yield self
        parent = self.parent
        if parent is not None:
            yield from parent.ancestors()

    def traverse(
        self,
        preorder: bool = True,
        include_self: bool = True,
    ) -> Iterable[Node]:
        if include_self and preorder:
            yield self
        for child in self.children:
            yield from child.traverse(preorder)
        if include_self and not preorder:
            yield self


@dataclasses.dataclass
class Edge:
    parent: Node
    child: Node

    def __hash__(self) -> int:
        return self.parent.index * (2**30) + self.child.index

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Edge):
            return self.parent == other.parent and self.parent == other.parent
        else:
            return False

    def __lt__(self, other: Edge) -> bool:
        if self.parent.index == other.parent.index:
            return self.child.index < other.child.index
        else:
            return self.parent.index < other.parent.index


def _collect_descendants(node: Node, include_self: bool = False) -> set[int]:
    ret = set()
    if include_self:
        ret.add(node.index)
    for child in node.children:
        descendants = _collect_descendants(child, include_self=True)
        ret.update(descendants)
    return ret


@serde.serde
@dataclasses.dataclass
class SplitNode:
    size: int
    reward_mean: dict[str, float] | None = None
    children: set[int] = dataclasses.field(default_factory=set)
    parent: int | None = None


def save_split_nodes(split_nodes: dict[int, SplitNode], path: Path) -> None:
    json_nodes = to_json(split_nodes)
    print(json_nodes)
    with path.open(mode="w") as f:
        json.dump(json_nodes, f)


def load_split_nodes(path: Path) -> dict[int, SplitNode]:
    with path.open(mode="r") as f:
        json_nodes = json.load(f)
    return from_json(dict[int, SplitNode], json_nodes)


@functools.cache
def compute_reward_mean(
    node: Node,
    is_root: bool = False,
    skipped_edges: frozenset[tuple[int, int]] | None = None,
    reward_keys: tuple[str, ...] = (),
) -> tuple[int, dict[str, float]]:
    if reward_keys[0] not in node.info:
        if is_root:
            size_list = [0]
            reward_mean_lists = {key: [0.0] for key in reward_keys}
        else:
            return 0, {key: 0.0 for key in reward_keys}
    else:
        size_list = [1]
        reward_mean_lists = {key: [node.info[key]] for key in reward_keys}

    for child in node.children:
        if skipped_edges is not None and (node.index, child.index) in skipped_edges:
            continue
        n_children, reward_mean = compute_reward_mean(
            child,
            skipped_edges=skipped_edges,
            reward_keys=reward_keys,
        )
        size_list.append(n_children)
        for key, rmean in reward_mean.items():
            reward_mean_lists[key].append(rmean)

    total_size = np.sum(size_list)
    rmean_dict = {}
    for key, rmean in reward_mean_lists.items():
        rsum = np.sum([nc * rm for nc, rm in zip(size_list, rmean)])
        rmean_dict[key] = rsum / total_size
    return total_size, rmean_dict


@dataclasses.dataclass
class Tree:
    root: Node
    nodes: dict[int, Node]

    @staticmethod
    def from_iter(
        iterator: Iterable[tuple[int, int] | tuple[int, int, dict]],
        root_idx: int = 0,
        root_info: dict | None = None,
    ) -> Tree:
        root = Node(index=root_idx, is_root=True)
        if root_info is not None:
            root.info = root_info
        nodes = {}

        for item in iterator:
            if len(item) == 2:
                idx, parent_idx = item
                kwargs = {}
            else:
                idx, parent_idx, kwargs = item

            if parent_idx in nodes:
                parent = nodes[parent_idx]
            elif parent_idx == root_idx:
                parent = root
            else:
                parent = Node(parent_idx)
                nodes[parent_idx] = parent
            if idx not in nodes:
                nodes[idx] = Node(index=idx)
            parent.add_child(nodes[idx], **kwargs)

        for node in nodes.values():
            if node.parent_ref is None:
                root.add_child(node)

            node.sort_children()
        nodes[root_idx] = root
        return Tree(root, nodes)

    @staticmethod
    def from_table(
        table: Table,
        initial_population: int | None = None,
        root_idx: int = 0,
        root_info: dict | None = None,
    ) -> Tree:
        birth_steps = {}

        def table_iter() -> Iterable[tuple[int, int, dict]]:
            for batch in table.to_batches():
                for row in batch.to_pylist():
                    idx = row.pop("unique_id")
                    birth_steps[idx] = row.pop("birthtime")
                    yield idx, row.pop("parent"), row

        tree = Tree.from_iter(table_iter(), root_idx=root_idx, root_info=root_info)
        for idx, node in tree.nodes.items():
            if idx in birth_steps:
                node.birth_time = birth_steps[idx]
            else:
                node.birth_time = 0

        if initial_population is not None:
            for i in range(1, initial_population + 1):
                if i not in tree.nodes:
                    node = Node(index=i)
                    tree.nodes[i] = node
                    tree.root.add_child(node)
                    node.birth_time = 0
        return tree

    def add_root(self, node: Node) -> None:
        self.root.add_child(node)

    def all_edges(self) -> Iterable[Edge]:
        for node in self.nodes.values():
            for child in node.children:
                yield Edge(node, child)

    def as_networkx(self) -> nx.Graph:
        tree = nx.Graph()
        for node in self.nodes.values():
            tree.add_node(node.index)
            for child in node.children:
                tree.add_edge(node.index, child.index)
        return tree

    def length(self) -> int:
        return len(self.nodes)

    def traverse(self, preorder: bool = True) -> Iterable[Node]:
        return self.root.traverse(preorder=preorder, include_self=False)

    def split(
        self,
        min_group_size: int = 1000,
        method: Literal["greedy", "reward-mean", "reward-sum"] = "greedy",
        n_trial: int = 5,
        reward_keys: list[str] | None = None,
    ) -> dict[int, SplitNode]:
        if method == "greedy":
            split_nodes = self._split_greedy(min_group_size, reward_keys)
        elif method in ["reward-mean", "reward-sum"]:
            assert reward_keys is not None
            is_mean = "mean" in method
            split_nodes = self._split_reward_mean(
                min_group_size,
                n_trial,
                reward_keys,
                compare_mean=is_mean,
            )
        else:
            raise ValueError(f"Unsupported split method: {method}")

        return split_nodes

    def colorize(self, split_nodes: dict[int, SplitNode]) -> dict[int, int]:
        categ = {node.index: 0 for node in self.nodes.values()}

        def colorize_impl(node: Node, color: int) -> None:
            categ[node.index] = color
            for child in node.children:
                if child.index not in split_nodes:
                    colorize_impl(child, color)

        for i, node_idx in enumerate(split_nodes):
            colorize_impl(self.nodes[node_idx], i)
        return categ

    def _split_greedy(
        self,
        min_group_size: int,
        reward_keys: list[str] | None,
    ) -> dict[int, SplitNode]:
        split_nodes = {}
        split_edges = set()

        def split(node: Node, threshold: int) -> int:
            size = 1
            for child in node.children:
                # Number of children that are not splitted
                n_existing_children = split(child, threshold)
                size += n_existing_children

            if size >= threshold:
                parent = node.parent
                split_nodes[node.index] = SplitNode(size)
                if parent is not None:
                    split_edges.add((parent.index, node.index))
                return 0
            else:
                return size

        size = split(self.root, min_group_size)
        if size > 0:
            split_nodes[self.root.index] = SplitNode(size)

        for node_index in split_nodes:
            if node_index == self.root.index:
                continue
            # Find Parent
            ancestor = self.nodes[node_index].parent
            while ancestor is not None:
                if ancestor.index in split_nodes:
                    split_nodes[ancestor.index].children.add(node_index)
                    split_nodes[node_index].parent = ancestor.index
                    break
                ancestor = ancestor.parent

        if reward_keys is not None:
            reward_keys_t = tuple(reward_keys)
            frozen_split_edges = frozenset(split_edges)
            for node_index, split_node in split_nodes.items():
                node = (
                    self.root
                    if node_index == self.root.index
                    else self.nodes[node_index]
                )
                size, reward = compute_reward_mean(
                    node,
                    skipped_edges=frozen_split_edges,
                    reward_keys=reward_keys_t,
                )
                split_node.size = size
                split_node.reward_mean = reward

        return split_nodes

    def _split_reward_mean(
        self,
        min_group_size: int,
        n_trial: int,
        reward_keys: list[str],
        compare_mean: bool = True,
    ) -> dict[int, SplitNode]:
        split_nodes = {}
        split_edges = set()
        reward_keys_t = tuple(reward_keys)

        def find_group_root(node: Node) -> int:
            ancestor_idx = node.index
            ancestor = node.parent
            while (
                ancestor is not None
                and (ancestor.index, ancestor_idx) not in split_edges
            ):
                ancestor_idx = ancestor.index
                ancestor = ancestor.parent
            return ancestor_idx

        def find_maxdiff_edge(
            frozen_split_edges: frozenset[tuple[int, int]],
        ) -> tuple[float, Edge]:
            max_effect = 0.0
            max_effect_edge = None
            failure_causes = {
                "Edge already used": 0,
                "Group size is too small": 0,
                "Effect is too small": 0,
            }
            for edge in self.all_edges():
                if (edge.parent.index, edge.child.index) in split_edges:
                    failure_causes["Edge already used"] += 1
                    continue
                parent_root = self.nodes[find_group_root(edge.parent)]
                parent_size, parent_reward = compute_reward_mean(
                    parent_root,
                    is_root=parent_root.index == self.root.index,
                    skipped_edges=frozen_split_edges,
                    reward_keys=reward_keys_t,
                )
                child_size, child_reward = compute_reward_mean(
                    edge.child,
                    skipped_edges=frozen_split_edges,
                    reward_keys=reward_keys_t,
                )
                if (
                    child_size < min_group_size
                    or (parent_size - child_size) < min_group_size
                ):
                    failure_causes["Group size is too small"] += 1
                    continue
                assert parent_size > child_size, (parent_size, child_size)
                split_size = parent_size - child_size
                total_diff = 0.0
                for key in reward_keys:
                    parent_rew_total = parent_reward[key] * parent_size
                    child_rew_total = child_reward[key] * child_size
                    if compare_mean:
                        split_rew = (parent_rew_total - child_rew_total) / split_size
                        total_diff += (child_reward[key] - split_rew) ** 2
                    else:
                        total_diff += (parent_rew_total - child_rew_total) ** 2
                effect = total_diff**0.5
                if effect > max_effect:
                    max_effect = effect
                    max_effect_edge = edge
                else:
                    failure_causes["Effect is too small"] += 1
            assert max_effect_edge is not None, (
                f"Couldn't find maxdiff_edge anymore (Reason: {failure_causes})"
            )
            return max_effect, max_effect_edge

        for _ in range(n_trial):
            frozen_split_edges = frozenset(split_edges)
            maxe, edge = find_maxdiff_edge(frozen_split_edges)
            parent_root = self.nodes[find_group_root(edge.parent)]
            parent_size, parent_reward = compute_reward_mean(
                parent_root,
                is_root=parent_root.index == self.root.index,
                skipped_edges=frozen_split_edges,
                reward_keys=reward_keys_t,
            )
            child_size, child_reward = compute_reward_mean(
                edge.child,
                skipped_edges=frozen_split_edges,
                reward_keys=reward_keys_t,
            )
            split_size = parent_size - child_size
            assert split_size > 0, (parent_size, child_size, edge)
            split_rew = {}
            for key in parent_reward:
                parent_rew_total = parent_reward[key] * parent_size
                child_rew_total = child_reward[key] * child_size
                split_rew[key] = (parent_rew_total - child_rew_total) / split_size
            # Make nodes
            if parent_root.index in split_nodes:
                # Add child
                split_nodes[parent_root.index].size = split_size
                split_nodes[parent_root.index].reward_mean = split_rew
                split_nodes[parent_root.index].children.add(edge.child.index)
            else:
                split_nodes[parent_root.index] = SplitNode(
                    split_size,
                    split_rew,
                    children=set([edge.child.index]),
                )
            split_nodes[edge.child.index] = SplitNode(
                child_size,
                child_reward,
                parent=parent_root.index,
            )
            # Find parent's children that should be moved to child
            descendants = _collect_descendants(edge.child, include_self=False)
            moved = descendants.intersection(split_nodes[parent_root.index].children)
            for child in moved:
                split_nodes[parent_root.index].children.remove(child)
                split_nodes[edge.child.index].children.add(child)
            split_edges.add((edge.parent.index, edge.child.index))

        return split_nodes

    def as_datadict(
        self,
        split_nodes: dict[int, SplitNode] | None = None,
    ) -> dict[str, NDArray]:
        """Returns a dict immediately convertable to Pandas dataframe"""

        indices = list(self.nodes.keys())
        data_dict = {"unique_id": np.array(indices, dtype=int)}
        representive_node = next(iter(self.nodes.values()))
        for key in representive_node.info.keys():
            collected = []
            for node in self.nodes.values():
                if key in node.info:
                    collected.append(node.info[key])
            if len(collected) == len(self.nodes):
                data_dict[key] = np.array(collected, dtype=type(collected[0]))

        if split_nodes is not None:
            labels = self.colorize(split_nodes)
            data_dict["label"] = np.array([labels[idx] for idx in indices], dtype=int)

        return data_dict

    def plot(
        self,
        split: int | dict[int, int] | None = None,
        palette: Sequence[tuple[float, float, float]] | None = None,
        **kwargs,
    ) -> None:
        nx_graph = self.as_networkx()
        default_kwargs = dict(
            with_labels=False,
            arrows=False,
            node_size=5,
            node_shape="o",
            width=0.5,
            node_color=range(len(self.nodes)),
            cmap="plasma",
            pos=graphviz_layout(nx_graph, prog="dot"),
        )
        draw_kwargs = default_kwargs | kwargs
        if split is not None:
            if isinstance(split, int):
                labels = self.split(split)
            else:
                labels = split
            node_colors = [labels[idx] for idx in list(nx_graph)]
            if palette is None:
                draw_kwargs["node_color"] = node_colors  # type: ignore
            else:
                draw_kwargs["node_color"] = [  # type: ignore
                    palette[c] for c in node_colors
                ]
                del draw_kwargs["cmap"]
        nx.draw(nx_graph, **draw_kwargs)

    def __repr__(self) -> str:
        repr_nodes = []
        nodes = list(self.nodes.values())
        for _, nodeval in zip(range(3), nodes):
            repr_nodes.append(str(nodeval))
        if len(nodes) > 3:
            repr_nodes.append("...")
        return f"Tree(root={self.root}, nodes={', '.join(repr_nodes)})"


@dataclasses.dataclass
class TreeRange:
    start: float
    end: float

    def adjust(self, parent: TreeRange) -> None:
        current_range = self.end - self.start
        parent_range = parent.end - parent.start
        adjusted_range = current_range * parent_range
        self.start = parent.start + parent_range * self.start
        self.end = self.start + adjusted_range

    @property
    def mid(self) -> float:
        return (self.start + self.end) * 0.5


def align_split_tree(split_nodes: dict[int, SplitNode]) -> dict[int, TreeRange]:
    @functools.cache
    def n_children(index: int) -> int:
        total = 1
        for child_index in split_nodes[index].children:
            total += n_children(child_index)
        return total

    def assign_space(nc_list: list[int]) -> list[TreeRange]:
        spaces = []
        nc_sum = sum(nc_list)
        prev = 0.0
        for nc in nc_list:
            assigned_space = nc / nc_sum
            spaces.append(TreeRange(prev, prev + assigned_space))
            prev += assigned_space
        return spaces

    spaces = {}

    def assign_space_to_node(
        index: int,
        parent: TreeRange | None = None,
    ) -> None:
        nc = len(split_nodes[index].children)
        if nc == 0:
            return
        nc_list = [n_children for index in split_nodes[index].children]
        assigned = assign_space(nc_list)
        if parent is not None:
            assigned.adjust(parent)
        for child in split_nodes[index].children:
            assign_space_to_node(child, assigned)
        spaces[index] = assigned

    nc_list = [
        n_children(index) for index, node in split_nodes.items() if node.parent is None
    ]
    root_space_list = assign_space(nc_list)
    for index, space in zip(split_nodes, root_space_list):
        assign_space_to_node(index, parent=space)

    return spaces
