"""Tree data implementation used for analyzing agent's phylogeny."""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Iterable, Sequence
from typing import Any
from weakref import ReferenceType
from weakref import ref as make_weakref

import networkx as nx
import numpy as np
from networkx.drawing.nx_agraph import graphviz_layout
from numpy.typing import NDArray
from pyarrow import Table


datafield = functools.partial(dataclasses.field, compare=False, hash=False, repr=False)
_ROOT_INDEX = -1


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
    def n_total_children(self) -> int:
        total = 0
        for child, _ in self.children:
            total += 1 + child.n_total_children
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
        if parent is not None and parent.index != _ROOT_INDEX:
            yield from parent.ancestors()

    def traverse(
        self,
        preorder: bool = True,
        include_self: bool = True,
    ) -> Iterable[Node]:
        if include_self and preorder:
            yield self
        for child, _ in self.children:
            yield from child.traverse(preorder)
        if include_self and not preorder:
            yield self


@dataclasses.dataclass
class Edge:
    parent: Node
    child: Node

    def __hash__(self) -> int:
        return self.parent.index * (2**30) + self.child.index

    def __eq__(self, other: Edge) -> bool:
        return self.parent == other.parent and self.parent == other.parent

    def __lt__(self, other: Edge) -> bool:
        if self.parent.index == other.parent.index:
            return self.child.index < other.child.index
        else:
            return self.parent.index < other.parent.index


@dataclasses.dataclass
class SplitNode:
    size: int
    reward_mean: dict[str, float] | None = None
    children: list[int] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Tree:
    root: Node
    nodes: dict[int, Node]

    @staticmethod
    def from_iter(
        iterator: Iterable[tuple[int, int] | tuple[int, int, dict]], root_idx: int = 0
    ) -> Tree:
        nodes = {}
        root = Node(index=_ROOT_INDEX, is_root=True)

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
        return Tree(root, nodes)

    @staticmethod
    def from_table(
        table: Table,
        initial_population: int | None = None,
        root_idx: int = -1,
    ) -> Tree:
        birth_steps = {}

        def table_iter() -> Iterable[tuple[int, int, dict]]:
            for batch in table.to_batches():
                for row in batch.to_pylist():
                    idx = row.pop("unique_id")
                    birth_steps[idx] = row.pop("birthtime")
                    yield idx, row.pop("parent"), row

        tree = Tree.from_iter(table_iter(), root_idx=root_idx)
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
        method: str = "greedy",
        n_trial: int = 5,
        reward_keys: list[str] | None = None,
    ) -> dict[int, SplitNode]:
        if method == "greedy":
            split_nodes = self._split_greedy(min_group_size)
        elif method == "reward":
            split_nodes = self._split_reward_mean(min_group_size, n_trial, reward_keys)
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

    def _split_greedy(self, min_group_size) -> dict[int, SplitNode]:
        split_nodes = {}

        def split(node: Node, threshold: int) -> int:
            size = 0
            for child in node.children:
                # Number of children that are not splitted
                n_existing_children = split(child, threshold)
                size += n_existing_children

            if size >= threshold:
                split_nodes[node.index] = SplitNode(size)
                return 0
            else:
                return size

        def find_children(node: Node) -> list[int]:
            children = []
            for child in node.children:
                children += find_children(child)

            if node in split_nodes:
                split_nodes[node.index].children = children
                return list[node.index]
            else:
                return children

        for root in self.root.children:
            size = split(root, min_group_size)
            if size >= min_group_size:
                split_nodes[root.index] = SplitNode(size)
                find_children(root)

        return split_nodes

    def _split_reward_mean(
        self,
        min_group_size: int,
        n_trial: int,
        reward_keys: list[str],
    ) -> dict[int, str]:
        split_nodes = {}
        split_edges = set()

        @functools.cache
        def compute_reward_mean(
            node: Node,
            n_split: int = 0,
            is_root: bool = False,
        ) -> tuple[int, dict[str, float]]:
            if is_root:
                size_list = [0]
                reward_mean_lists = {key: [0.0] for key in reward_keys}
            else:
                if reward_keys[0] not in node.info:
                    return 0, {key: 0.0 for key in reward_keys}
                size_list = [1]
                reward_mean_lists = {key: [node.info[key]] for key in reward_keys}

            for child in node.children:
                if (node.index, child.index) in split_edges:
                    continue
                n_children, reward_mean = compute_reward_mean(child, n_split=n_split)
                size_list.append(n_children)
                for key, rmean in reward_mean.items():
                    reward_mean_lists[key].append(rmean)

            total_size = np.sum(size_list)
            rmean_dict = {}
            for key, rmean in reward_mean_lists.items():
                rsum = np.sum([nc * rm for nc, rm in zip(size_list, rmean)])
                rmean_dict[key] = rsum / total_size
            return total_size, rmean_dict

        def find_maxdiff_edge(n_split: int, min_group_size: int) -> tuple[float, Edge]:
            max_effect = 0.0
            max_effect_edge = None
            for edge in self.all_edges():
                if (edge.parent.index, edge.child.index) in split_edges:
                    continue
                parent_size, parent_reward = compute_reward_mean(
                    edge.parent,
                    n_split=n_split,
                )
                child_size, child_reward = compute_reward_mean(
                    edge.child,
                    n_split=n_split,
                )
                if (
                    child_size < min_group_size
                    or (parent_size - child_size) < min_group_size
                ):
                    continue
            assert parent_size > child_size, (parent_size, child_size)
            split_size = parent_size - child_size
            total_diff = 0.0
            for key in reward_keys:
                parent_rew_total = parent_reward[key] * parent_size
                child_rew_total = child_reward[key] * child_size
                split_rew = (parent_rew_total - child_rew_total) / split_size
                total_diff += (child_reward[key] - split_rew) ** 2
            effect = total_diff**0.5
            if effect > max_effect:
                max_effect = effect
                max_effect_edge = edge
            return max_effect, max_effect_edge

        for i in range(n_trial):
            maxe, edge = find_maxdiff_edge(i, min_group_size)
            parent_size, parent_reward = compute_reward_mean(edge.parent, n_split=i)
            child_size, child_reward = compute_reward_mean(edge.child, n_split=i)
            size_new = parent_size - child_size
            rew_new = {}
            for key in parent_reward:
                rew_new[key] = (
                    parent_reward[key] * parent_size - child_reward[key] * child_size
                ) / size_new
            # Make nodes
            if edge.parent.index in split_nodes:
                # Add child
                split_nodes[edge.parent.index].size = size_new
                split_nodes[edge.parent.index].reward_mean = rew_new
                split_nodes[edge.parent.index].children.append(edge.child.index)
            else:
                split_nodes[edge.parent.index] = SplitNode(
                    size_new,
                    rew_new,
                    children=[edge.child.index],
                )
            # Find Parent
            ancestor = edge.parent.parent
            while ancestor is not None:
                if ancestor.index in split_nodes:
                    if edge.parent.index not in split_nodes[ancestor.index].children:
                        split_nodes[ancestor.index].children.append(edge.parent.index)
                        split_nodes[ancestor.index].size -= size_new
                    break
                ancestor = ancestor.parent
            split_nodes[edge.child.index] = SplitNode(child_size, child_reward)
            split_edges.add((edge.parent.index, edge.child.index))

        return split_nodes

    def as_datadict(self, split: int | dict[int, int] | None) -> dict[str, NDArray]:
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

        if split is not None:
            if isinstance(split, int):
                labels = self.split(split)
                split_group_size = split
            else:
                labels = split
                split_group_size = len(set(labels.values()))
            data_dict["label"] = np.array([labels[idx] for idx in indices], dtype=int)
            multi_labels = self.multilabel_split(split_group_size)
            for label, labelset in enumerate(multi_labels):
                bool_list = [idx in labelset for idx in indices]
                data_dict[f"in-label-{label}"] = np.array(bool_list, dtype=bool)

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
