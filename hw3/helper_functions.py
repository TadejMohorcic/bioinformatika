"""Helper functions for HW3"""
import numpy as np
from copy import deepcopy
from matplotlib.axes import Axes
from collections import defaultdict


class Node:
    def __init__(
        self,
        name: str,
        left: "Node",
        left_distance: float,
        right: "Node",
        right_distance: float,
        confidence: float = None,
    ):
        """A node in a binary tree produced by neighbor joining algorithm.

        Parameters
        ----------
        name: str
            Name of the node.
        left: Node
            Left child.
        left_distance: float
            The distance to the left child.
        right: Node
            Right child.
        right_distance: float
            The distance to the right child.
        confidence: float
            The confidence level of the split determined by the bootstrap method.
            Only used if you implement Bonus Problem 1.

        Notes
        -----
        The current public API needs to remain as it is, i.e., don't change the
        names of the properties in the template, as the tests expect this kind
        of structure. However, feel free to add any methods/properties/attributes
        that you might need in your tree construction.

        """
        self.name = name
        self.left = left
        self.left_distance = left_distance
        self.right = right
        self.right_distance = right_distance
        self.confidence = confidence


def neighbor_joining(distances: np.ndarray, labels: list) -> Node:
    """The Neighbor-Joining algorithm.

    For the same results as in the later test dendrograms;
    add new nodes to the end of the list/matrix and
    in case of ties, use np.argmin to choose the joining pair.

    Parameters
    ----------
    distances: np.ndarray
        A 2d square, symmetric distance matrix containing distances between
        data points. The diagonal entries should always be zero; d(x, x) = 0.
    labels: list
        A list of labels corresponding to entries in the distances matrix.
        Use them to set names of nodes.

    Returns
    -------
    Node
        A root node of the neighbor joining tree.

    """
    # Construct the original nodes before hand, as they all have no left or right child.
    nodes = []

    for label in labels:
        nodes.append(Node(label, None, 0, None, 0))

    # In each step, we remove 2 nodes and add one, so we will loop until n > 2, and then root the
    # tree. Each of the new nodes will be named "1", "2", "3", ... and the last one will be "ROOT".
    n = len(nodes)
    node_name = 1

    while n > 2:

        # Calculate the net divergence for each vertex.
        net_divergence = []

        for i in range(n):
            net_divergence.append(np.sum(distances[i, :]))

        # Calculate the score matrix M.
        M = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                M[i, j] = (n - 2) * distances[i, j] - net_divergence[i] - net_divergence[j]

        # Get the minimal score of the matrix M, and get indices i and j, for merging.
        min_score = np.argmin(M)
        row, col = min_score // n, min_score % n

        # Calculate the new distances.
        distance_row = (distances[row, col] + (net_divergence[row] - net_divergence[col]) / (n - 2)) / 2
        distance_col = distances[row, col] - distance_row

        new_row = []
        for i in range(n):
            if i == row or i == col:
                continue
            new_row.append((distances[i, row] + distances[i, col] - distances[row, col]) / 2)

        # Construct the new matrix by removing the rows and columns with indices row and col. We will
        # remove the larger index first so that they stay the same (If we remove min first, then max
        # becomes max - 1).
        min_id, max_id = min(row, col), max(row, col)
        kept_distances = distances

        # Remove row and col.
        for i in [max_id, min_id]:
            kept_distances = np.vstack((kept_distances[:i, :], kept_distances[i+1:, :]))
            kept_distances = np.hstack((kept_distances[:, :i], kept_distances[:, i+1:]))

        # Append the new row and column to the distance matrix.
        new_distances = np.zeros((n-1, n-1))
        new_distances[:n-2, :n-2] = kept_distances

        for i in range(len(new_row)):
            new_distances[n-2, i] = new_row[i]
            new_distances[i, n-2] = new_row[i]

        # Update the node_list, associate distance_row with n1 if max_id = row
        n1, n2 = nodes.pop(max_id), nodes.pop(min_id)

        if max_id == row:
            nodes.append(Node(str(node_name), n1, distance_row, n2, distance_col))
        else:
            nodes.append(Node(str(node_name), n1, distance_col, n2, distance_row))

        # Update i, n, and distances-
        node_name += 1
        n -= 1
        distances = new_distances

    # Construct the node "Root" from the two remaining nodes.
    dist = distances[0, 1] / 2
    n1, n2 = nodes.pop(), nodes.pop()

    return Node("ROOT", n1, dist, n2, dist)

# Count how many leaves a given node has.
def number_of_leaves(node):
        if node == None:
            return 0
        elif node.left == None:
            return 1
        else:
            return number_of_leaves(node.left) + number_of_leaves(node.right)

def plot_nj_tree(tree: Node, ax: Axes = None, color: dict = dict()) -> None:
    """A function for plotting neighbor joining phylogeny dendrogram.

    Parameters
    ----------
    tree: Node
        The root of the phylogenetic tree produced by `neighbor_joining(...)`.
    ax: Axes
        A matplotlib Axes object which should be used for plotting.
    kwargs
        Feel free to replace/use these with any additional arguments you need.
        But make sure your function can work without them, for testing purposes.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>>
    >>> tree = neighbor_joining(distances)
    >>> fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    >>> plot_nj_tree(tree=tree, ax=ax)
    >>> fig.savefig("example.png")

    """
    # Each time we calculate the midpoint - number of leaves to the left + 0.5.
    midpoint = number_of_leaves(tree.left) + 0.5
    lines = [[[-100, 0], [midpoint, midpoint]]]
    labels = {}

    # Loop while this is not empty.
    to_check = [(tree, 0.5, 0)]

    while to_check != []:
        node, height, x_cord = to_check.pop()

        if node.left != None and node.right != None:
            midpoint = number_of_leaves(node.left) + height
            # Calculate how many leaves does the left of the left node have and the left of the
            # right node have - so we get a horizontal line right in between these two groups.
            low = number_of_leaves(node.left.left) + height if number_of_leaves(node.left.left) != 0 else height + 0.5
            high = number_of_leaves(node.right.left) + midpoint if number_of_leaves(node.right.left) != 0 else midpoint + 0.5
            # Add horizontal and vertical lines.
            lines.append([[x_cord, x_cord], [low, high]])
            lines.append([[x_cord, x_cord + node.left_distance], [low, low]])
            lines.append([[x_cord, x_cord + node.right_distance], [high, high]])
            # Add left and right node to the loop.
            to_check.append((node.left, height, x_cord + node.left_distance))
            to_check.append((node.right, midpoint, x_cord + node.right_distance))
        else:
            # If node is a leaf, then just add it's label.
            labels[node.name] = [x_cord + 200, height + 0.3]

    # Plot the lines.
    for x, y in lines:
        ax.plot(x, y, color='black')

    # Plot the labels.
    for label in labels:
        x, y = labels[label]
        c = color.get(label, ["black"])[0]
        ax.text(x, y, label, fontsize = 10, color = c)

    return ax

def _find_a_parent_to_node(tree: Node, node: Node) -> tuple:#
    """Utility function for reroot_tree"""
    stack = [tree]

    while len(stack) > 0:

        current_node = stack.pop()
        if node.name == current_node.left.name:
            return current_node, "left"
        elif node.name == current_node.right.name:
            return current_node, "right"

        stack += [
            n for n in [current_node.left, current_node.right] if n.left is not None
        ]

    return None


def _remove_child_from_parent(parent_node: Node, child_location: str) -> None:
    """Utility function for reroot_tree"""
    setattr(parent_node, child_location, None)
    setattr(parent_node, f"{child_location}_distance", 0.0)


def reroot_tree(original_tree: Node, outgroup_node: Node) -> Node:
    """A function to create a new root and invert a tree accordingly.

    This function reroots tree with nodes in original format. If you
    added any other relational parameters to your nodes, these parameters
    will not be inverted! You can modify this implementation or create
    additional functions to fix them.

    Parameters
    ----------
    original_tree: Node
        A root node of the original tree.
    outgroup_node: Node
        A Node to set as an outgroup (already included in a tree).
        Find it by it's name and then use it as parameter.

    Returns
    -------
    Node
        Inverted tree with a new root node.
    """
    tree = deepcopy(original_tree)

    parent, child_loc = _find_a_parent_to_node(tree, outgroup_node)
    distance = getattr(parent, f"{child_loc}_distance")
    _remove_child_from_parent(parent, child_loc)

    new_root = Node("new_root", parent, distance / 2, outgroup_node, distance / 2)
    child = parent

    while tree != child:
        parent, child_loc = _find_a_parent_to_node(tree, child)

        distance = getattr(parent, f"{child_loc}_distance")
        _remove_child_from_parent(parent, child_loc)

        empty_side = "left" if child.left is None else "right"
        setattr(child, f"{empty_side}_distance", distance)
        setattr(child, empty_side, parent)

        if tree.name == parent.name:
            break
        child = parent

    other_child_loc = "right" if child_loc == "left" else "left"
    other_child_distance = getattr(parent, f"{other_child_loc}_distance")

    setattr(child, f"{empty_side}_distance", other_child_distance + distance)
    setattr(child, empty_side, getattr(parent, other_child_loc))

    return new_root


def sort_children_by_leaves(tree: Node) -> None:
    """Sort the children of a tree by their corresponding number of leaves.

    The tree can be changed inplace.

    Paramteres
    ----------
    tree: Node
        The root node of the tree.

    """
    if tree.left == None or tree.right == None:
        return tree
    else:
        left = number_of_leaves(tree.left)
        right = number_of_leaves(tree.right)
        left_dist = tree.left_distance
        right_dist = tree.right_distance
        if left > right:
            return Node(tree.name, sort_children_by_leaves(tree.right), right_dist, sort_children_by_leaves(tree.left), left_dist)
        else:
            return Node(tree.name, sort_children_by_leaves(tree.left), left_dist, sort_children_by_leaves(tree.right), right_dist)

# Global alignment from previous homework.
def global_alignment(seq1, seq2, scoring_function):
    """Global sequence alignment using the Needleman–Wunsch algorithm.

    Indels should be denoted with the "-" character.

    Parameters
    ----------
    seq1: str
        First sequence to be aligned.
    seq2: str
        Second sequence to be aligned.
    scoring_function: Callable

    Returns
    -------
    str
        First aligned sequence.
    str
        Second aligned sequence.
    float
        Final score of the alignment.

    Examples
    --------
    >>> global_alignment("abracadabra", "dabarakadara", lambda x, y: [-1, 1][x == y])
    ('-ab-racadabra', 'dabarakada-ra', 5.0)

    Other alignments are not possible.

    """
    score_matrix = defaultdict(int)
    backtracking = {}

    score_matrix[0, 0] = 0

    for i in range(1, len(seq1) + 1):
        score_matrix[i, 0], backtracking[i, 0] = float(score_matrix[i - 1, 0] + scoring_function('*', seq1[i - 1])), (i - 1, 0)

    for i in range(1, len(seq2) + 1):
        score_matrix[0, i], backtracking[0, i] = float(score_matrix[0, i - 1] + scoring_function('*', seq2[i - 1])), (0, i - 1)

    for i in range(1, len(seq1) + 1):
        for j in range(1, len(seq2) + 1):
            score_matrix[i, j], backtracking[i, j] = max((float(score_matrix[i - 1, j] + scoring_function('*', seq1[i - 1])), (i - 1, j)),
                                                         (float(score_matrix[i, j - 1] + scoring_function('*', seq2[j - 1])), (i, j - 1)),
                                                         (float(score_matrix[i - 1, j - 1] + scoring_function(seq1[i - 1], seq2[j - 1])), (i - 1, j - 1)))

    final_score = score_matrix[max(score_matrix)]

    alignment_1, alignment_2 = '', ''
    current = (len(seq1), len(seq2))
    
    while current != (0, 0):
        prev = backtracking[current]
        alignment_1 += '-' if prev[0] == current[0] else seq1[prev[0]]
        alignment_2 += '-' if prev[1] == current[1] else seq2[prev[1]]
        current = backtracking[current]

    return alignment_1[::-1], alignment_2[::-1], final_score

def plot_nj_tree_radial(tree: Node, ax: Axes = None, **kwargs) -> None:
    """A function for plotting neighbor joining phylogeny dendrogram
    with a radial layout.

    Parameters
    ----------
    tree: Node
        The root of the phylogenetic tree produced by `neighbor_joining(...)`.
    ax: Axes
        A matplotlib Axes object which should be used for plotting.
    kwargs
        Feel free to replace/use these with any additional arguments you need.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>>
    >>> tree = neighbor_joining(distances)
    >>> fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    >>> plot_nj_tree_radial(tree=tree, ax=ax)
    >>> fig.savefig("example_radial.png")

    """
    raise NotImplementedError()