"""
Approximate Rank-Order Clustering (AROC) algorithm.
https://arxiv.org/abs/1604.00989
"""
from functools import partial
from multiprocessing import Pool
import numpy as np
import pyflann


def aroc(features, n_neighbors, threshold, num_proc=4):
    """
    Calculates the pairwise distances between each face and merge all the faces
    with distances below a threshold.
    """
    # k-nearest neighbours using FLANN
    flann = pyflann.FLANN()
    params = flann.build_index(features, algorithm='kdtree', trees=4)
    nearest_neighbors, distances = flann.nn_index(
        features, n_neighbors, checks=params['checks'])

    # Build lookup table for nearest neighbors
    neighbor_lookup = {}
    for i in range(nearest_neighbors.shape[0]):
        neighbor_lookup[i] = nearest_neighbors[i]

    # Calculate pairwise distances
    pool = Pool(processes=num_proc)
    func = partial(_pairwise_distance, neighbor_lookup)
    results = pool.map(func, range(len(neighbor_lookup)))
    pool.close()
    distances = []
    for val in results:
        distances.append(val[0])
    distances = np.array(distances)

    # Build lookup table for nearest neighbors filtered by threshold
    for i, neighbor in neighbor_lookup.items():
        neighbor_lookup[i] = set(list(np.take(neighbor, np.where(
            distances[i] <= threshold)[0])))

    # Transitive merging
    clusters = []
    nodes = set(list(np.arange(0, distances.shape[0])))

    while nodes:
        node = nodes.pop()
        group = {node}  # Set of connected nodes
        queue = [node]  # Build a queue

        while queue:
            node = queue.pop(0)  # Get the first node to visit
            neighbors = neighbor_lookup[node]  # Neighbors of the curent node
            # Elements common to nodes and visit
            intersection = nodes.intersection(neighbors)
            # Intersection after removing elements found in group
            intersection.difference_update(group)
            # Nodes after removing elements found in intersection
            nodes.difference_update(intersection)
            group.update(intersection)  # Add connected neighbors
            queue.extend(intersection)  # Add to queue

        clusters.append(group)

    return clusters


def _pairwise_distance(neighbor_lookup, row_no):
    """
    Calculates the distance based on directly summing the presence/absence
    of shared nearest neighbors.
    """
    distance = np.zeros([1, len(neighbor_lookup[row_no])])
    row = neighbor_lookup[row_no]

    # For all neighbor(s) as face B in face A's nearest neighbors
    for i, neighbor in enumerate(row[1:]):
        oa_b = i + 1  # i-th face in the neighbor list of face A

        # Rank of face A in face B's neighbor list
        try:
            neighbors_face_b = neighbor_lookup[neighbor]
            ob_a = np.where(neighbors_face_b == row_no)[0][0] + 1
        except IndexError:
            ob_a = len(neighbor_lookup[row_no]) + 1

        # # of neighbor(s) that are not in face B's top k nearest neighbors
        neighbors_face_a = set(row[:oa_b])
        neighbors_face_b = set(neighbor_lookup[neighbor])
        d_ab = len(neighbors_face_a.difference(neighbors_face_b))

        # # of neighbor(s) that are not in face A's top k nearest neighbors
        neighbors_face_a = set(neighbor_lookup[row_no])
        neighbors_face_b = set(neighbor_lookup[neighbor][:ob_a])
        d_ba = len(neighbors_face_b.difference(neighbors_face_a))

        distance[0, oa_b] = float(d_ab + d_ba) / min(oa_b, ob_a) \
            if ob_a != len(neighbor_lookup[row_no]) + 1 \
            else 9999

    return distance
