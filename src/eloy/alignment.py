from scipy.spatial import cKDTree
from twirl import quads
from twirl.geometry import get_transform_matrix, pad
from twirl.match import count_cross_match
import numpy as np


def cross_match(S1, S2, tolerance=10, return_idxs=False, none=True):
    # cleaning
    s1 = S1.copy()
    s2 = S2.copy()

    s1[np.any(np.isnan(s1), 1)] = (1e15, 1e15)
    s2[np.any(np.isnan(s2), 1)] = (1e15, 1e15)

    # matching
    matches = []

    for i, s in enumerate(s1):
        distances = np.linalg.norm(s - s2, axis=1)
        closest = np.argmin(distances)
        if distances[closest] < tolerance:
            matches.append([i, closest])
        else:
            if none:
                matches.append([i, np.nan])

    matches = np.array(matches)
    matches = matches[np.all(~np.isnan(matches), 1)]
    matches = matches.astype(int)

    if return_idxs:
        return matches
    else:
        if len(matches) > 0:
            return s1[matches[:, 0]], s2[matches[:, 1]]
        else:
            return np.array([]), np.array([])


def twirl_reference(coords):
    quads_ref, asterisms_ref = quads.hashes(coords)
    tree_ref = cKDTree(quads_ref)
    return tree_ref, asterisms_ref


def rotation_matrix(coords, ref_coords, reference, tolerance=2, refine=True, rtol=0.02):
    tree_ref, asterisms_ref = reference
    quads_image, asterisms_image = quads.hashes(coords)
    tree_image = cKDTree(quads_image)
    min_match = 0.7

    ball_query = tree_image.query_ball_tree(tree_ref, r=rtol)
    pairs = []
    for i, j in enumerate(ball_query):
        if len(j) > 0:
            pairs += [[i, k] for k in j]

    matches = []

    for i, j in pairs:
        M = get_transform_matrix(asterisms_ref[j], asterisms_image[i])
        test = (M @ pad(ref_coords).T)[0:2].T
        match = count_cross_match(coords, test, tolerance)
        matches.append(match)

        if min_match is not None:
            if isinstance(min_match, float):
                if match >= min_match * len(coords):
                    break

    if len(matches) == 0:
        return None
    else:
        i, j = pairs[np.argmax(matches)]
        if refine:
            M = get_transform_matrix(asterisms_ref[j], asterisms_image[i])
            test = (M @ pad(ref_coords).T)[0:2].T
            s1, s2 = cross_match(coords, test, tolerance=tolerance, return_idxs=True).T
            M = get_transform_matrix(ref_coords[s2], coords[s1])
        else:
            M = get_transform_matrix(asterisms_ref[j], asterisms_image[i])

        return M
