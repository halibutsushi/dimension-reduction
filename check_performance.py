import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

BATCH_LIMIT = 500
SAMPLE_NUM = 10


def get_approximate_score(origin, reduced):
    num_vectors = origin.shape[0]
    if num_vectors > BATCH_LIMIT:
        ret = 0
        for i in range(SAMPLE_NUM):
            random_sample_index = np.random.randint(num_vectors, size=BATCH_LIMIT)
            origin_sample = origin[random_sample_index, :]
            reduced_sample = reduced[random_sample_index, :]
            ret += get_score(origin_sample, reduced_sample) / SAMPLE_NUM

        return ret

    return get_score(origin, reduced)


def get_score(origin, reduced):
    num_vectors = origin.shape[0]

    origin = origin / np.max(origin, axis=0)
    reduced = reduced / np.max(reduced, axis=0)

    num_vectors = origin.shape[0]

    origin_distances = np.zeros([num_vectors, num_vectors], dtype=np.int16)
    reduced_distances = np.zeros([num_vectors , num_vectors], dtype=np.int16)

    for i in range(0, num_vectors):
        diff = origin - origin[i]
        diff = diff * diff
        dist = np.sum(diff, axis=1)
        origin_distances[i] = dist.argsort().argsort()

        diff = reduced - reduced[i]
        diff = diff * diff
        dist = np.sum(diff, axis=1)
        reduced_distances[i] = dist.argsort().argsort()

    rank_change_count = np.zeros([num_vectors-1, num_vectors-1])

    for i in range(num_vectors):
        for j in range(0, num_vectors):
            if i == j:
                continue

            rank_change_count[origin_distances[i][j] - 1][reduced_distances[i][j] - 1] += 1

    q_values =[]
    lcmc_max_val_rank_range = 0
    lcmc_max_val = None

    for i in range(num_vectors - 1):
        q_value = np.sum(rank_change_count[:i + 1, :i + 1]) / ((i + 1) * (num_vectors))
        q_values.append(q_value)
        lcmc_val = q_value - (i + 1) / (num_vectors - 1)

        if lcmc_max_val is None or lcmc_val > lcmc_max_val:
            lcmc_max_val = lcmc_val
            lcmc_max_val_rank_range = i + 1

    ret = 0
    for i in range(lcmc_max_val_rank_range):
        ret += q_values[i]

    return ret / lcmc_max_val_rank_range


LOWER_DIMENSION = 3

if __name__ == "__main__":
    df = pd.read_csv('creditcard.csv', sep=',', header=0)
    df.fillna(0, inplace=True)
    array = df.as_matrix()

    pca = PCA(n_components=min(LOWER_DIMENSION, array.shape[1]), copy=False)
    pca.fit(array)
    pca_result = pca.transform(array)
    print(pca_result.shape)

    # print(get_score(array[:25, :], pca_result[:25, :]))
    print(get_approximate_score(array, pca_result))