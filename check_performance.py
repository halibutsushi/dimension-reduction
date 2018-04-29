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


def compare_reuction_results(origin, reduced_data1, reduced_data2):
    score1 = get_approximate_score(array, reduced_data1)
    score2 = get_approximate_score(array, reduced_data2)

    if score1 > score2:
        return 1

    return 2


# a simple algorithm that just selects first n dimensions
def simple_dimension_reduction_algorithm(input_array, num_components):

    return input_array[:, :min(input_array.shape[1], num_components)]


if __name__ == "__main__":
    df = pd.read_csv('creditcard.csv', sep=',', header=0)
    df.fillna(0, inplace=True)
    array = df.as_matrix()

    for dim in range(1, 10):
        pca = PCA(n_components=min(dim, array.shape[1]), copy=False)
        pca.fit(array)
        pca_result = pca.transform(array)
        simple_result = simple_dimension_reduction_algorithm(array, dim)

        print('number of dimensions: {}'.format(dim))
        print('pca score: {}'.format(get_approximate_score(array, pca_result)))
        print('my simple algorithm score: {} \n'.format(get_approximate_score(array, simple_result)))


""" the test results (Surprisingly, my simple algorithm beats PCA up to dimension 9)

number of dimensions: 1
pca score: 0.32530948814986654
my simple algorithm score: 0.3286464773846756 

number of dimensions: 2
pca score: 0.3316649527596166
my simple algorithm score: 0.3812492238306688 

number of dimensions: 3
pca score: 0.31839595566998746
my simple algorithm score: 0.3964351732337903 

number of dimensions: 4
pca score: 0.33293613699784735
my simple algorithm score: 0.42310084459058894 

number of dimensions: 5
pca score: 0.3632980651286149
my simple algorithm score: 0.4457351576189828 

number of dimensions: 6
pca score: 0.3770226767955756
my simple algorithm score: 0.45231211530025417 

number of dimensions: 7
pca score: 0.3834171794216121
my simple algorithm score: 0.4592437711307776 

number of dimensions: 8
pca score: 0.386995813947692
my simple algorithm score: 0.4775338056576217 

number of dimensions: 9
pca score: 0.3912612270911561
my simple algorithm score: 0.4844064545864031 
"""
