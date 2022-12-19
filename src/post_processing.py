import numpy as np
from scipy import cluster as clst
from sklearn.decomposition import PCA


def cluster_based(representations, n_cluster: int, n_pc: int, hidden_size: int = 768):
    """ Improving Isotropy of input representations using cluster-based method

        representations:
            input representations numpy array(n_samples, n_dimension)
        n_cluster:
            the number of clusters
        n_pc:
            the number of directions to be discarded
        hidden_size:
            model hidden size

        returns:
            isotropic representations (n_samples, n_dimension)
    """

    centroids, labels = clst.vq.kmeans2(representations, n_cluster, minit='points', missing='warn', check_finite=True)
    cluster_means = []
    for i in range(max(labels) + 1):
        summ = np.zeros([1, hidden_size])
        for j in np.nonzero(labels == i)[0]:
            summ = np.add(summ, representations[j])
        cluster_means.append(summ / len(labels[labels == i]))

    zero_mean_representations = []
    for i in range(len(representations)):
        zero_mean_representations.append((representations[i]) - cluster_means[labels[i]])

    cluster_representations = {}
    for i in range(n_cluster):
        cluster_representations.update({i: {}})
        for j in range(len(representations)):
            if labels[j] == i:
                cluster_representations[i].update({j: zero_mean_representations[j]})

    # ...why couldn't that have been done in one step?
    cluster_representations2 = []
    for j in range(n_cluster):
        cluster_representations2.append([])
        for key, value in cluster_representations[j].items():
            cluster_representations2[j].append(value)

    # probably unnecessary, gives you dtype object and a deprecation warning
    # cluster_representations2 = np.array(cluster_representations2)

    model = PCA()
    post_rep = np.zeros((representations.shape[0], representations.shape[1]))

    for i in range(n_cluster):
        model.fit(np.array(cluster_representations2[i]).reshape((-1, hidden_size)))
        component = np.reshape(model.components_, (-1, hidden_size))

        for index in cluster_representations[i]:
            sum_vec = np.zeros((1, hidden_size))

            for j in range(n_pc):
                sum_vec = sum_vec + np.dot(cluster_representations[i][index],
                                           np.expand_dims(np.transpose(component)[:, j], 1)) * component[j]

            post_rep[index] = cluster_representations[i][index] - sum_vec

    return post_rep


