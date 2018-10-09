"""
Experiment for AROC algorithm.
"""
import time
import scipy.io as sio
from aroc import aroc
from metrics import f1_score


def main():
    """
    Main.
    """
    # Load features
    data = sio.loadmat('data/LightenedCNN_C_lfw.mat')
    features = data['features']
    labels = data['labels_original'][0]
    label_lookup = {}
    for idx, label in enumerate(labels):
        label_lookup[idx] = int(label[0][:])
    print('Features shape: ', features.shape)

    start_time = time.time()
    clusters = aroc(features, 200, 1.1, 12)
    print('Time taken for clustering: {:.3f} seconds'.format(
        time.time() - start_time))

    _, _, _, precision, recall, score = f1_score(
        clusters, label_lookup)
    print('Clusters: {}  Precision: {:.3f}  Recall: {:.3f}  F1: {:.3f}'.format(
        len(clusters), precision, recall, score))


if __name__ == '__main__':
    main()
