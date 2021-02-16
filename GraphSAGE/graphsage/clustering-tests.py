import sklearn.cluster as sklc
from sklearn.externals import joblib
from graphsage.utils import load_data
import networkx as nx
def run_cluster(features, n_clusters, max_iter, random_state, eps, min_samples):
    #model = sklc.KMeans(n_clusters, max_iter = max_iter, random_state=random_state)
    model = sklc.DBSCAN(eps = eps, min_samples = min_samples, n_jobs = 4)
    # model = sklc.SpectralClustering(n_clusters)
    model.fit(features)
    print('The number of clusters is: ', len(set(model.labels_)))
    return model
G, feats, id_map, walks, class_map = load_data('/home/thummala/graph-datasets/ppi/ppi')
print('Starting the clustering...')
Cmodel=run_cluster(feats,1,1,123,10,10)
print('Done')
joblib.dump(Cmodel, '/home/thummala/GraphSage/GraphSAGE/clustermodels/ppi-dbscan.skl')
