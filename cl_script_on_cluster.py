import matplotlib
matplotlib.use('Agg')

import clustering
ca = clustering.ClusterAnalysis('./ims_2018_09_13.py')

X = ca.get_data()
Xs = ca.normalize(X, method='p')

res = ca.hierarchical_clustering(Xs)
print('DONE')
