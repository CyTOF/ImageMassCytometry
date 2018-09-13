import matplotlib
matplotlib.use('Agg')

import clustering
ca = clustering.ClusterAnalysis('./ims_2018_08_23.py')

X = ca.get_data()
Xnorm = ca.normalize(X, method='p')
Xs = ca.subsample(Xnorm, 2000)

res = ca.hierarchical_clustering(Xs)
print('DONE')