import os
import pickle
import argparse
import pdb 

# numpy
import numpy as np 

from scipy.stats import gaussian_kde
from scipy.cluster import hierarchy

import pandas as pd 

import matplotlib.pyplot as plt

# seaborn for plotting
import seaborn as sns

# skimage imports
from skimage.measure import label, regionprops
from skimage.color import grey2rgb
import skimage.io 

import colorsys

# sklearn imports
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.utils import resample


# project imports
from settings import Settings, overwrite_settings
from sequence_importer import SequenceImporter
from fissure_detection import FissureDetection
from cell_detection import SpotDetector, CellDetection
from process_ilastik import Ilastik
from visualization import Overlays

class ClusterAnalysis(object):
    
    def __init__(self, settings_filename=None, settings=None, 
                 tissue_id=None):
        if settings is None and settings_filename is None:
            raise ValueError("Either a settings object or a settings filename has to be given.")
        if not settings is None:
            self.settings = settings
        elif not settings_filename is None:
            self.settings = Settings(os.path.abspath(settings_filename), dctGlobals=globals())

        if not tissue_id is None:
            print('Overwriting settings ... ')
            self.settings = overwrite_settings(self.settings, tissue_id)
            print('tissue id: %s' % tissue_id)
            print('output_folder: %s' % self.settings.output_folder)

        for folder in self.settings.makefolders:
            if not os.path.exists(folder):
                os.makedirs(folder)
                
        self.cluster_folder = os.path.join(self.settings.output_folder, 'clustering')
        self.plot_projection_folder = os.path.join(self.settings.output_folder, 'projections')
        for folder in [self.cluster_folder, self.plot_projection_folder]: 
            if not os.path.isdir(folder):
                os.makedirs(folder)
            
        self.sp = SpotDetector(settings=self.settings)
        self.fissure = FissureDetection(settings=self.settings)
        self.cell_detector = CellDetection(settings=self.settings)
        self.ilastik = Ilastik(settings=self.settings)
        
        #self.markers = ['Bcl-6', 'CD279(PD-1)', 'CD3',
        #                'CD45', 'CD14',
        #                'CD370', 'CD141',
        #                'CD11c',  'CD1c-biotin-NA',  'HLA-DR',
        #                'CD123', 'CD303(BDCA2)']

        #CD1c : remove (too bad)
        #ICOS : enlever (similaire a PD1)
        #CD185 : does not look good (should only be in B-region)
        #CD56 : probably wrong marker --> check in the pipeline
        #CD11b : unsure what to do ... artefact, spatial effect ? 
        #CD303 : high noise level, there should be quite a number of cells in T-region.

        # all markers except for empty, 
        self.all_markers = ['CD206', 'IL-21', 'CD185(CXCR5)', 'CD45', 'CXCL13', 
                            'CD1c-biotin-NA', 'CD303(BDCA2)', 'CD11b', 'Bcl-6', 'CD45RA', 
                            'E-Cadherin', 'CD141', 'CD123', 'CD68', 'CD279(PD-1)', 
                            'HLA-DR', 'aSMA', 'CD370', 'CD11c', 'CD19', 'ICOS', 
                            'CD56(NCAM)', 'CD3', 'CD14']
        
        self.acceptable_markers = ['CD370', 'CD19', 'CD45RA', 'HLA-DR', 'IL-21',
                                   'CXCL13', 'CD206', 'CD279(PD-1)', 'CD45',
                                   'CD3', 'Bcl-6', 'CD14', 'CD141', 'CD68', 'CD123',
                                   'CD11c', 'E-Cadherin', 'aSMA']

        self.markers = self.acceptable_markers
        self.nb_clusters = 40
        # + CD19 (cellules B), abondant !
        # + E-cadherin (la crypte)
        # + alphasma (vaisseaux sanguins)
 
    def get_data(self, force=False):
        
        if not force:
            print('new calculation not forced ... ')
            
        print('Reading data ... ')
        
        si = SequenceImporter(self.markers)
        img, channel_names = si(self.settings.input_folder)

        self.channel_names = channel_names

        # get the individual cells.
        ws = self.cell_detector.get_image(False)
        
        # number of samples
        N = ws.max()
        
        # number of features
        P = len(channel_names)

        X = np.zeros((N, P))

        for j in range(P):
            props = regionprops(ws, img[:,:,j])
            intensities = np.array([props[k]['mean_intensity'] for k in range(len(props))])
            
            # centers = np.array([props[k]['centroid'] for k in range(len(props))])
            X[:,j] = intensities
        
        return X 
    
    def normalize(self, X, method='z'):
        if method[0] == 'z':
            print('calculating z-score')
            Xnorm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        elif method[0] == 'p': 
            print('calculating percentile normalization')
            perc = np.percentile(X, [1, 99], axis=0)
            Xnorm = (X - perc[0]) / (perc[1] - perc[0])
        elif method[0] == 'm': 
            print('calculating min-max normalization')
            maxval = np.max(X, axis=0)
            minval = np.min(X, axis=0)
            Xnorm = (X - minval) / (maxval - minval)
        return Xnorm
    
    def pca(self, X=None, save_fig=True):
        if X is None:
            X = self.get_data()
            X = self.normalize(X)
            
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(X)
        
        if save_fig:
            fig = plt.figure(figsize = (8,8))
            ax = fig.add_subplot(1,1,1) 
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_title('PCA %s' % self.settings.dataset)
            ax.scatter(principalComponents[:,0],
                       principalComponents[:,1],
                       c="red", 
                       s=15)
            ax.grid()
            plt.savefig(os.path.join(self.plot_projection_folder, 'pca.png'))
            plt.close()
        #pdb.set_trace()
        return principalComponents
    
    def subsample(self, X, K):
        indices = np.arange(X.shape[0])
        Xsample, ind_sample = resample(X, indices, replace=False, n_samples=K)
        return Xsample, ind_sample
    
    def tsne(self, X=None, save_fig=True):
        if X is None:
            X = self.get_data()
            X = self.normalize(X)
            X, indices = self.subsample(X, 1200)

        tsne = TSNE(n_components=2).fit_transform(X)
        
        if save_fig:
            fig = plt.figure(figsize = (8,8))
            ax = fig.add_subplot(1,1,1) 
            ax.set_xlabel('tsne component 1')
            ax.set_ylabel('tsne component 2')
            ax.set_title('TSNE %s' % self.settings.dataset)
            ax.scatter(tsne[:,0],
                       tsne[:,1],
                       c="red", 
                       s=10)
            ax.grid()
            plt.savefig(os.path.join(self.plot_projection_folder, 'tsne.png'))
            plt.close()
        #pdb.set_trace()
        return tsne
        
    def hierarchical_clustering(self, X=None, markers=None, filename_ext='',
                                export=True, indices=None):
                        
        Xs = X.copy()
        
        if markers is None:
            markers = self.markers
        else:
            marker_indices = np.array([self.markers.index(x) for x in markers])
            Xs = Xs[:,marker_indices]

        columns = markers
        df = pd.DataFrame(Xs, columns=columns)

        # color palette for clusters
        col_pal = sns.color_palette("husl", self.nb_clusters)

        # perform clustering
        cluster_res = hierarchy.linkage(Xs, method='ward', metric='euclidean')

        # cut the tree
        ct = hierarchy.cut_tree(cluster_res, n_clusters=self.nb_clusters)
        cluster_assignment = ct.T[0]
        col_vec = np.array(col_pal)[ct.T[0]]
        
        # main result
        res = dict(zip(range(self.nb_clusters), [np.where(cluster_assignment==i) for i in range(self.nb_clusters)]))
        
        cmap = sns.light_palette("navy", as_cmap=True)
        g = sns.clustermap(df, row_linkage=cluster_res, robust=True, cmap=cmap, 
                           col_cluster=False, yticklabels=False,
                           row_colors=col_vec)

        # legend for class colors
        for label in range(self.nb_clusters):
            g.ax_col_dendrogram.bar(0, 0, color=col_pal[label],
                                    label='%i(%i)' % (label, len(res[label][0])), 
                                    linewidth=0)
        g.ax_col_dendrogram.legend(loc="center", ncol=5)

        # save file
        g.savefig(os.path.join(self.cluster_folder, 'ward%s.png' % filename_ext))
 
    
        full_res = {'res': res, 'colors': col_pal, 'linkage': cluster_res, 'indices': indices}
        if export:
            filename = os.path.join(self.cluster_folder, 'cluster_assignment%s.pickle' % filename_ext)
            fp = open(filename, 'wb')
            pickle.dump(full_res, fp)
            fp.close()

        # save dendrogram
        #fig = plt.figure(figsize=(Xs.shape[0] / 10, 8))
        #dn = hierarchy.dendrogram(cluster_res)
        #plt.savefig(os.path.join(self.cluster_folder, 'dendrogram_ward%s.pdf' % filename_ext))
        #plt.close('all')
                
        return full_res
    
    def export_cell_cluster_maps(self, cluster_filename):
        
        filename = os.path.join(self.cluster_folder, 'cluster_assignment%s.pickle' % cluster_filename)
        if not os.path.isfile(filename): 
            filename = cluster_filename
            cluster_filename = ''
        if not os.path.isfile(filename):
            raise ValueError("A filename or filename extension must be given."
                             "First, we interpret cluster_filename as a filename extension and look for"
                             "the corresponding pickle file in %s" % self.cluster_folder)

        clustermaps_folder = os.path.join(self.cluster_folder, 'clustermaps')
        if not os.path.isdir(clustermaps_folder):
            os.makedirs(clustermaps_folder)

        # import cluster results
        fp = open(filename, 'rb')
        full_res = pickle.load(fp)
        fp.close()
        cluster_assignment = full_res['res']
        col_pal = full_res['colors']
        
        #full_res = {'res': res, 'colors': col_pal, 'linkage': cluster_res, 'indices': indices}
        
        # get the individual cells.
        ws = self.cell_detector.get_image(False)

        # number of cells
        N = np.max(ws)

        background, b_region, t_region = self.ilastik.read_region_images()
        region_image = np.zeros(b_region.shape, dtype=np.uint8)
        region_image[b_region>0] = 100
        region_image[t_region>0] = 200
        
        for cluster_label in cluster_assignment:
            
            bin_vec = np.zeros(N + 1, dtype=np.float)
            labels = np.array(cluster_assignment[cluster_label][0])
            if not full_res['indices'] is None:
                labels = full_res['indices'][labels]
            else:
                labels = labels + 1
            bin_vec[labels] = 1

            color_matrix_hsv = np.array((N + 1) * colorsys.rgb_to_hsv(*col_pal[cluster_label])).reshape((N + 1, 3))         
            noise_vec_l = np.random.rand(N + 1)
            noise_vec_s = np.random.rand(N + 1)

            color_matrix_hsv[:,2] = color_matrix_hsv[:,2] + 0.5 * (noise_vec_l - 0.5)
            color_matrix_hsv[:,1] = color_matrix_hsv[:,1] + 0.5 * (noise_vec_s - 0.5)
            color_matrix_hsv[color_matrix_hsv > 1] = 1
            color_matrix_hsv[color_matrix_hsv < 0] = 0

            color_matrix_rgb = np.array([ colorsys.hsv_to_rgb(*color_matrix_hsv[i]) for i in range(color_matrix_hsv.shape[0])])
            color_matrix_rgb = (bin_vec * color_matrix_rgb.T).T

            color_img = 255.0 * color_matrix_rgb[ws]
            color_img[color_img==0] = grey2rgb(region_image)[color_img==0]
            
            filename = os.path.join(clustermaps_folder,
                                    'clustermap%s_clusterid_%i_nb_%i.png' % (cluster_filename, 
                                                                             cluster_label, 
                                                                             len(labels)))
            print('write %s' % filename)
            skimage.io.imsave(filename, color_img.astype(np.uint8))

        return
    
    def make_intensity_density_plot(self, x, y, filename, x_name, y_name, 
                                    reduce_perc=None, 
                                    plot_grid=True, plot_lines=True, 
                                    additional_stats=None):
    
        if not reduce_perc is None:
            th_x = np.percentile(x, [reduce_perc])[0]
            th_y = np.percentile(y, [reduce_perc])[0]
            ind_x = x > th_x
            ind_y = y > th_y
            indices = np.multiply(ind_x, ind_y)
            x = x[indices]
            y = y[indices]
        
        perc = np.percentile(x, [0.1, 50, 99.9])
        xmin = perc[0]; xmax = perc[-1]; xmed = perc[1]
        xmean = np.mean(x)
        xstd = np.std(x)
        
        perc = np.percentile(y, [0.1, 50, 99.9])
        ymin = perc[0]; ymax = perc[-1]; ymed = perc[1]
        ymean = np.mean(y)
        ystd = np.std(y)
        
        # Calculate the point density
        xy = np.vstack([x,y])
        z = np.log(gaussian_kde(xy)(xy))
    
        # Sort the points by density, so that the densest points are plotted last
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
    
        fig = plt.figure(1)
        ax = plt.subplot(1,1,1)
    
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.title('Density plot: %s vs. %s' % (x_name, y_name))
        cax = ax.scatter(x, y, c=z, s=10, edgecolor='')
        cbar = plt.colorbar(cax)
        
        axis = [xmin - (xmax - xmin) * 0.05,
                xmax + (xmax - xmin) * 0.05,
                ymin - (ymax - ymin) * 0.05,
                ymax + (ymax - ymin) * 0.05]
        plt.axis(axis)
    
        #print '%s:\t%.2f, %.2f\t%.2f,%.2f' % (filename, xmin, xmax, ymin, ymax)
        #pdb.set_trace()
    
        #plt.colorbar()
        if plot_grid:
            plt.grid(b=True, which='major', linewidth=1)
        if plot_lines:
            added_lines = {}
            added_lines[0] = plt.axvline(#ymin=ymin, ymax=ymax,
                        x=xmean,
                        color=(0, 0.2, 0.5),
                        linewidth=2, ls='-')
            added_lines[1] = plt.axvline(#ymin=ymin, ymax=ymax,
                        x=xmean + 2 * xstd,
                        color=(0, 0.5, 1),
                        linewidth=2, ls='-')
    
            plt.axhline(#xmin=xmin, xmax=xmax,
                        y=ymean,
                        color=(0, 0.2, 0.5),
                        linewidth=2, ls='-')
            plt.axhline(#xmin=xmin, xmax=xmax,
                        y=ymean + 2 * ystd,
                        color=(0, 0.5, 1),
                        linewidth=2, ls='-')
    
            added_lines[2] = plt.axvline(#ymin=ymin, ymax=ymax,
                        x=xmed,
                        color=(0.8, 0.5, 0),
                        linewidth=2, ls='-')
            plt.axhline(#xmin=xmin, xmax=xmax,
                        y=ymed,
                        color=(0.8, 0.5, 0),
                        linewidth=2, ls='-')
            legend_lines = ['mean', 'mean + 2std', 'median']
            
            if not additional_stats is None:
                added_lines[3] = plt.axhline(#xmin=xmin, xmax=xmax,
                            y=additional_stats[y_name]['mean'],
                            color=(0.5, 0, 0.5),
                            linewidth=2, ls='-')
                added_lines[4] = plt.axhline(#xmin=xmin, xmax=xmax,
                            y=additional_stats[y_name]['mean'] + 
                            2 * additional_stats[y_name]['std'],
                            color=(0.8, 0, 0.4),
                            linewidth=2, ls='-')
                added_lines[5] = plt.axhline(#xmin=xmin, xmax=xmax,
                            y=additional_stats[y_name]['median'],
                            color=(1, 0, 0.2),
                            linewidth=2, ls='-')
    
                added_lines[3] = plt.axvline(#ymin=ymin, ymax=ymax,
                            x=additional_stats[x_name]['mean'],
                            color=(0.5, 0, 0.5),
                            linewidth=2, ls='-')
                added_lines[4] = plt.axvline(#ymin=ymin, ymax=ymax,
                            x=additional_stats[x_name]['mean'] + 
                            2 * additional_stats[x_name]['std'],
                            color=(0.8, 0, 0.4),
                            linewidth=2, ls='-')
                added_lines[5] = plt.axvline(#ymin=ymin, ymax=ymax,
                            x=additional_stats[x_name]['median'],
                            color=(1, 0, 0.2),
                            linewidth=2, ls='-')
                legend_lines += ['mean', 'mean + 2std', 'median']
            leg = ax.legend([added_lines[i] for i in range(len(added_lines))], 
                            legend_lines, 
                            loc='upper right')
    
    
        plt.savefig(filename)
        plt.close(1)
        return

    def make_pairwise_scatter_plots(self, X):
        out_folder = os.path.join(self.settings.plot_folder, 'density_plots')
        if not os.path.isdir(out_folder):
            os.makedirs(out_folder)
        for population in self.settings.populations[self.settings.dataset]:
            markers = self.settings.populations[self.settings.dataset][population]
            print(markers)
            for i in range(len(markers)-1):
                for j in range(i, len(markers)):
                    filename = os.path.join(out_folder, 'joint_density_%s_%s_%s.png' % (population, 
                                                                                        markers[i], 
                                                                                        markers[j]))
                    index_i = self.markers.index(markers[i])
                    index_j = self.markers.index(markers[j])
                    self.make_intensity_density_plot(X[:,index_i], 
                                                     X[:,index_j], 
                                                     filename, 
                                                     markers[i], 
                                                     markers[j], 
                                                     reduce_perc=None, 
                                                     plot_grid=True, 
                                                     plot_lines=True)
        return

if __name__ == '__main__':

    parser = argparse.ArgumentParser( \
        description=('Run post filter on Ilastik results in order to'
                     'get a smoother output and to assign grey levels'
                     'according to what is required for the rest.'))

    parser.add_argument('-s', '--settings_file', dest='settings_file', required=True,
                        type=str,
                        help='settings file for the analysis. Often the settings file is in the same folder.')
    parser.add_argument('-t', '--tissue_id', dest='tissue_id', required=False,
                        type=str, default=None, 
                        help='Tissue id (optional). If not specificied, the tissue id from the settings file is taken.')
    parser.add_argument('--pca', dest='pca', required=False,
                        action='store_true',
                        help='Make PCA.')
    
    parser.add_argument('--ward', dest='ward', required=False,
                        action='store_true',
                        help='Perform hierarchical clustering (ward).')
    parser.add_argument('--nb_clusters', dest='nb_clusters', required=False,
                        type=int, default=40, 
                        help='Number of clusters')
    parser.add_argument('--downsample', dest='downsample', required=False,
                        type=int, default=None, 
                        help='Downsample: number of samples to draw randomly from the cells.')
    parser.add_argument('--normalize', dest='normalize', required=False,
                        type=str, default='percentile', 
                        help='Normalization to be applied to the average intensity measured on the cell area.'
                        'Possible values are: minmax, percentile, sigmoid')

    args = parser.parse_args()
    
    ca = ClusterAnalysis(args.settings_file, tissue_id=args.tissue_id)
    if args.pca:
        print(' *** Perform principal component analysis ***')
        pc = ca.pca()
        
    if args.ward:
        print(' *** Perform hierarchical clustering ***')
        ca.nb_clusters = args.nb_clusters
        X = ca.get_data()
        Xnorm = ca.normalize(X, args.normalize)
        if not args.downsample is None:
            Xs, indices = ca.subsample(Xnorm, args.downsample)
        else:
            Xs = Xnorm
        filename_ext='_normalization_%s' % args.normalize 
        print('starting the hierarchical clustering ... ')
        full_res = ca.hierarchical_clustering(Xs, filename_ext=filename_ext, indices=indices)
        
        print('exporting the cluster maps')
        ca.export_cell_cluster_maps(cluster_filename = filename_ext)
        print('DONE')
        
        
        
        