import os
import sys
import argparse
import pdb 

import pickle
import numpy as np

if 'CLUSTER_ENV' in os.environ:
    import matplotlib
    matplotlib.use('Agg')
else:
    import matplotlib.pyplot as plt

# seaborn for plotting
import seaborn as sns

# project imports
from settings import Settings, overwrite_settings
from sequence_importer import SequenceImporter
from fissure_detection import FissureDetection
from cell_detection import SpotDetector, CellDetection
from process_ilastik import Ilastik
from visualization import Overlays, make_random_colors

from scipy.ndimage.morphology import distance_transform_edt
from skimage.measure import label, regionprops

from scipy.cluster import hierarchy


class DistanceAnalysis(object):
    
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
        self.cell_detector = CellDetection(settings=self.settings)
        self.ilastik = Ilastik(settings=self.settings)
                
        self.markers = self.settings.cluster_markers[self.settings.dataset]
        self.nb_clusters = 60

    def read_fused_cluster_results(self, filename_ext='_normalization_percentile_metric_euclidean_method_ward'):
        
        filename = os.path.join(self.cluster_folder, 'cluster_assignment%s.pickle' % filename_ext)
        print('loading clustering from %s' % filename)
        fp = open(filename, 'rb')
        full_res = pickle.load(fp)
        fp.close()

        # cluster_assignment: for each cluster, we get the list of cell labels.
        cluster_assignment = full_res['res']

        fused_clusters = {}
        fusion_info = self.settings.cluster_fusion[self.settings.dataset]
        cluster_names = list(fusion_info.keys())
        for k, population_name in enumerate(cluster_names):
            fused_clusters[k] = np.hstack([cluster_assignment[i][0] for i in fusion_info[population_name]])
        col_pal = sns.color_palette("husl", len(fusion_info))

        return fused_clusters, cluster_names, col_pal 


    def get_coordinates(self, force=False):
        
        if not force:
            print('new calculation not forced ... ')
            
        print('Reading data ... ')
        
        # get the individual cells.
        ws = self.cell_detector.get_image(False)
        
        # number of samples
        N = ws.max()
        
        props = regionprops(ws)
        coordinates = np.array([props[k]['centroid'] for k in range(len(props))])
                    
        return coordinates

    def plot_distance_histogram(self, vec, img_values, filename,
                                region_name='B-cell region', 
                                subset_name='1'):
        
        plot_folder = os.path.join(self.settings.plot_folder, 'distance_histograms')
        if not os.path.isdir(plot_folder):
            os.makedirs(plot_folder)
        full_filename = os.path.join(plot_folder, filename)
        
        #hist, bins = np.histogram(vec, bins=20, normed=1)
        X = [vec, img_values]
        if len(vec) > 0:
            mean_val = np.mean(vec)
        else:
            mean_val = 0
        if len(vec) > 1:
            std = np.std(vec)
        else: 
            std = 0

        #width = 0.7 * (bins[1] - bins[0])
        #center = (bins[:-1] + bins[1:]) / 2
        
        fig = plt.figure()

        #plt.bar(center, hist, align='center', width=width)
        colors = ['red', 'blue']
        labels = ['%s(%i)' % (subset_name, len(vec)), 'CSR(%i)' % len(img_values)]
        plt.hist(X, density=True, bins=32, histtype='bar', color=colors, label=labels)
        plt.title('distances: %s (%.2f +/- %.2f)' % (region_name, 
                                                     mean_val, 
                                                     mean_val + 2 * std))
        plt.legend(prop={'size': 10})
        plt.xlabel('distance in pixels')
        plt.ylabel('frequency')
        fig.savefig(full_filename)
        plt.close()
        
        fig = plt.figure()
        flierprops = dict(marker='o', markerfacecolor='red', markersize=6,
                          linestyle='none')
        plt.boxplot(X, sym='o', flierprops=flierprops, labels=labels)
        plt.title('Distance boxplots: %s in %s' % (subset_name, region_name))

        full_filename = os.path.join(plot_folder, 'boxplot_%s' % filename)
        fig.savefig(full_filename)
        
        plt.close()
        return

    def plot_distance_boxplots_joint(self, res, colors):
        plot_folder = os.path.join(self.settings.plot_folder, 'distance_histograms')
        #res[cluster_name]['bgdist'] = distances
        self.region_names = {'bdist': 'B-region',
                             'tdist': 'T-region',
                             'bgdist': 'Crypt / Background'}
        colors = colors + [(.80, .80, .80)]
        for region in self.region_names:
            cluster_names = list(res.keys())
            X = [res[cluster_name][region] for cluster_name in cluster_names] + [res[cluster_names[0]][region+'_h0']]
            fig = plt.figure()
            ax = plt.subplot(1,1,1)
            #flierprops = dict(marker='o', markerfacecolor='red', markersize=6,
            #                  linestyle='none')
            #box = plt.boxplot(X, sym='o', flierprops=flierprops, labels=cluster_names+["CSR"], patch_artist=True, whis=[5,95])
            box = plt.boxplot(X, sym='', labels=cluster_names+["CSR"], patch_artist=True, whis=[5,95])
            plt.title('Distance boxplots: %s' % self.region_names[region])
            xlabels = ['%s(%i)' % (x, len(res[x][region])) for x in cluster_names] + ["CSR"]
            ax.set_xticklabels(xlabels, rotation=40, ha='right')
            plt.xticks(rotation=45, fontsize=8)
            #plt.setp(xtickNames, rotation=45, fontsize=8)
            for patch, color in zip(box['boxes'], colors):
                patch.set_facecolor(color)

            plt.tight_layout()

            full_filename = os.path.join(plot_folder, 'all_boxplot_%s.pdf' % region)
            fig.savefig(full_filename)
            plt.close('all')

            # violin plot
            fig = plt.figure(figsize=(16, 8))
            ax = sns.violinplot(data=X, palette=colors)
            ax.set_xticklabels(xlabels, rotation=40, ha='right')
            plt.title('Distance boxplots: %s' % self.region_names[region])
            plt.tight_layout()
            full_filename = os.path.join(plot_folder, 'all_violin_%s.pdf' % region)
            fig.savefig(full_filename)
            plt.close('all')
        return

    def analyze_population_distances(self, write_coordinates=False, plot_individual=True):
        background, b_region, t_region = self.ilastik.read_region_images()

        b_dist = distance_transform_edt(b_region)
        t_dist = distance_transform_edt(t_region)
        bg_dist = distance_transform_edt(background)
        
        fused_clusters, cluster_names, col_pal = self.read_fused_cluster_results()
        coordinates = self.get_coordinates()
        coordinates = np.floor(coordinates).astype(np.uint16)
        
        nb_cells = coordinates.shape[0]

        res = {}
        cluster_stats = {}
        for k, cluster_name in enumerate(cluster_names):

            # coordinates for this cluster
            cluster_coordinates = coordinates[fused_clusters[k]]
            nb_cluster_cells = cluster_coordinates.shape[0]

            if write_coordinates: 
                coord_out_folder = os.path.join(self.settings.output_folder,
                                            'coordinates')
                if not os.path.isdir(coord_out_folder):
                    os.makedirs(coord_out_folder)

                coord_filename = os.path.join(coord_out_folder, 
                                              'coordinates__%s__population__%s.txt' % 
                                              (self.settings.dataset, cluster_name))

                fp = open(coord_filename, 'w')
                for row, col in cluster_coordinates:
                    fp.write('%i, %i\n' % (row, col))
                fp.close()

                coord_filename = os.path.join(coord_out_folder, 
                                              'coordinates__%s__population__%s.pickle' % 
                                              (self.settings.dataset, cluster_name))
        
                fp = open(coord_filename, 'w')
                pickle.dump(cluster_coordinates, fp)
                fp.close()

            print()
            print(' *************************** ')
            print('DISTANCE ANALYSIS for %s' % cluster_name)

            rows = cluster_coordinates[:,0]
            cols = cluster_coordinates[:,1]

            distances = b_dist[rows, cols]
            distances = distances[distances > 0]
            h0_distances = b_dist[b_dist>0]
            res[cluster_name] = {'bdist': distances, 'bdist_h0': h0_distances}
            cluster_stats[cluster_name] = {'bdist':  len(distances) / np.float(nb_cluster_cells),
                                           'bdist_nb': len(distances),
                                           'total_nb': len(h0_distances)}
            print('\tpercentage of cells in B-region: %.3f' % (len(distances) / np.float(nb_cluster_cells)))
            if plot_individual:
                self.plot_distance_histogram(distances, h0_distances,
                                             filename='%s_population_%s_B_cell_region_distances.pdf' % (self.settings.dataset, 
                                                                                                        cluster_name), 
                                             region_name='B-region',
                                             subset_name=cluster_name)

            distances = t_dist[rows, cols]
            distances = distances[distances > 0]
            h0_distances = t_dist[t_dist>0]
            res[cluster_name]['tdist'] = distances
            res[cluster_name]['tdist_h0'] = h0_distances
            cluster_stats[cluster_name]['tdist'] = len(distances) / np.float(nb_cluster_cells)
            cluster_stats[cluster_name]['tdist_nb'] = len(distances)
            print('\tpercentage of cells in T-region: %.3f' % (len(distances) / np.float(nb_cluster_cells)))
            if plot_individual:
                self.plot_distance_histogram(distances[distances>0], h0_distances,
                                             filename='%s_population_%s_T_cell_region_distances.pdf' % (self.settings.dataset, 
                                                                                                        cluster_name), 
                                             region_name='T-region',
                                             subset_name=cluster_name)

            distances = bg_dist[rows, cols]
            distances = distances[distances > 0]
            h0_distances = bg_dist[bg_dist>0]
            res[cluster_name]['bgdist'] = distances
            res[cluster_name]['bgdist_h0'] = h0_distances
            cluster_stats[cluster_name]['bgdist'] = len(distances) / np.float(nb_cluster_cells)
            cluster_stats[cluster_name]['bgdist_nb'] = len(distances)
            print('\tpercentage of cells in background/crypt: %.3f' % (len(distances) / np.float(nb_cluster_cells)))
            if plot_individual:
                self.plot_distance_histogram(distances, h0_distances,
                                             filename='%s_population_%s_crypt_region_distances.pdf' % (self.settings.dataset, 
                                                                                                       cluster_name), 
                                             region_name='Crypt/Background',
                                             subset_name=cluster_name)
            max_img = np.maximum(np.maximum(bg_dist, b_dist), t_dist)
            distances = max_img[rows, cols]
            distances = distances[distances==0]
            print('\ton the border: %.3f' % (len(distances) / np.float(nb_cluster_cells)))

        self.plot_distance_boxplots_joint(res, col_pal)

        filename = os.path.join(self.settings.plot_folder, 'distance_histograms', 'region_percentages.txt')
        fp = open(filename, 'w')
        fp.write('\t'.join(['Cell Type', 'B-region', 'T-region', 'Crypt / Background']) + '\n')
        for k, cluster_name in enumerate(cluster_names):
            fp.write('\t'.join([cluster_name] + ['%.3f' % cluster_stats[cluster_name][region] for region in ['bdist', 'tdist', 'bgdist']]) + '\n')
        fp.close()

        # output_folder = os.path.join(self.settings.output_folder, 'distance_maps')
        # if not os.path.isdir(output_folder):
        #     os.makedirs(output_folder)

        # color = (255, 0, 0)
        # region_image = np.zeros(b_region.shape, dtype=np.uint8)
        # region_image[b_region>0] = 100
        # region_image[t_region>0] = 200
        # filename_addon = '-'.join(channel_names.values())
        # filename = os.path.join(output_folder, 'region_population_%s_%s_overlay.png' % (population, filename_addon))
        # overlay_img = self.sp.make_overlay_img(region_image, coordinates, color,
        #                                        perimeter=False)
        # skimage.io.imsave(filename, overlay_img)

        
        # distances = bg_dist[y, x]
        # nb_distances = max(len(distances), 1)
        # print('percentage of cells in background: %.3f' % (float(len(distances[distances>0])) / nb_distances))
        # res['perc_BG'] = float(len(distances[distances>0])) / nb_distances
                
        return

if __name__ == '__main__':

    parser = argparse.ArgumentParser( \
        description=('Distance Analysis'))

    parser.add_argument('-s', '--settings_file', dest='settings_file', required=True,
                        type=str,
                        help='settings file for the analysis. Often the settings file is in the same folder.')
    parser.add_argument('-t', '--tissue_id', dest='tissue_id', required=False,
                        type=str, default=None, 
                        help='Tissue id (optional). If not specificied, the tissue id from the settings file is taken.')

    args = parser.parse_args()

    da = DistanceAnalysis(settings_filename=args.settings_file, tissue_id=args.tissue_id)
    da.analyze_population_distances()

