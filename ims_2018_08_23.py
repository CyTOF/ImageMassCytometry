# datasets to be used:
# Tonsil_120628_D1-2A_4MAY2017_1
# Tonsil_120828_D18-2A_5MAY2017_1

# the tissue
dataset = 'Tonsil_D1'

marker_list = ['CD19', 'CD3']

# folder settings
base_folder = '/Users/twalter/data/Elodie/'
input_folder = os.path.join(base_folder, 'new_data', dataset)
output_folder = os.path.join(base_folder, 'results', dataset)
debug_folder = os.path.join(base_folder, 'debug')
plot_folder = os.path.join(output_folder, 'plots')

# ilastik folder settings
ilastik_input_folder = os.path.join(base_folder, 'Ilastik', 'channels', dataset)
ilastik_input_rgb_folder = os.path.join(base_folder, 'Ilastik', 'data', 'rgb')
ilastik_input_rgb_filename = os.path.join(ilastik_input_rgb_folder, 'rgb_%s.tif' % dataset)

ilastik_folder = os.path.join(base_folder, 'Ilastik', 'data','rgb')
ilastik_filename = os.path.join(ilastik_folder, 'rgb_%s_Simple Segmentation.png' % dataset)
ilastik_backup_filename = os.path.join(ilastik_folder, 'rgb_%s_Simple Segmentation_backup.png' % dataset)
downsample_image = os.path.join(ilastik_folder, 'rgb_%s.tif' % dataset)

makefolders = [output_folder, debug_folder, plot_folder]
debug = True

prefilter_gaussian_bandwidth = 3.0
prefilter_diameter_closing = 20
prefilter_area_closing = 400
prefilter_median_size = 6
thresh_alpha = 1.2

prefilter_asf_sizes = [1, 2, 4, 8]


cluster_fusion = {
'Tonsil_D1': {'Macrophage' : [4, 9, 14, 34],
              'Crypt_epithelial_cells' : [1, 5, 52, 36, 37, 44, 50, 52],
              'B_cells' : [12, 29, 31, 33, 40, 43],
              'B_cells_Tfh' : [18, 41], 
              'B_cells_FDC': [10, 11, 16, 17],
              'Tfh' : [13, 19], 
              'T_cells_other' : [20, 21, 26, 30, 35],
              'DC1' : [22, 48],
              'pDC' : [23], 
              'Blood_vessels' : [24, 38, 39, 47, 28],
              'CD14-_CD206+_cells' : [8],
              },
'Tonsil_D2':  {'Macrophage' : [30, 41, 22],
              'Crypt_epithelial_cells' : [6, 16, 33, 34 ],
              'B_cells' : [8, 12, 13, 14, 15, 29],
              'B_cells_Tfh' : [11], 
              'B_cells_FDC': [0, 45],
              'Tfh' : [10, 19, 32], 
              'T_cells_other' : [1, 17, 31, 28],
              'DC1' : [35],
              'pDC' : [42, 43], 
              'Blood_vessels' : [20, 23, 25, 26, 37, 38, 44, 59, 18],
              'CD14-_CD206+_cells' : [2],
              },
'Tonsil_120828_D18-2A_5MAY2017_1': {'Macrophage' : [35, 36, 42],
                                    'Crypt_epithelial_cells' : [15, 22, 27, 54, 55],
                                    'B_cells' : [13, 18, 21, 28, 31, 32, 39],
                                    'B_cells_Tfh' : [26],                                     
                                    'Tfh' : [38, 45, 51, 52], 
                                    'T_cells_other' : [1, 3, 4, 5, 6, 7, 8, 10, 12, 14, 19, 20, 23, 24],
                                    'Blood_vessels' : [2, 16, 25, 30, 37, 43, 44],
                                    },
}


cluster_markers = {
    'Tonsil_D1': ['E-Cadherin', 'aSMA', 'CD45', 'IL-21',                   
                  'CXCL13', 'CD279(PD-1)', 'CD3', 'CD45RA',                   
                  'Bcl-6', 'CD19', 'HLA-DR', 'CD11c', 'CD14',
                  'CD68', 'CD206', 'CD370', 'CD141', 'CD123'],
    'Tonsil_D2': ['E-Cadherin', 'aSMA', 'CD45', 'IL-21',                   
                  'CXCL13', 'CD279(PD-1)', 'CD3', 'CD45RA',                   
                  'Bcl-6', 'CD19', 'HLA-DR', 'CD11c', 'CD14',
                  'CD68', 'CD206', 'CD370', 'CD141', 'CD123'],
    'Tonsil_120628_D1-2A_4MAY2017_1': ['E-Cadherin', 'AlphaSMA', 'CD45', 'IL21', 'CXCL13',
                                       'PD1', 'CD3', 'CD45RA', 'Bcl6', 'CD19', 'CD11c',
                                       'CD14', 'Bcl2', 'CD8a', 'FoxP3', 'KI67'],
    'Tonsil_120828_D18-2A_5MAY2017_1': ['E-Cadherin', 'AlphaSMA', 'CD45', 'IL21', 'CXCL13',
                                        'PD1', 'CD3', 'CD45RA', 'Bcl6', 'CD19', 'CD11c',
                                        'CD14', 'Bcl2', 'CD8a', 'FoxP3', 'KI67'],
}


# This is no longer used
populations = {
    'Tonsil_D1': {
        'Tfh' : ['Bcl-6', 'CD279(PD-1)', 'CD3'],
        'Macrophage' : ['CD45', 'CD14', 'CD11c'],
        'DC1' : ['CD370', 'CD141',  'HLA-DR'],
        'DC2' : ['CD11c',  'CD1c-biotin-NA',  'HLA-DR'],
        'pDC' : ['CD123', 'CD303(BDCA2)',  'HLA-DR'],
        },
    'Tonsil_D2': {
        'Tfh' : ['Bcl-6', 'CD279(PD-1)', 'CD3'],
        'Macrophage' : ['CD45', 'CD14', 'CD11c'],
        'DC1' : ['CD370', 'CD141',  'HLA-DR'],
        'DC2' : ['CD11c',  'CD1c-biotin-NA',  'HLA-DR'],
        'pDC' : ['CD123', 'CD303(BDCA2)',  'HLA-DR'],
        },
    }

# This is no longer used
optimal_thresholds = {
    'Tonsil_ID120828D18_M2_2': {
        'CD3': 1.202,
        'PD1': 0.4,
        'CXCR5': 0.3,
        'CD11c': 1.0, # could still be lowered
        'CD1c': 0.3,
        'CD14': 1.0, # difficult to tell
        'CD370': 0.3,
        'CD45': 1.6, # imnipresent
        'Bcl6': 0.5, 
        'CD206': 0.507},
    'Tonsil_ID120828D18_M1_2': {
        'CD3': 2.5,
        'PD1': 0.04,
        'CXCR5': 0.03,
        'CD11c': 1.0,
        'CD1c': 0.03,
        'CD14': 0.03,
        'CD370': 0.03,
        'CD45': 1.5,
        'Bcl6': 0.6},
    'Tonsil_120628_D1-2A_4MAY2017_1':{
        'CD3': 5.0,
        'PD1': 0.49,
        'CXCR5': 0.461,
        'CD11c': 0.62,
        'CD1c': 0.62,
        'CD14': 0.89,
        'CD45': 18.2,
        'Bcl6': 0.386},
    'Tonsil_120628_D1-2A_4MAY2017_2': {
        'CD3': 5.0,
        'PD1': 0.909,
        'CXCR5': 0.65,
        'CD11c': 1.523,
        'CD370': 0.56,
        'CD206': 0.339,
        'CD1c': 0.376,
        'CD45': 13.0,
        'Bcl6': 0.30},
    'Tonsil_120828_D18-2A_5MAY2017_1': {
        'CD3': 5.0,
        'PD1': 0.89,
        'CXCR5': 0.461,
        'CD11c': 1.523,
        'CD1c': 0.453,
        'CD14': 1.4,
        'CD45': 15.0,
        'Bcl6': 0.416},
    'Tonsil_120828_D18-2A_5MAY2017_2': {
        'CD3': 5.0,
        'PD1': 0.909,
        'CXCR5': 0.65,
        'CD11c': 1.523,
        'CD370': 0.56,
        'CD206': 0.339,
        'CD1c': 0.376,
        'CD45': 13.0,
        'Bcl6': 0.30},
    'Tonsil_D1': {
        'CD3': 11.0,
        'CD279(PD-1)': 2.15,
        'CXCR5': 0.65,
        'CD11c': 6.12,
        'CD370': 1.2,
        'CD206': 0.339,
        'CD1c-biotin-NA': 0.6,
        'CD45': 8.0,
        'Bcl-6': 0.412,
        'CD14': 3.0,
        'CD141': 0.8,
        'CD123': 1.0,
        'CD303(BDCA2)': 0.24,
        'HLA-DR': 50.0
        },
    'Tonsil_D2': {
        'CD3': 11.0,
        'CD279(PD-1)': 2.15,
        'CXCR5': 0.65,
        'CD11c': 6.12,
        'CD370': 1.2,
        'CD206': 0.339,
        'CD1c-biotin-NA': 0.6,
        'CD45': 8.0,
        'Bcl-6': 0.412,
        'CD14': 3.0,
        'CD141': 0.8,
        'CD123': 1.0,
        'CD303(BDCA2)': 0.24,
        'HLA-DR': 50.0
        },
}

spot_detection = {#'eps': 0.016,
                  'eps': 0.02,
                  'h': 0.006,
                  'sigma': 2.3,
                  'gauss_threshold': 10,
                  'normalization_low': 0.1,
                  'normalization_high': 8.0,
                  'area_threshold': 5}

# dna_spot_detection = {#'eps': 0.016,
#                   'eps': 0.008,
#                   'h': 0.003,
#                   'sigma': 2.1,
#                   'gauss_threshold': 0.08,
#                   'normalization_low': 0.1,
#                   'normalization_high': 5.0,
#                   'area_threshold': 5}

dna_spot_detection = {#'eps': 0.016,
                  'eps': 0.004,
                  'h': 0.001,
                  'sigma': 2.3,
                  'gauss_threshold': 0.05,
                  'normalization_low': 0.1,
                  'normalization_high': 5.0,
                  'area_threshold': 5}

sample_spot_detection = {#'eps': 0.016,
                  'eps': 0.02,
                  'h': 0.006,
                  'sigma': 2.3,
                  'gauss_threshold': 0.1,
                  'normalization_low': 0.1,
                  'normalization_high': 5.0,
                  'area_threshold': 5}
