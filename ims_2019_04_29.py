# the tissue
dataset = 'Tonsil_D2'

marker_list = ['CD19', 'CD3']

# folder settings
base_folder = '/Users/twalter/data/Elodie/project_data'
input_folder = os.path.join(base_folder, dataset, 'in_data')
output_folder = os.path.join(base_folder, dataset, 'results')
debug_folder = os.path.join(base_folder, dataset, 'debug')
plot_folder = os.path.join(output_folder, 'plots')

# ilastik folder settings
ilastik_input_rgb_folder = os.path.join(base_folder, dataset, 'Ilastik', 'rgb')
ilastik_input_rgb_filename = os.path.join(ilastik_input_rgb_folder, 'rgb_%s.tif' % dataset)
ilastik_folder = os.path.join(base_folder, dataset, 'Ilastik', 'result')
ilastik_filename = os.path.join(ilastik_folder, 'rgb_%s_Simple Segmentation.png' % dataset)

makefolders = [output_folder, debug_folder, plot_folder, ilastik_folder]
debug = True

# parameters for cell detection
dna_spot_detection = {#'eps': 0.016,
                  'eps': 0.004,
                  'h': 0.001,
                  'sigma': 2.3,
                  'gauss_threshold': 0.05,
                  'normalization_low': 0.1,
                  'normalization_high': 5.0,
                  'area_threshold': 5}

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
                  'CD68', 'CD206', 'CD370', 'CD141', 'CD123', 'CD1c-biotin-NA'],
    'Tonsil_120628_D1-2A_4MAY2017_1': ['E-Cadherin', 'AlphaSMA', 'CD45', 'IL21', 'CXCL13',
                                       'PD1', 'CD3', 'CD45RA', 'Bcl6', 'CD19', 'CD11c',
                                       'CD14', 'Bcl2', 'CD8a', 'FoxP3', 'KI67'],
    'Tonsil_120828_D18-2A_5MAY2017_1': ['E-Cadherin', 'AlphaSMA', 'CD45', 'IL21', 'CXCL13',
                                        'PD1', 'CD3', 'CD45RA', 'Bcl6', 'CD19', 'CD11c',
                                        'CD14', 'Bcl2', 'CD8a', 'FoxP3', 'KI67'],
}
