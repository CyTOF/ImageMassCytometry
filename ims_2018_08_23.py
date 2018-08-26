
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
downsample_image = os.path.join(ilastik_folder, 'rgb_%s.tif' % dataset)

makefolders = [output_folder, debug_folder, plot_folder]
debug = True

prefilter_gaussian_bandwidth = 3.0
prefilter_diameter_closing = 20
prefilter_area_closing = 400
prefilter_median_size = 6
thresh_alpha = 1.2

prefilter_asf_sizes = [1, 2, 4, 8]

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
        'CD3': 5.0,
        'PD1': 0.909,
        'CXCR5': 0.65,
        'CD11c': 1.523,
        'CD370': 0.56,
        'CD206': 0.339,
        'CD1c': 0.376,
        'CD45': 13.0,
        'Bcl6': 0.30,
        'CD14': 1.0,
        'CD141': 1.0,
        'CD123': 1.0,
        'CD303': 1.0,
        'HLADR': 1.0
        },
    'Tonsil_D2': {
        'CD3': 5.0,
        'PD1': 0.909,
        'CXCR5': 0.65,
        'CD11c': 1.523,
        'CD370': 0.56,
        'CD206': 0.339,
        'CD1c': 0.376,
        'CD45': 13.0,
        'Bcl6': 0.30,
        'CD14': 1.0,
        'CD141': 1.0,
        'CD123': 1.0,
        'CD303': 1.0,
        'HLADR': 1.0
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
