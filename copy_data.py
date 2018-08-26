import os
import shutil

sample_id = 'Tonsil_D2'

folder = '/Users/twalter/data/Elodie/data/Core D2'
out_folder =  os.path.join('/Users/twalter/data/Elodie/new_data', sample_id)
rgb_channel_folder = os.path.join('/Users/twalter/data/Elodie/Ilastik/channels', sample_id)

# make the folders
for temp_folder in [out_folder, rgb_channel_folder]: 
    if not os.path.isdir(temp_folder):
        print('making %s' % temp_folder)
        os.mkdir(temp_folder)

channel_metal_dict = {
'147Sm':    'CD303(BDCA2)',
'151Eu':    'CD185(CXCR5)',
'173Yb':    'CD1c-biotin-NA',
'175Lu':    'CD279(PD-1)',
'176Yb':    'CD56(NCAM)',
'191Ir':    'DNA1',
'193Ir':    'DNA2',
'158Gd':    'E-Cadherin',
}

filenames = filter(lambda x: x[0] != '.', os.listdir(folder))

for filename in filenames:
    full_filename = os.path.join(folder, filename)
    info = os.path.splitext(filename)[0].split('_')

    metal = info[0]
    if metal in channel_metal_dict:
        channel = channel_metal_dict[metal]
    else:
        temp = info[1].split('.')[0].split('-')
        if len(temp) == 1:
            channel = 'empty'
        else:
            channel = '-'.join(temp[1:])
    print('channel: %s\tmetal: %s' % (channel, metal))
    new_filename = os.path.join(out_folder, 'Tonsil_D2_%s_%s.tif' % (metal, channel))
    shutil.copy(full_filename, new_filename)
    
    # copied for preparation of RGB image (for illastik)
    if channel in ['CD3', 'CD19']:
        new_filename = os.path.join(rgb_channel_folder, 'Tonsil_D2_%s_%s.tif' % (metal, channel))
        shutil.copy(full_filename, new_filename)
    