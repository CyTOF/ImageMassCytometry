import os
import shutil
import new

#folder = '/Users/twalter/data/Elodie/raw_data/M2_tiff'
#out_folder = '/Users/twalter/data/Elodie/new_data/Tonsil_ID120828D18_M2_2'
sample_id = 'Tonsil_D2'

folder = '/Users/twalter/data/Elodie/data/Core D2'
out_folder =  os.path.join('/Users/twalter/data/Elodie/new_data', sample_id)
rgb_channel_folder = os.path.join('/Users/twalter/data/Elodie/Ilastik/channels', sample_id)

for temp_folder in [out_folder, rgb_channel_folder]: 
    if not os.path.isdir(temp_folder):
        print 'making %s' % temp_folder
        os.mkdir(temp_folder)
        
#147Sm_147Sm-CD303(BDCA2).ome
filenames = filter(lambda x: x[0] != '.', os.listdir(folder))

for filename in filenames:
    full_filename = os.path.join(folder, filename)
    info = os.path.splitext(filename)[0].split('_')
    #channel = info[0]
    #metal = info[1]
    metal = info[0]
    temp = info[1].split('.')[0].split('-')
    if len(temp) == 1:
        channel = 'empty'
    else:
        channel = '-'.join(temp[1:])
    print channel, metal 
    new_filename = os.path.join(out_folder, 'Tonsil_D2_%s_%s.tif' % (metal, channel))
    shutil.copy(full_filename, new_filename)
    
    if channel in ['CD3', 'CD19']:
        new_filename = os.path.join(rgb_channel_folder, 'Tonsil_D2_%s_%s.tif' % (metal, channel))
        shutil.copy(full_filename, new_filename)
    