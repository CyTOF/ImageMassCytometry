from sequence_importer import RGB_Generator

rg = RGB_Generator(channel_order=['CD3', 'CD19'])
rg.batch_processing('/Users/twalter/data/Elodie/Ilastik/channels',
                    '/Users/twalter/data/Elodie/Ilastik/data/rgb',
                    tissue_ids = ['Tonsil_D1', 'Tonsil_D2'])
