import os, sys, time, re

def overwrite_settings(settings, dataset):
    base_folder = settings.base_folder 
    
    settings.dataset = dataset
    settings.input_folder = os.path.join(base_folder, 'new_data', dataset)
    settings.output_folder = os.path.join(base_folder, 'results', dataset)
    settings.debug_folder = os.path.join(base_folder, 'debug')
    settings.plot_folder = os.path.join(settings.output_folder, 'plots')

    # ilastik folder settings
    settings.ilastik_input_folder = os.path.join(base_folder, 'Ilastik', 'channels', dataset)
    settings.ilastik_input_folder = os.path.join(base_folder, 'Ilastik', 'channels', dataset)
    settings.ilastik_input_rgb_filename = os.path.join(settings.ilastik_input_rgb_folder, 
                                                       'rgb_%s.tif' % dataset)
    settings.ilastik_folder = os.path.join(base_folder, 'Ilastik', 'data','rgb')

    settings.ilastik_filename = os.path.join(settings.ilastik_folder, 
                                             'rgb_%s_Simple Segmentation.png' % dataset)
    settings.ilastik_backup_filename = os.path.join(settings.ilastik_folder, 
                                           'rgb_%s_Simple Segmentation_backup.png' % dataset)
    settings.downsample_image = os.path.join(settings.ilastik_folder, 
                                             'rgb_%s.tif' % dataset)
    
    settings.makefolders = [settings.output_folder, 
                            settings.debug_folder, 
                            settings.plot_folder]
    return settings

class Settings(object):
    """
    Simple container to hold all settings from an external python file as own
    class attributes.
    Should be made singleton.
    """

    def __init__(self, filename=None, dctGlobals=None):
        self.strFilename = filename
        if not filename is None:
            self.load(filename, dctGlobals)


    def load(self, filename, dctGlobals=None):
        self.strFilename = filename
        if dctGlobals is None:
            dctGlobals = globals()

        #self.settings_dir = os.path.abspath(os.path.dirname(self.strFilename))
        #execfile(self.strFilename, dctGlobals, self.__dict__)
        exec(open(self.strFilename).read(), dctGlobals, self.__dict__)
    
    def update(self, dctNew, bExcludeNone=False):
        for strKey, oValue in dctNew.items():
            if not bExcludeNone or not oValue is None:
                self.__dict__[strKey] = oValue

    def __getattr__(self, strName):
        if strName in self.__dict__:
            return self.__dict__[strName]
        else:
            raise ValueError("Parameter '%s' not found in settings file '%s'." %
                             (strName, self.strFilename))

    def __call__(self, strName):
        return getattr(self, strName)

    def all(self, copy=True):
        return self.__dict__.copy()



