import os, sys, time, re

def overwrite_settings(settings, dataset):
    base_folder = settings.base_folder 
    
    old_dataset = settings.dataset
    settings.dataset = dataset
    settings.input_folder = settings.input_folder.replace(old_dataset, dataset)
    settings.output_folder = settings.output_folder.replace(old_dataset, dataset)
    settings.debug_folder = settings.debug_folder.replace(old_dataset, dataset)
    settings.plot_folder = settings.plot_folder.replace(old_dataset, dataset)

    # ilastik folder settings
    settings.ilastik_input_rgb_folder = settings.ilastik_input_rgb_folder.replace(old_dataset, dataset)   
    settings.ilastik_input_rgb_filename = settings.ilastik_input_rgb_filename.replace(old_dataset, dataset)
    settings.ilastik_folder = settings.ilastik_folder.replace(old_dataset, dataset)
    settings.ilastik_filename = settings.ilastik_filename.replace(old_dataset, dataset)

    settings.makefolders = [settings.output_folder, 
                            settings.debug_folder, 
                            settings.plot_folder, 
                            settings.ilastik_folder,
                            settings.ilastik_input_rgb_folder
                            ]
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



