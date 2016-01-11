class Dataset(object):
    '''
    A class that holds data, the target locations, and the target names.
    '''
    def __init__(self, data, target, target_names):
        self.data         = data
        self.target       = target
        self.target_names = target_names