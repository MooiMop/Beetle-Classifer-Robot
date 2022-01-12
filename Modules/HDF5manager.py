import h5py
import re
import datetime
import numpy as np

try:
    import Modules.tools as tools
except ModuleNotFoundError:
    import tools


class HDF5():

    def __init__(self, filename, mode, user=None, date=True, metadata={},
                 verbose=True):

        if'.hdf5' not in filename:
            filename += '.hdf5'

        if date:
            # Check if filename already start with date
            try:
                # if these lines doesn't raise an error, the filename starts
                # with a date
                test = filename[0:10]
                datetime.datetime.strptime(test, '%Y.%m.%d')
            except ValueError:
                filename = tools.get_time(format='date') + ' ' + filename

        if mode == 'create':
            try:
                self.file = h5py.File(filename, 'x')
                if user is not None:
                    HDF5.write_metadata(self.file, {'user': user})
                if len(metadata) > 0:
                    HDF5.write_metadata(self.file, metadata)
                if verbose:
                    tools.logprint(f'Created file {filename}', 'blue')
            except (FileExistsError, OSError):
                tools.logprint(f'File {filename} already exists. Opening '
                               'instead.', 'yellow')
                mode = 'open'

        if mode == 'open':
            self.file = h5py.File(filename, 'r+')
            if verbose:
                tools.logprint('Opened file ' + tools.bcolors.blue(filename))

        elif mode != 'create':
            raise ValueError(f'mode parameter should be "open" or "create".')

        self.name = filename

    def group(self, name, parent=None, metadata={}):
        object = self._get_parent(parent)
        try:
            group = object[name]
        except KeyError:
            group = object.create_group(name)
            HDF5.write_metadata(group, metadata)
        tools.logprint('Created HDF5 group ' + tools.bcolors.blue(name))
        return group

    def create_dataset(self, name, data, parent=None, metadata={},
                       overwrite=False, verbose=True):
        object = self._get_parent(parent)

        # Check if name already exists and if so add iterator
        if not overwrite:
            items = ' _ '.join(object.keys())
            n = len(re.findall(name, items))
            name += ' ' + str(n)

        if type(data) is tuple:
            dataset = object.create_dataset(
                name, data, compression="gzip",
                maxshape=(None, *data[1:]))
            HDF5.write_metadata(dataset, metadata)
        else:
            try:
                shape = np.shape(data)
                dataset = object.create_dataset(
                    name, data=data, compression="gzip",
                    maxshape=(None, *shape[1:]))
                HDF5.write_metadata(dataset, metadata)
            except ValueError:
                if overwrite:
                    if verbose:
                        tools.logprint('Waring: dataset already exists, '
                                       'will overwrite.', 'yellow')
                    del object[name]
                    shape = np.shape(data)
                    dataset = object.create_dataset(
                        name, data=data, compression="gzip",
                        maxshape=(None, *shape[1:]))
                    HDF5.write_metadata(dataset, metadata)
                else:
                    if verbose:
                        tools.logprint('Waring: dataset already exists, '
                                       'selected existing dataset as target.',
                                       'yellow')
                    dataset = object[name]

        return dataset

    def append_dataset(dataset, new_data):
        current = dataset.shape[0]
        new = new_data.shape[0]
        total = current + new
        dataset.resize(total, axis=0)
        dataset[current:total] = new_data

    def write_metadata(object, metadata):
        if not type(metadata) is dict:
            raise TypeError('Variable "settings" should be of type dict.')

        for key in metadata.keys():
            object.attrs[key] = metadata[key]

    def read_metadata(object, print=True):
        metadata = dict(object.attrs)
        if print:
            tools.print_dict(metadata)
        return metadata

    def _get_parent(self, parent):
        if parent is None:
            return self.file
        else:
            return self.file[parent]

    def __name__(self):
        return self.name

    def __del__(self):
        self.file.close()
