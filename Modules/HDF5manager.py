import h5py
import re
import datetime
import numpy as np
from tqdm import tqdm

try:
    import Modules.tools as tools
except ModuleNotFoundError:
    import tools


class HDF5():

    def __init__(self, filename, mode, user=None, date=False, metadata={},
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
            except BlockingIOError:
                raise BlockingIOError(
                    'Cannot open HDF5 file as it is open somewhere.'
                )

        if mode == 'open':
            self.file = h5py.File(filename, 'r+')
            if verbose:
                tools.logprint('Opened file ' + tools.bcolors.blue(filename))
        elif mode == 'read only':
            self.file = h5py.File(filename, 'r')
        elif mode != 'create':
            raise ValueError(f'mode parameter should be "open", "create", or "read only".')

        self.name = filename
        self.filename = filename
        self.verbose = verbose

    # Core functions
    def group(self, name, parent=None, metadata={}):
        object = self._get_parent(parent)
        try:
            group = object[name]
            #tools.logprint('Opened HDF5 group ' + tools.bcolors.blue(name))
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

    # Useful supplementary functions
    def append_dataset(dataset, new_data):
        current = dataset.shape[0]
        new = new_data.shape[0]
        total = current + new
        dataset.resize(total, axis=0)
        dataset[current:total] = new_data

    def combine_datasets(self, parent=None, keyword='Frames'):
        tools.logprint(
            f'Combining datasets with name {keyword} to single dataset.')
        object = self._get_parent(parent)
        datasets = ''.join(object.keys())
        valid = re.findall(keyword + ' \d', datasets)
        for index, dataset in enumerate(tqdm(valid)):
            dset = object[dataset]
            data = dset[:]
            data = np.expand_dims(data, axis=0)
            if index == 0:
                metadata = HDF5.read_metadata(dset, False)
                dset_combined = self.create_dataset(
                    keyword + ' combined',
                    data,
                    parent,
                    metadata=metadata,
                    overwrite=True)
            else:
                HDF5.append_dataset(dset_combined, data)

    def contains_key(self, key: str) -> bool:
        keys = HDF5.allkeys(self.file)
        keys = ' _ '.join(keys)
        return key in keys

    def print_structure(self):
        print('\nFile metadata:')
        for m in self.file.attrs:
            print(f'{m}:  {self.file.attrs[m]}')

        print('\nGroups:')
        for key in self.file.keys():
            print(key)

        grps = list(self.file.keys())
        for grp in grps:
            print(f'\nDatasets of group "{grp}":')
            for key in self.file[grp].keys():
                print(key)

        print(f'\nMetadata of group "{grp}":')
        for m in self.file[grp].attrs:
            print(f'{m}:  {self.file[grp].attrs[m]}')

    def read_metadata(object, print=False):
        metadata = dict(object.attrs)
        if print:
            tools.print_dict(metadata)
        return metadata

    def write_metadata(object, metadata):
        if not type(metadata) is dict:
            raise TypeError('Variable "settings" should be of type dict.')

        for key in metadata.keys():
            object.attrs[key] = metadata[key]

    # Internal functions
    def _get_parent(self, parent):
        if parent is None:
            return self.file
        else:
            return self.file[parent]

    # Dunders
    def __name__(self):
        return self.filename

    #def __del__(self):
    #    try:
    #        self.file.close()
    #        if self.verbose:
    #                tools.logprint(f'Closed file {self.__name__()}.', 'blue')
    #    except ImportError:
    #        None

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        try:
            self.file.close()
            if self.verbose:
                    tools.logprint(f'Closed file {self.__name__()}.', 'blue')
        except ImportError:
            None

    # Functions that can be used without instance

    def merge_files(source, target):
        '''This function assumes no groups of level greater than 1'''
        source = h5py.File(source, 'r')
        target = HDF5(target, 'open')
        for grp in source:
            group = source[grp]
            m = HDF5.read_metadata(group)
            target.group(grp, metadata=m)
            tools.logprint(f'Transferring {len(group)} datasets.')
            for dset in tqdm(group):
                dataset = group[dset]
                m = HDF5.read_metadata(dataset)
                target.create_dataset(dset, dataset[:], grp, m, True, False)
        source.close()
        del target

    def allkeys(obj):
        "Recursively find all keys in an h5py.Group."
        keys = (obj.name,)
        if isinstance(obj, h5py.Group):
            for key, value in obj.items():
                if isinstance(value, h5py.Group):
                    keys = keys + HDF5.allkeys(value)
                else:
                    keys = keys + (value.name,)
        return keys
