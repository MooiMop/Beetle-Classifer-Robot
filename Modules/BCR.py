# Use pycodestyle [filename] to check style.

import matplotlib.pyplot as plt
import numpy as np
import os
import time
import h5py
import re
import datetime

from tqdm import tqdm

try:
    import Modules.tools as tools
    import Modules.ESP as ESP
    from Modules.camera import Cam
    from Modules.HDF5manager import HDF5
except ModuleNotFoundError:
    import tools
    from ESP import ESP
    from camera import Cam
    from HDF5manager import HDF5


class BCR():

    # Useful variables
    # Directory of this file
    source_folder = os.path.dirname(os.path.abspath(__file__))
    # parent directory of source
    main_folder = os.path.abspath(
        os.path.join(source_folder, os.pardir))
    default_folder = os.path.join(
        main_folder, 'Experiments')

    def __init__(self, name, path, author, camsettings={},testflight=False):

        print(tools.bcolors.blue('Welcome to the Beetle Classifier Robot, '
                                 'great to have you back!'))
        print(f'Let\'s start your experiment named: '
              f'{tools.bcolors.red(name)}.\n\n')

        # convert supplied parameters to their 'self' equivalents
        for key in dir():
            if 'self' not in key:
                self.__setattr__(key, eval(key))

        # set experiment folder
        if not self._set_path(name):
            return None

        # initialize devices
        ids = ['GPIB0::1', 'GPIB0::2']
        self.ESP = ESP.ESP(ids)
        self.ESP1 = self.ESP.instruments[0]
        self.ESP2 = self.ESP.instruments[1]

        self.big_arm = ESP.Motor(self.ESP2, 1, [-39,100], 10, testflight)
        self.sample = ESP.Motor(self.ESP2, 2, testflight = testflight)
        self.polarizer = ESP.Polarizer(self.ESP1, self.ESP2, 1, 3, testflight = testflight)

        self.cam = Cam(camsettings, testflight=testflight)

    def _set_path(self, name):

        # default_path = os.path.join(self.default_folder, name)
        # print(default_path)
        default_path = self.default_folder
        cond1 = os.path.exists(self.path)
        cond2 = os.path.exists(os.path.join(self.main_folder,self.path))
        # Check if experiment folder exists
        if self.path is None:
            tools.logprint('No project folder defined. '
                           'Using default output folder.')
            self.path = default_path
        elif not cond1 and not cond2:
            tools.logprint('Project folder does not exist. '
                           'If you are on Windows, make sure to use double '
                           'backslashes. This is what you gave as input: '
                           '\n\n' + self.path + '\n')
            tools.logprint('Using default output folder.')
            self.path = default_path
        try:
            os.chdir(self.path)
            tools.logprint('Working directory changed to: ' + os.getcwd())
        except OSError:
            tools.logprint('Can\'t change the current working directory for '
                           'some reason. Changing to default folder.')
            self.path = default_path
            try:
                os.chdir(self.path)
                tools.logprint('Working directory changed to: ' + os.getcwd())
            except OSError:
                tools.logprint('Still can\'t change working directory. '
                               'Exiting program. Better investigate what\'s '
                               'going on!')
                tools.logprint('Exiting.')
                return False
        return True

    def brewster(self, comment, metadata={}, **kwargs):
        # Parameters
        default_params = {
            'dark': True,
            'domain': [30, 75],
            'median': True,
            'nframes': 25,
            'overwrite': False,
            'repeats': 10,
            'show': False,
            'step_size': 0.5,
            'verbose': False,
        }
        default_params.update(kwargs)
        globals().update(default_params)

        # Metadata
        hierarchy = 'One group per polarization. '\
                    'Each group has one dataset per run with '\
                    'shape [frame,x,y,3]. '\
                    'Positions (in degrees) embedded as list in metadata of '\
                    'dataset with key "positions".'
        default_metadata = {
            'date': tools.get_time(format='date'),
            'domain': domain,
            'hierarchy': hierarchy,
            'median': median,
            'nframes': nframes,
            'repeats': repeats,
            'step_size': step_size,
            'polarization_definition': '0 is vertical (S) polarization',
        }
        default_metadata.update(metadata)
        default_metadata.update({'readme':comment})
        default_metadata.update(self.cam.get_settings())
        metadata = default_metadata

        # Prepare savefile
        file = HDF5(self.name, 'create', self.author, True, metadata)
        file.create_group('P polarization')
        file.create_group('S polarization')
        test_frame = self.cam.take_images(1, False, False)

        # Get dark frame
        if dark:
            dark_frame = self.get_dark_frame()
            file.create_dataset('dark frame', dark_frame)

        # Start measurement
        tools.logprint('Starting measurement.')
        positions = np.arange(domain[0], domain[1] + step_size, step_size)

        for repeat in range(repeats):
            tools.logprint('Starting run ' + str(repeat + 1))
            # Create new empty datasets
            name = f'run {repeat + 1}'
            sdset = file.create_dataset(name, test_frame.shape, 'S polarization')
            sdset.attrs['positions'] = [-1]
            pdset = file.create_dataset(name, test_frame.shape, 'P polarization')
            pdset.attrs['positions'] = [-1]

            for position in tqdm(positions):
                self.sample.move(position, 'absolute', verbose)
                self.big_arm.move(-90 + 2 * position, 'absolute', verbose)

                self.polarizer.set(0, verbose)
                img = self.cam.take_images(nframes, median, show)
                img = np.expand_dims(img, axis=0)
                HDF5.append_dataset(sdset,img)
                self.polarizer.set(90, verbose)
                img = self.cam.take_images(nframes, median, show)
                img = np.expand_dims(img, axis=0)
                HDF5.append_dataset(pdset,img)

                p = sdset.attrs['positions']
                p = np.append(p, self.sample.get_current_position())
                sdset.attrs['positions'] = p
                pdset.attrs['positions'] = p

        sdset.attrs['positions'] = sdset.attrs['positions'][1:]
        pdset.attrs['positions'] = pdset.attrs['positions'][1:]
        file.close()
        tools.logprint('Measurement sequence completed!', 'green')

    def light_source_variance(self, end_datetime, comment,
                              metadata={}, **kwargs):
        # Check if end_datetime makes sense
        if type(end_datetime) is datetime.datetime:
            None
        elif type(end_datetime) is str or type(end_datetime) is tuple:
            try:
                end_datetime = datetime.datetime(*end_datetime)
            except TypeError:
                tools.logprint('ERROR: end_datetime argument invalid. '
                               'Exiting', 'red')
                return

        # Move arms to starting positions.
        #self.big_arm.move(90, 'absolute')
        #self.sample.move(90, 'absolute')
        #self.polarizer.set(0)

        # Parameters
        default_params = {
            'dark': True,
            'dt': 60,
            'median': True,
            'nframes': 25,
            'overwrite': False,
            'show': False,
        }
        default_params.update(kwargs)
        globals().update(default_params)

        # Metadata
        hierarchy = 'No groups. Single dataset of shape [frame,x,y,3]. '\
                    'Timestamps embedded as list in metadata of dataset with '\
                    'key "timestamps".'
        default_metadata = {
            'dt (seconds)': dt,
            'hierarchy': hierarchy,
            'median': median,
            'nframes': nframes,
        }
        default_metadata.update(metadata)
        default_metadata.update({'readme':comment})
        default_metadata.update(self.cam.get_settings())
        metadata = default_metadata

        # Prepare savefile
        start = tools.get_time(format='date')
        end = tools.convert_time(end_datetime, '%m.%d')
        name = f'Lamp variation from {start} to {end}'
        file = HDF5(name, 'create', self.author, False, metadata)
        test_frame = self.cam.take_images(1, False, False)
        dset = file.create_dataset('frames', test_frame.shape)
        dset.attrs['timestamps'] = ['to_remove']

        # Get dark frame
        if dark:
            dark_frame = self.get_dark_frame()
            timestamp = tools.get_time()
            file.create_dataset('dark frame', file, dark_frame,
                                metadata={'timestamp': timestamp})

        # Calculate number of steps of size dt between now and end
        T = end_datetime - datetime.datetime.now()
        steps = int(T.total_seconds() / dt)

        # Start measurement
        tools.logprint('Starting light source variation measurement.')
        tools.logprint(f'Total time: '
                       f'{T.days} days, '
                       f'{np.floor(T.seconds/3600)} hours, and '
                       f'{np.floor(T.seconds%3600/60)} minutes.')

        for t in tqdm(range(steps)):
            next_step = tools.get_time('datetime') + datetime.timedelta(0,dt)

            img = self.cam.take_images(nframes, median, show)
            img = np.expand_dims(img, axis=0)
            timestamp = tools.get_time()
            HDF5.append_dataset(dset,img)
            t = dset.attrs['timestamps']
            t = np.append(t, timestamp)
            dset.attrs['timestamps'] = t

            while tools.get_time('datetime') < next_step:
                time.sleep(dt/5)

        dset.attrs['timestamps'] = dset.attrs['timestamps'][1:]
        file.close()
        tools.logprint('Measurement sequence completed!', 'green')

    def light_source_polarization(self,metadata={},**kwargs):
        default_params = {
            'dark': True,
            'domain': [-45, 135],
            'median': True,
            'nframes': 25,
            'overwrite': False,
            'show': False,
            'step_size': 5,
        }
        default_params.update(kwargs)
        globals().update(default_params)

        default_metadata = {
            'Median': median,
            'Number of frames': nframes,
            'Polarization definition': '0 degrees = vertical (S) polarization',
        }
        default_metadata.update(metadata)
        default_metadata.update(self.cam.get_settings())
        metadata = default_metadata

        file = HDF5(self.name,self.author)
        group = file.create_group('Run', metadata)

        self.big_arm.move(90, 'absolute')
        self.sample.move(90, 'absolute')
        self.pol_axis.move(0, 'absolute')

        if dark:
            dark_frame = self.get_dark_frame()
            file.create_dataset(
                'dark frame', group, dark_frame, metadata={})

        tools.logprint('Starting measurement sequence.', 'yellow')
        positions = np.arange(domain[0], domain[1] + step_size, step_size)
        for position in positions:
            self.pol_axis.move(position, 'absolute')
            polarizor_position = int(np.floor(position + 45 + 152))
            input(f'Please move the polarizor to position '
                  f'{polarizor_position} degrees and '
                  'press ENTER to continue.')
            img = self.cam.take_images(nframes, median, show)
            measurement_metadata = {
                'Lambda/4 plate position  (degrees)': position,
                'Polarization angle (degrees)': position + 45,
                'Polarizor position  (degrees)': polarizor_position,
            }
            file.create_dataset(
                f'{position + 45} degrees', group, img, measurement_metadata)

        file.close()
        tools.logprint('Measurement sequence completed!', 'green')

    def get_dark_frame(self):

        print(tools.bcolors.red('WARNING. About to measure dark frame. Please '
                                'make sure all lights are off.'))
        input('Press ENTER to continue.')
        img = self.cam.take_images(100,True,False)
        tools.logprint(tools.bcolors.green('Captured dark frame.'))
        input('Press ENTER to start measurement.')
        return img

    # DEPRECATED ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def savefile_open(self, verbose=True):

        filename = self.name + '.hdf5'
        if verbose:
            tools.logprint('Opening file ' + tools.bcolors.blue(filename))
        return h5py.File(filename, 'a')

    def savefile_new_sequence(self, name, savefile, metadata, overwrite=False):

        date = time.strftime('%Y-%m-%d', time.localtime())
        name = f'{date} - {name}'
        try:
            sequence = savefile.create_group(name)
        except ValueError:
            if overwrite:
                tools.logprint('Waring: group already exists, will overwrite.',
                               'yellow')
                del savefile[name]
                sequence = savefile.create_group(name)
            else:
                # Calculate number of matching names and add that number + 1
                # to original name.
                text = ' _ '.join(savefile.keys())
                n = len(re.findall(f'{name}*', text))
                name += ' ' + str(n + 1)
                sequence = savefile.create_group(name)
        for key in metadata.keys():
            sequence.attrs[key] = metadata[key]
        tools.logprint('Created HDF5 group ' + tools.bcolors.blue(name))
        return sequence

    def savefile_save_measurement(self, name, sequence, data, metadata,
                                  overwrite=True):

        # Get number of HDF5 groups in file with the same name
        #n = len(list(sequence.keys()))
        try:
            measurement = sequence.create_dataset(
                name, data=data, compression="gzip")
        except ValueError:
            if overwrite:
                del sequence[name]
                measurement = sequence.create_dataset(
                    name, data=data, compression="gzip")
            else:
                tools.logprint('Cannot save data as dataset already exists. '
                               'Set overwrite=True in '
                               'BCR.savefile_save_measurement() to resolve '
                               'this error.', 'red')
                raise ValueError
        for key in metadata.keys():
            measurement.attrs[key] = metadata[key]


if __name__ == '__main__':

    name = 'BCR testflight'
    path = os.path.join('Experiments','Testflight')
    exp = BCR(name, path, {}, True)

    date = time.strftime('%Y-%m-%d', time.localtime())
    measurement_name = 'P polarization'
    metadata = {
        'Polarization source': 'P',
        'color filter': 'no',
    }
    '''exp.brewster(
        measurement_name, metadata,
        dark=False,
        step_size=5,
        show=False
    )'''
    now = datetime.datetime.now()
    now_plus_2 = now + datetime.timedelta(minutes = 2)

    #exp.light_source_variance(now_plus_2,{})

    exp.light_source_polarization()
