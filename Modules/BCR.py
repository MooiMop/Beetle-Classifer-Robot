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
    from Modules.ESP import ESP
    from Modules.camera import Cam
except ModuleNotFoundError:
    import tools
    from ESP import ESP
    from camera import Cam


class BCR():

    # Useful variables
    # Directory of this file
    source_folder = os.path.dirname(os.path.abspath(__file__))
    # parent directory of source
    main_folder = os.path.abspath(
        os.path.join(source_folder, os.pardir))
    # default_folder = os.path.join(
    #    main_folder, 'Experiments',
    #    time.strftime('%Y-%m-%d', time.localtime()))
    default_folder = os.path.join(
        main_folder, 'Experiments')

    def __init__(self, name, path, camsettings={},testflight=False):

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
        self.cam_axis = ESP(axis=1, velocity=10, testflight=testflight)
        self.sample_axis = ESP(axis=2, velocity=2, testflight=testflight)
        self.cam = Cam(camsettings,testflight=testflight)

    def _set_path(self, name):

        # default_path = os.path.join(self.default_folder, name)
        # print(default_path)
        default_path = self.default_folder
        cond1 = os.path.exists(self.path)
        codn2 = os.path.exists(os.path.join(self.main_folder,self.path))
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

    def brewster(self, name, metadata={}, **kwargs):
        default_params = {
            'dark': True,
            'domain': [30, 75],
            'median': True,
            'nframes': 25,
            'overwrite': False,
            'show': True,
            'step_size': 0.5,
            'verbose': False,
        }
        default_params.update(kwargs)
        globals().update(default_params)

        default_metadata = {
            'Date': time.strftime('%Y-%m-%d', time.localtime()),
            'Domain (degrees)': domain,
            'Median': median,
            'Number of frames': nframes,
            'Step size (degrees)': step_size,
        }
        default_metadata.update(metadata)
        default_metadata.update(self.cam.get_settings())
        metadata = default_metadata

        savefile = self.savefile_open()
        sequence = self.savefile_new_sequence(
            name, savefile, metadata, overwrite)

        if dark:
            dark_frame = self.get_dark_frame()
            self.savefile_save_measurement(
                'dark frame', sequence, dark_frame, metadata={})

        tools.logprint('Starting measurement sequence.', 'yellow')
        positions = np.arange(domain[0], domain[1] + step_size, step_size)
        for position in tqdm(positions):
            self.sample_axis.move_absolute(position, verbose)
            self.cam_axis.move_absolute(-90 + 2 * position, verbose)
            img = self.cam.take_images(nframes, median, show)
            measurement_metadata = {
                'angle': self.sample_axis.get_current_position(),
                'timestamp': time.strftime('%Y%m%d-%H:%M', time.localtime()),
            }
            self.savefile_save_measurement(
                f'{position} degrees', sequence, img, measurement_metadata)

        savefile.close()
        tools.logprint('Measurement sequence completed!', 'green')

    def light_source_variance(self, end_datetime, metadata={}, **kwargs):
        try:
            end = datetime.datetime(*end_datetime)
        except TypeError:
            try:
                end = end_datetime + datetime.timedelta(microseconds = 1)
            except TypeError:
                tools.logprint('ERROR: end_datetime argument invalid. '
                               'Exiting', 'red')
                return

        default_params = {
            'dark': True,
            'median': True,
            'nframes': 100,
            'overwrite': False,
            'show': False,
        }
        default_params.update(kwargs)
        globals().update(default_params)

        default_metadata = {
            'Median': median,
            'Number of frames': nframes,
        }
        default_metadata.update(metadata)
        default_metadata.update(self.cam.get_settings())
        metadata = default_metadata

        tools.logprint('Starting light source variation measurement.')
        to_go = end - datetime.datetime.now()
        tools.logprint(f'Total time: '
                       f'{to_go.days} days, '
                       f'{np.floor(to_go.seconds/3600)} hours, and '
                       f'{np.floor(to_go.seconds%3600/60)} minutes.')

        while datetime.datetime.now() < end:
            to_go = end - datetime.datetime.now()
            if to_go.seconds%3600 < 100:
                tools.logprint(f'{to_go.days} days and '
                               f'{np.floor(to_go.seconds/3600)} hours to go.')

            savefile = self.savefile_open(verbose=False)
            date = datetime.date.today().strftime('%Y.%m.%d')
            try:
                sequence = savefile.create_group(date)
                metadata['Date'] = date
                for key in metadata.keys():
                    sequence.attrs[key] = metadata[key]
            except ValueError:
                sequence = savefile[date]

            img = self.cam.take_images(nframes, median, show)
            timestamp = datetime.datetime.now().strftime('%H:%M')
            self.savefile_save_measurement(
                timestamp, sequence, img, {'timestamp':timestamp})
            savefile.close()
            time.sleep(60)

        tools.logprint('Measurement sequence completed!', 'green')
    def get_dark_frame(self):

        print(tools.bcolors.red('WARNING. About to measure dark frame. Please '
                                'make sure all lights are off.'))
        input('Press ENTER to continue.')
        img = self.cam.take_images(100,True,False)
        tools.logprint(tools.bcolors.green('Captured dark frame.'))
        input('Press ENTER to continue.')
        return img

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

    exp.light_source_variance(now_plus_2,{})
