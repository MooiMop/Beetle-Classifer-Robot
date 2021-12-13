'''Docstring
to do...
Use pycodestyle [filename] to check style.
'''

__author__ = "Naor Scheinowitz"
__credits__ = ["Naor Scheinowitz"]
__license__ = "GPL-3.0-or-later"
__version__ = "0.6"
__maintainer__ = "Naor Scheinowitz"
__email__ = "scheinowitz@physics.leidenuniv.nl"

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import h5py
import re
import datetime
from tqdm import tqdm

import Modules.tools as tools
import Modules.ESP as ESP
from Modules.camera import Cam
from Modules.HDF5manager import HDF5


class BCR():
    DEVICES = {
        'ESP': {
            'method': 'ESP.ESP',
            'identifiers': ['GPIB0::1', 'GPIB0::2'],
            },
        'big_arm': {
            'method': 'ESP.Motor',
            'instrument': 'self.ESP.instruments[1]',
            'axis': 1,
            'bounds': [-39, 100],
            'velocity': 10,
            },
        'sample': {
            'method': 'ESP.Motor',
            'instrument': 'self.ESP.instruments[1]',
            'axis': 2,
            },
        'polarizer': {
            'method': 'ESP.Polarizer',
            'inst1': 'self.ESP.instruments[0]',
            'inst2': 'self.ESP.instruments[1]',
            'ax1': 1,
            'ax2': 3,
            },
        'cam': {
            'method': 'Cam',
            'settings': 'camsettings',
            },
        }

    DEFAULT_PARAMS = {
        'dark': True,
        'median': True,
        'nframes': 10,
        'overwrite': False,
        'repeats': 10,
        'show': False,
        'verbose': False,
    }

    POLARIZATIONS = {
        'S': 0,
        'P': 90,
    }

    # Default folder paths
    PATH_MAIN = os.path.dirname(
        os.path.abspath(__file__))  # Directory of this file
    PATH_DEFAULT = os.path.join(
        PATH_MAIN, 'Experiments')

    def __init__(self, user, camsettings={}, testflight=False):

        tools.logprint('Welcome to the Beetle Classifier Robot. Great to have '
                       'you back!', 'blue')

        # convert supplied parameters to their 'self' equivalents
        for key in dir():
            if 'self' not in key:
                self.__setattr__(key, eval(key))

        # initialize devices
        for d in self.DEVICES:
            device = self.DEVICES[d]
            params = ', '.join(
                f'{x}={device[x]}' for x in list(device.keys())[1:])
            command = f'self.{d} = {device["method"]}({params}, '\
                      f'testflight={testflight}, name="{d}")'
            exec(command)

    # ~~~ Experiment sequences ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def brewster(self, mode, polarization, bounds, step_size, readme,
                 filter=None, name='Brewster',
                 path='Experiments\\Brewster Angle', metadata={}, **kwargs):

        hierarchy = 'One group per filter. Each group has two subgroups, one '\
                    'for each polarization. Every polarization group has N '\
                    'datasets of shape [frame,x,y,3], with N being set by '\
                    'parameter "repeats". Positions (in degrees) embedded as '\
                    'list in metadata of dataset with key "positions".'
        metadata['bounds'] = bounds
        metadata['step_size'] = step_size
        m = self._init_experiment(hierarchy, kwargs, metadata, path, readme)

        # Check if polarization is valid
        if polarization in ['S', 'P']:
            polarization += ' polarization'
        elif polarization not in ['S polarization', 'P polarization']:
            tools.logprint('Polarization name is invalid.', 'red')
            return None

        # Prepare savefile
        if mode == 'create':
            f = HDF5(name, 'create', self.user, True, m)
        elif mode == 'add':
            f = HDF5(name, 'open')
        if filter is None:
            filter = 'No filter'
        parent = '/'.join((filter, polarization))
        # Check if hdf5 file contains 'parent' groups and otherwise create them
        f.group(parent)

        if dark:
            self._get_dark_frame(f, parent)

        # Start measurement
        tools.logprint('Starting measurement.')
        positions = np.arange(bounds[0], bounds[1] + step_size, step_size)
        pol = self.POLARIZATIONS[polarization[0]]
        self.polarizer.set(pol, verbose)
        try:
            for repeat in tqdm(range(repeats)):
                tools.logprint('Starting run ' + str(repeat + 1))
                name = f'Frames {repeat + 1}'
                for index, position in enumerate(positions):
                    self.sample.move(position, 'absolute', verbose)
                    self.big_arm.move(-90 + 2 * position, 'absolute', verbose)
                    img = self.cam.take_images(nframes, median, show)
                    if index == 0:  # Create empty dataset
                        m = self.cam.get_settings(print=False)
                        m['positions'] = [position]
                        dset = f.create_dataset('Frames', img, parent, m)
                    else:
                        HDF5.append_dataset(dset, img)
                        old_pos = dset.attrs['positions']
                        new_pos = np.append(old_pos, position)
                        dset.attrs['positions'] = new_pos
            tools.logprint('Measurement sequence completed!', 'green')
        except KeyboardInterrupt:
            choice = input('Delete uncompleted measurement run? (y/N)')
            if choice in ['y', 'Y']:
                del f.file[dset.name]

    def reflection_angle(self, bounds, step_size, readme, angle=45,
                         name='Reflection Angle',
                         path='Experiments\\Reflection Angle', metadata={},
                         **kwargs):

        hierarchy = 'No groups. Single dataset of shape [frame,x,y,3]. '\
                    'Positions embedded as list in metadata of dataset with '\
                    'key "positions".'
        metadata['bounds'] = bounds
        metadata['step_size'] = step_size
        metadata['reflection_angle'] = angle
        m = self._init_experiment(hierarchy, kwargs, metadata, path, readme)
        f = HDF5(name, 'create', self.user, True, m)

        if dark:
            self._get_dark_frame(f)

        # Start measurement
        try:
            tools.logprint('Starting measurement.')
            positions = np.arange(bounds[0], bounds[1] + step_size, step_size)
            self.sample.move(angle, 'absolute', verbose)
            for repeat in range(repeats):
                tools.logprint('Starting run ' + str(repeat + 1))
                for index, position in enumerate(tqdm(positions)):
                    self.big_arm.move(-90 + 2 * position, 'absolute', verbose)
                    img = self.cam.take_images(nframes, median, show)
                    if index == 0:
                        m = self.cam.get_settings(print=False)
                        m['positions'] = [position]
                        dset = f.create_dataset('Frames', img, None, m)
                    else:
                        HDF5.append_dataset(dset, img)
                        old_pos = dset.attrs['positions']
                        new_pos = np.append(old_pos, position)
                        dset.attrs['positions'] = new_pos
            tools.logprint('Measurement sequence completed!', 'green')
        except KeyboardInterrupt:
            choice = input('Delete uncompleted measurement run? (y/N)')
            if choice in ['y', 'Y']:
                del f.file[dset.name]

    def lamp_variation(self, end_datetime, dt, readme, name='Lamp Variation',
                       path='Experiments\\Lamp Variation', metadata={},
                       **kwargs):

        hierarchy = 'No groups. Single dataset of shape [frame,x,y,3]. '\
                    'Timestamps embedded as list in metadata of dataset with '\
                    'key "timestamps".'
        metadata['dt (seconds)'] = dt
        m = self._init_experiment(hierarchy, kwargs, metadata, path, readme)

        # Check if end_datetime makes sense
        if type(end_datetime) is datetime.datetime:
            pass
        elif type(end_datetime) is str or type(end_datetime) is tuple:
            try:
                end_datetime = datetime.datetime(*end_datetime)
            except TypeError:
                tools.logprint('ERROR: end_datetime argument invalid. ', 'red')
                return

        # Prepare savefile
        start = tools.get_time(format='date')
        end = tools.convert_time(end_datetime, '%m.%d')
        name += f' from {start} to {end}'
        f = HDF5(name, 'create', self.user, False, m)

        if dark:
            self._get_dark_frame(f)

        # Start measurement
        try:
            tools.logprint('Starting light source variation measurement.')
            # Calculate number of steps of size dt between now and end
            T = end_datetime - datetime.datetime.now()
            steps = T.total_seconds() // dt
            for step in tqdm(range(steps)):
                next_step = (tools.get_time('datetime')
                             + datetime.timedelta(0, dt))
                img = self.cam.take_images(nframes, median, show)
                timestamp = tools.get_time()
                if step == 0:
                    m = self.cam.get_settings(print=False)
                    m['timestamps'] = [timestamp]
                    dset = f.create_dataset('Frames', img, None, m)
                else:
                    HDF5.append_dataset(dset, img)
                    t = dset.attrs['timestamps']
                    t = np.append(t, timestamp)
                    dset.attrs['timestamps'] = t
                while tools.get_time('datetime') < next_step:
                    time.sleep(1)
            tools.logprint('Measurement sequence completed!', 'green')
        except KeyboardInterrupt:
            pass

    def lamp_polarization(self, bounds, step_size, readme,
                          name='Lamp Polarization',
                          path='Experiments\\Lamp Polarization', metadata={},
                          **kwargs):

        hierarchy = 'No groups. One dataset of shape [polarizations,x,y,3]'\
                    'Positions (in polarization angle) embedded as list in '\
                    'metadata of dataset with key "positions".'
        metadata['bounds'] = bounds
        metadata['step_size'] = step_size
        m = self._init_experiment(hierarchy, kwargs, metadata, path, readme)
        f = HDF5(name, 'create', self.user, True, m)

        if dark:
            self._get_dark_frame(f)

        try:
            tools.logprint('Starting measurement.')
            min = bounds[0]
            max = bounds[1]
            positions = np.arange(min, max + step_size, step_size)
            self.sample.move(90, 'absolute', True)
            self.big_arm.move(90, 'absolute', True)
            for repeat in tqdm(range(repeats)):
                for index, position in enumerate(positions):
                    self.polarizer.set(position, verbose)
                    img = self.cam.take_images(nframes, median, show)
                    if index == 0:
                        m = self.cam.get_settings(print=False)
                        m['positions'] = [position]
                        dset = f.create_dataset('Frames', img, None, m)
                    else:
                        HDF5.append_dataset(dset, img)
                        old_pos = dset.attrs['positions']
                        new_pos = np.append(old_pos, position)
                        dset.attrs['positions'] = new_pos
            tools.logprint('Measurement sequence completed!', 'green')
        except KeyboardInterrupt:
            choice = input('Delete uncompleted measurement run? (y/N)')
            if choice in ['y', 'Y']:
                del f.file[dset.name]

    def median_test(self, readme, dt, polarization=0, name='Median Test',
                    path='Experiments\\Median Test', metadata={}, **kwargs):

        hierarchy = 'No groups. Datasets of shape [nframes,x,y,3]'
        metadata['dt'] = dt
        metadata['median'] = False
        metadata['polarization'] = polarization
        m = self._init_experiment(hierarchy, kwargs, metadata, path, readme)
        f = HDF5(name, 'create', self.user, True, m)

        if dark:
            self._get_dark_frame(f)

        try:
            tools.logprint('Starting measurement.')
            self.polarizer.set(polarization, verbose)
            tools.logprint(f'Acquiring {nframes} frames {repeats} times {dt} '
                           'seconds apart.')
            for repeat in tqdm(range(repeats)):
                img = self.cam.take_images(nframes, False, False)
                m = self.cam.get_settings(print=False)
                dset = f.create_dataset('Frames', img, None, m)
                time.sleep(dt)
            tools.logprint('Measurement sequence completed!', 'green')
        except KeyboardInterrupt:
            choice = input('Delete uncompleted measurement run? (y/N)')
            if choice in ['y', 'Y']:
                del f.file[dset.name]

    def _get_dark_frame(self, savefile=None, parent=None, n=50):

        tools.logprint('WARNING. About to measure dark frame. Please make'
                       'sure all lights are off.', 'yellow')
        input('Press ENTER to continue.')
        img = self.cam.take_images(n, False, True)
        tools.logprint('Captured dark frame.', 'green')
        if savefile is not None:
            savefile.create_dataset('dark frame', img, parent)
        input('Press ENTER to continue.')
        return img

    def _init_experiment(self, hierarchy, kwargs, metadata, path, readme):
        # set experiment folder
        self._set_path(path)

        # Load default parameters and update them with the new ones.
        globals().update(self.DEFAULT_PARAMS | kwargs)

        # Metadata
        default_metadata = {
            'date': tools.get_time(format='date'),
            'hierarchy': hierarchy,
            'median': median,
            'nframes': nframes,
            'readme': readme,
            'repeats': repeats,
            'polarization_definition': '0 is vertical (S) polarization',
            'software_verion': __version__,
        }
        metadata = default_metadata | metadata

        return metadata

    def _set_path(self, path):
        os.chdir(self.PATH_MAIN)
        default_path = self.PATH_DEFAULT
        cond1 = os.path.exists(path)
        cond2 = os.path.exists(os.path.join(self.PATH_MAIN, path))

        # Check if experiment folder exists
        if path is None:
            tools.logprint('No project folder defined. '
                           'Using default output folder.')
            path = self.PATH_DEFAULT
        elif not cond1 and not cond2:
            tools.logprint('Given folder does not exist. '
                           'If you are on Windows, it might be useful to use '
                           'double backslashes. This is what you gave as '
                           f'input: \n{path}\n', 'red')
            tools.logprint('Using default output folder.', 'yellow')
            path = self.PATH_DEFAULT

        try:
            os.chdir(path)
            tools.logprint('Working directory changed to: ' + os.getcwd())
        except OSError:
            tools.logprint('Can\'t change the current working directory for '
                           'some reason. Changing to default folder.')
            try:
                os.chdir(self.PATH_DEFAULT)
                tools.logprint('Working directory changed to: ' + os.getcwd())
            except OSError:
                tools.logprint('Still can\'t change working directory. '
                               'Exiting program. Better investigate what\'s '
                               'going on!')
                raise OSError()


if __name__ == '__main__':

    exp = BCR('Naor Scheinowitz', testflight=True)
    tools.logprint('Loaded BCR session with label "exp". You can input '
                   'commands by typing them in the command line and pressing '
                   'ENTER. Type "exit" to quit experiment.')

    while True:
        command = input('~ ')
        if command in ['exit', 'Exit', 'EXIT']:
            tools.logprint('Thanks for using BCR. Please come again!')
            break
        else:
            exec(command)
