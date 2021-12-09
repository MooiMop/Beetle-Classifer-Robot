'''Docstring
to do...
Use pycodestyle [filename] to check style.
'''

__author__ = "Naor Scheinowitz"
__credits__ = ["Naor Scheinowitz"]
__license__ = "GPL-3.0-or-later"
__version__ = "0.5"
__maintainer__ = "Naor Scheinowitz"
__email__ = "scheinowitz@physics.leidenuniv.nl"
__status__ = "Production"

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
            'bounds': [-39,100],
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

    def __init__(self, user, camsettings={},testflight=False):

        print(tools.bcolors.red('Welcome to the Beetle Classifier Robot, '
                                 'great to have you back!'))

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

    def brewster(self, mode, polarization, bounds, step_size, comment,
            filter=None, name='Brewster', path='Experiments\Brewster Angle',
            metadata={}, **kwargs):

        # Check if polarization is valid
        if polarization in ['S', 'P']:
            polarization += ' polarization'
        elif not polarization in ['S polarization', 'P polarization']:
            tools.logprint('Polarization name is invalid.', 'red')
            return None

        # set experiment folder
        self._set_path(path)

        # Load default parameters and update them with the new ones.
        globals().update(self.DEFAULT_PARAMS | kwargs)

        # Metadata
        hierarchy = '''One group per filter. Each group has two subgroups, one
                    for each polarization. Every polarization group has N
                    datasets of shape [frame,x,y,3], with N being set by
                    parameter "repeats". Positions (in degrees) embedded as
                    list in metadata of dataset with key "positions".'''
        default_metadata = {
            'date': tools.get_time(format='date'),
            'hierarchy': hierarchy,
            'median': median,
            'nframes': nframes,
            'bounds': bounds,
            'repeats': repeats,
            'step_size': step_size,
            'polarization_definition': '0 is vertical (S) polarization',
        }
        metadata = default_metadata | metadata
        metadata.update({'readme':comment})

        # Prepare savefile
        if mode == 'create':
            f = HDF5(name, 'create', self.user, True, metadata)
        elif mode == 'add':
            f = HDF5(name, 'open')
        if filter is None:
            filter = 'No filter'
        try:
            group = f.file[filter]
        except KeyError:
            group = f.create_group(filter)
        try:
            subgroup = group[polarization]
        except KeyError:
            subgroup = f.create_group(polarization, parent=filter)
        parent = '/'.join((filter, polarization))

        # Get dark frame
        if dark:
            dark_frame = self.get_dark_frame()
            f.create_dataset('dark frame', dark_frame, parent=parent)

        # Start measurement
        tools.logprint('Starting measurement.')
        positions = np.arange(bounds[0], bounds[1] + step_size, step_size)
        pol = self.POLARIZATIONS[polarization[0]]
        self.polarizer.set(pol, verbose)

        for repeat in range(repeats):
            tools.logprint('Starting run ' + str(repeat + 1))
            name = f'Frames {repeat + 1}'
            for index, position in enumerate(tqdm(positions)):
                # Move
                self.sample.move(position, 'absolute', verbose)
                self.big_arm.move(-90 + 2 * position, 'absolute', verbose)

                # Measure
                img = self.cam.take_images(nframes, median, show)

                # Save
                if index == 0:  # Create empty dataset
                    metadata = self.cam.get_settings(print=False) | {'positions': [position]}
                    dset = f.create_dataset(name, img, parent, metadata)
                else:
                    HDF5.append_dataset(dset,img)
                    old_pos = dset.attrs['positions']
                    new_pos = np.append(old_pos, position)
                    HDF5.write_metadata(dset, {'positions': new_pos})

        f.close()
        tools.logprint('Measurement sequence completed!', 'green')

    def reflection_angle(self, bounds, step_size, comment, angle=45,
            name='reflection_angle', path='Experiments\Reflection Angle',
            metadata={}, **kwargs):

        # set experiment folder
        self._set_path(path)

        # Load default parameters and update them with the new ones.
        globals().update(self.DEFAULT_PARAMS | kwargs)

        # Metadata
        hierarchy = 'No groups. Single dataset of shape [frame,x,y,3]. '\
                    'Positions embedded as list in metadata of dataset with '\
                    'key "positions".'
        default_metadata = {
            'reflection_angle': angle,
            'date': tools.get_time(format='date'),
            'hierarchy': hierarchy,
            'median': median,
            'nframes': nframes,
            'bounds': bounds,
            'repeats': repeats,
            'step_size': step_size,
        }
        metadata = default_metadata | metadata
        metadata.update({'readme':comment})

        # Prepare savefile
        f = HDF5(name, 'create', self.user, True, metadata)

        # Get dark frame
        if dark:
            dark_frame = self.get_dark_frame()
            f.create_dataset('dark frame', dark_frame)

        # Start measurement
        tools.logprint('Starting measurement.')
        positions = np.arange(bounds[0], bounds[1] + step_size, step_size)
        self.sample.move(angle, 'absolute', verbose)

        for repeat in range(repeats):
            tools.logprint('Starting run ' + str(repeat + 1))
            name = f'Frames {repeat + 1}'
            for index, position in enumerate(tqdm(positions)):
                # Move
                self.big_arm.move(-90 + 2 * position, 'absolute', verbose)

                # Measure
                img = self.cam.take_images(nframes, median, show)

                # Save
                if index == 0:  # Create empty dataset
                    metadata = self.cam.get_settings(print=False) | {'positions': [position]}
                    dset = f.create_dataset(name, img, metadata=metadata)
                else:
                    HDF5.append_dataset(dset,img)
                    old_pos = dset.attrs['positions']
                    new_pos = np.append(old_pos, position)
                    HDF5.write_metadata(dset, {'positions': new_pos})

        f.close()
        tools.logprint('Measurement sequence completed!', 'green')

    def light_source_variance(self, end_datetime, comment, dt, name='Lamp variance',
                              path='Experiments\Brewster Angle', metadata={},
                              **kwargs):
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

        # set experiment folder
        self._set_path(path)

        # Load default parameters and update them with the new ones.
        globals().update(self.DEFAULT_PARAMS | kwargs)

        # Metadata
        hierarchy = '''No groups. Single dataset of shape [frame,x,y,3].
                    Timestamps embedded as list in metadata of dataset with
                    key "timestamps".'''
        default_metadata = {
            'dt (seconds)': dt,
            'hierarchy': hierarchy,
            'median': median,
            'nframes': nframes,
        }
        metadata = default_metadata | metadata | self.cam.get_settings()
        metadata.update({'readme':comment})

        # Prepare savefile
        start = tools.get_time(format='date')
        end = tools.convert_time(end_datetime, '%m.%d')
        name = f'Lamp variation from {start} to {end}'
        f = HDF5(name, 'create', self.user, False, metadata)

        # Get dark frame
        if dark:
            dark_frame = self.get_dark_frame()
            timestamp = tools.get_time()
            f.create_dataset('dark frame', dark_frame,
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

        for step in tqdm(range(steps)):
            next_step = tools.get_time('datetime') + datetime.timedelta(0,dt)
            img = self.cam.take_images(nframes, median, show)
            timestamp = tools.get_time()
            if step == 0:
                dset = f.create_dataset('frames',img)
                dset.attrs['timestamps'] = [timestamp]
            else:
                HDF5.append_dataset(dset,img)
                t = dset.attrs['timestamps']
                t = np.append(t, timestamp)
                dset.attrs['timestamps'] = t
            while tools.get_time('datetime') < next_step:
                time.sleep(dt/5)

        f.close()
        tools.logprint('Measurement sequence completed!', 'green')

    def light_source_polarization(self, bounds, step_size, comment,
        name='Lamp Polarization', path='Experiments\Lamp Polarization',
        metadata={}, **kwargs):
        # Parameters

        # set experiment folder
        self._set_path(path)

        # Load default parameters and update them with the new ones.
        globals().update(self.DEFAULT_PARAMS | kwargs)

        # Metadata
        hierarchy = 'No groups. One dataset of shape [polarizations,x,y,3]'\
                    'Positions (in polarization angle) embedded as list in '\
                    'metadata of dataset with key "positions".'
        default_metadata = {
            'date': tools.get_time(format='date'),
            'hierarchy': hierarchy,
            'median': median,
            'nframes': nframes,
            'repeats': repeats,
            'step_size': step_size,
            'bounds': bounds,
            'polarization_definition': '0 is vertical (S) polarization',
        }
        metadata = default_metadata | metadata
        metadata.update({'readme':comment})

        # Prepare savefile
        f = HDF5(name, 'create', self.user, True, metadata)

        # Get dark frame
        if dark:
            dark_frame = self.get_dark_frame()
            f.create_dataset('dark frame', dark_frame)

        # Start measurement
        tools.logprint('Starting measurement.')
        min = bounds[0]
        max = bounds[1]
        positions = np.arange(min, max + step_size, step_size)
        self.sample.move(90, 'absolute', True)
        self.big_arm.move(90, 'absolute', True)
        for repeat in tqdm(range(repeats)):
            #tools.logprint('Starting run ' + str(repeat + 1))
            name = f'Frames {repeat + 1}'
            for index, position in enumerate(positions):
                # Move
                self.polarizer.set(position, verbose)

                # Measure
                img = self.cam.take_images(nframes, median, show)
                pos = self.polarizer.lin_polarizer.get_current_position()

                # Save
                if index == 0:
                    # Create new empty dataset
                    metadata = self.cam.get_settings(print=False) | {'positions': [pos]}
                    dset = f.create_dataset(name, img, metadata=metadata)
                else:
                    HDF5.append_dataset(dset,img)
                    old_pos = dset.attrs['positions']
                    new_pos = np.append(old_pos, pos)
                    HDF5.write_metadata(dset, {'positions': new_pos})

        f.close()
        tools.logprint('Measurement sequence completed!', 'green')

    def median_test(self, comment, name=None, metadata={}, **kwargs):
        # Parameters
        default_params = {
            'dark': True,
            'nframes': 100,
            'verbose': False,
        }
        default_params.update(kwargs)
        globals().update(default_params)

        # Metadata
        hierarchy = 'No groups. One dataset of shape [nframes,x,y,3]'
        default_metadata = {
            'date': tools.get_time(format='date'),
            'hierarchy': hierarchy,
            'nframes': nframes,
        }
        default_metadata.update(metadata)
        default_metadata.update({'readme':comment})
        default_metadata.update(self.cam.get_settings(False))
        metadata = default_metadata

        # Prepare savefile
        if name == None:
            name = self.name
        f = HDF5(name, 'create', self.user, True, metadata)

        # Get dark frame
        if dark:
            dark_frame = self.get_dark_frame()
            f.create_dataset('dark frame', dark_frame)

        # Start measurement
        tools.logprint('Starting measurement.')

        # Measure
        tools.logprint(f'Acquiring {nframes} frames.')
        img = self.cam.take_images(nframes, False, False)

        # Save
        dset = f.create_dataset('frames', img)

        f.close()
        tools.logprint('Measurement sequence completed!', 'green')

    def get_dark_frame(self):

        tools.logprint('WARNING. About to measure dark frame. Please '
                                'make sure all lights are off.', 'yellow')
        time.sleep(30)
        #input('Press ENTER to continue.')
        self.big_arm.move(0, 'home', True)
        self.sample.move(0, 'home', True)
        img = self.cam.take_images(50,False,True)
        tools.logprint('Captured dark frame.', 'green')
        return img


if __name__ == '__main__':

    exp = BCR('Naor Scheinowitz')
    tools.logprint('Loaded BCR session with label "exp". You can input '
                   'commands by typing them in the command line and pressing '
                   'ENTER. Type "exit" to quit experiment.')
    t = 0.0004
    exp.cam.instrument.set_frame_period(t)
    exp.cam.instrument.set_exposure(t)
    while True:
        command = input('~ ')
        if command in ['exit', 'Exit', 'EXIT']:
            tools.logprint('Thanks for using BCR. Please come again!')
            break
        else:
            exec(command)
