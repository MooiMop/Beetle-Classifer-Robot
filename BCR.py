'''Master module for running experiments using the Beetle Classification Robot.

Class method BCR can be called to initialize connected devices. See
BCR.__doc__ for more information on the different methods and README.md for
more information about the project.
'''

__author__ = "Naor Scheinowitz"
__email__ = "scheinowitz@physics.leidenuniv.nl"
__credits__ = ["Naor Scheinowitz"]
__license__ = "GPL-3.0-or-later"
__version__ = "1.0"

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
    '''
    Class for executing measurement sequences.

    Upon initialization, a connection will be established with all devices
    listed in class constant DEVICES. This dict also contains the default
    settings of the devices in the setup.

    After initialization, a measurement sequence can be performed by calling
    any of the different class methods.

    Attributes
    ----------
    user : str
        Name of experimenter to be saved to metadata of data files.
    camsettings : dict (default empty)
        Dictionary of camera settings to be loaded during initialization.
    testflight : bool (default False)
        If True, no actual connections to devices are made. Debug mode.

    Methods
    -------
    beetle_polarization:
        Measure beelte polarization for fixed angle_in and different angles_out.
    brewster:
        Measure (chromatic) brewster angle.
    simple_reflection:
        Take images at different angles without polarization information.
    lamp_variation:
        Take identical images during time interval.
    lamp_polarization:
        Measure linear polarization of lamp.
    median_test:
        Take (many) identical images in series.
    _get_dark_frame:
        Take dark frame.
    _load_defaults:
        Load default values for measurement sequence/
    _set_path:
        Change working directory.
    _estimate_duration:
        Estimate duration of measurement sequence.

    Constants
    ---------
    DEVICES : dict
        Dictionary of devices containing dictionaries of device parameters.
    DEFAULT_PARAMS : dict
        Default parameters for measurement sequence.
    POLARIZATIONS : dict
        Definitions of 'S' and 'P' polarizations.
    '''

    DEVICES = {
        'ESP': {
            'method': 'ESP.ESP',
            'identifiers': ['GPIB0::1', 'GPIB0::2'],
            },
        'big_arm': {
            'method': 'ESP.Motor',
            'instrument': 'self.ESP.instruments[1]',
            'axis': 1,
            'bounds': [-30, 100],
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
        # Check input types
        if type(testflight) != bool:
            raise TypeError('testflight should be of type bool')
        if type(camsettings) != dict:
            raise TypeError('camsettings should be of type dict')

        tools.logprint('Welcome to the Beetle Classifier Robot. Great to have '
                       'you back!', 'blue')

        # convert supplied parameters to their 'self' equivalents
        for key in dir():
            if 'self' not in key:
                self.__setattr__(key, eval(key))

        # initialize devices from DEVICES class constant
        for d in self.DEVICES:
            device = self.DEVICES[d]
            keys = np.array(list(device.keys()))
            keys = keys[keys!='method']  # Remove method key from list
            params = ', '.join(
                f'{x}={device[x]}' for x in keys)
            command = f'self.{d} = {device["method"]}({params}, '\
                      f'testflight={testflight}, name="{d}")'
            exec(command)

    # ~~~ Experiment sequences ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def beetle_polarization(self, mode, angle_in, angles_out,
                 step_size, readme, parent='auto', pol_steps=16,
                 metadata={}, name='Beetle Hyperspectral',
                 path='Experiments', **kwargs):

        metadata['angle_in'] = angle_in
        metadata['angles_out'] = angles_out
        metadata['step_size'] = step_size
        m = self._load_defaults(kwargs, metadata, path, readme)
        high_level_keys = [
            'hierarchy', 'polarization_definition', 'readme',
            'software_version', 'user']
        m_high_level =  {k: v for k, v in m.items() if k in high_level_keys}
        m_low_level =  {k: v for k, v in m.items() if k not in high_level_keys}

        # Create or open savefile
        if mode == 'create':
            args = (name, 'create', self.user, True, m_high_level)
        elif mode == 'add':
            args = (name, 'open')

        with HDF5(*args) as f:
            # Check if HDF5 group exists and create if not
            if type(parent) is str:
                if parent == 'auto':
                    parent = f'{angle_in} degree reflection'
                f.group(parent, metadata=m_low_level)
            elif parent is None:
                pass  # this means that the dataset will be saved without a group

            if dark:
                self._get_dark_frame(f, parent)

            # Start measurement
            tools.logprint('Starting measurement.')
            self._estimate_duration(pol_steps, angles_out, step_size)
            self.sample.move(angle_in, 'absolute', verbose)
            angles = np.arange(
                angles_out[0],
                angles_out[1] + step_size,
                step_size)
            polarizer_angles = np.round(
                np.linspace(0, 180, pol_steps),
                2)
            try:
                self.polarizer.lin_polarizer.move(90, 'absolute', verbose)
                for repeat in tqdm(range(repeats)):
                    for index, angle in enumerate(angles):
                        # Move stuff
                        position_adjusted = -90 + angle_in + angle
                        self.big_arm.move(position_adjusted, 'absolute', verbose)

                        # Take images at different polarizer angles
                        img = []
                        for p in polarizer_angles:
                            self.polarizer.quart_lambda.move(
                                p, 'absolute', verbose)
                            img.append(
                                self.cam.take_images(nframes, median, show)[0])
                        img = np.array(img)  # transform list to array
                        img = img[np.newaxis, :]   # add new axis so it can be appended to other images

                        # Save images
                        if index == 0:  # create dataset after first images
                            m = self.cam.get_settings(print=False)
                            m['angles_out'] = angles
                            m['polarizer_angles'] = polarizer_angles
                            dset = f.create_dataset('Frames', img, parent, m)
                        else:
                            HDF5.append_dataset(dset, img)
                tools.logprint('Measurement sequence completed!', 'green')
            except KeyboardInterrupt:
                choice = input('Delete uncompleted measurement run? (y/N)')
                if choice in ['y', 'Y']:
                    del f.file[dset.name]

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
        metadata['hierarchy'] = hierarchy
        m = self._load_defaults(kwargs, metadata, path, readme)

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

    def simple_reflection(self, bounds, step_size, readme, angle=45,
                         name='Reflection Angle',
                         path='Experiments\\Reflection Angle', metadata={},
                         **kwargs):

        hierarchy = 'No groups. Single dataset of shape [frame,x,y,3]. '\
                    'Positions embedded as list in metadata of dataset with '\
                    'key "positions".'
        metadata['bounds'] = bounds
        metadata['step_size'] = step_size
        metadata['reflection_angle'] = angle
        metadata['hierarchy'] = hierarchy
        m = self._load_defaults(kwargs, metadata, path, readme)
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
                        dset = f.create_dataset('Frames', img, None, m, overwrite, verbose)
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
        metadata['hierarchy'] = hierarchy
        m = self._load_defaults(kwargs, metadata, path, readme)

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
                    dset = f.create_dataset('Frames', img, None, m, overwrite, verbose)
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
        metadata['hierarchy'] = hierarchy
        m = self._load_defaults(kwargs, metadata, path, readme)
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
                        dset = f.create_dataset('Frames', img, None, m, overwrite, verbose)
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
        metadata['hierarchy'] = hierarchy
        m = self._load_defaults(kwargs, metadata, path, readme)
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
                dset = f.create_dataset('Frames', img, None, m, overwrite, verbose)
                time.sleep(dt)
            tools.logprint('Measurement sequence completed!', 'green')
        except KeyboardInterrupt:
            choice = input('Delete uncompleted measurement run? (y/N)')
            if choice in ['y', 'Y']:
                del f.file[dset.name]

    def _get_dark_frame(self, savefile=None, parent=None, n=5):

        tools.logprint('WARNING. About to measure dark frame. Please make'
                       'sure all lights are off.', 'yellow')
        input('Press ENTER to continue.')
        img = self.cam.take_images(n, False, False)
        tools.logprint('Captured dark frame.', 'green')
        if savefile is not None:
            savefile.create_dataset('dark frame', img, parent)
        input('Press ENTER to continue.')
        return img

    def _load_defaults(self, kwargs, metadata, path, readme):
        # set experiment folder
        self._set_path(path)

        # Load default parameters and update them with the new ones.
        globals().update(self.DEFAULT_PARAMS | kwargs)

        # Metadata
        default_metadata = {
            'date': tools.get_time(format='date'),
            'median': median,
            'nframes': nframes,
            'readme': readme,
            'repeats': repeats,
            'polarization_definition': 'linear polarizer: 0 is vertical '
            + 'quarter lambda plate: 0 is horizontal fast axis.',
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

    def _estimate_duration(self, pol_steps, angles_out, step_size):
        if not self.testflight:
            motor_steps = np.diff(angles_out) // step_size
            steps = nframes * pol_steps * motor_steps
            time_per_step = self.cam.instrument.get_frame_timings()[1] + 1
            measurement = np.round(time_per_step / 3600 * steps, 2)
            duration = measurement * repeats
            tools.logprint(
                f'Estimated duration of measurement: {measurement} hours.\n'
                f'Estimated duration of sequence: {duration} hours.')


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
