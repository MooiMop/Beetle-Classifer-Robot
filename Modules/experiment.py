import matplotlib.pyplot as plt
import numpy as np
import os
import time

try:
    import Modules.tools as tools
    from Modules.ESP import ESP
    from Modules.camera import Cam
except ModuleNotFoundError:
    import tools
    from ESP import ESP
    from camera import Cam


class Experiment():

    # Useful variables
    # Directory of this file
    source_folder = os.path.dirname(os.path.abspath(__file__))
    # parent directory of this file
    main_folder = os.path.abspath(
        os.path.join(source_folder, os.pardir))
    default_folder = os.path.join(
        main_folder, 'Experiments',
        time.strftime('%Y-%m-%d', time.localtime()))

    def __init__(self, name, path, testflight=False):

        print(tools.bcolors.blue('Welcome to the Beetle Classifier Robot, '
                                 'great to have you back!'))
        print(f'Let\'s start your experiment named: {name}')

        # convert supplied parameters to their 'self' equivalents
        for key in dir():
            if 'self' not in key:
                self.__setattr__(key, eval(key))

        # set experiment folder
        self._set_path(name)

        # initialize devices
        cam_axis = ESP(axis=1, testflight=testflight)
        sample_axis = ESP(axis=2, testflight=testflight)
        cam = Cam(testflight=testflight)

    def _set_path(self, name):

        default_path = os.path.join(self.default_folder, name)
        # Check if experiment folder exists
        if self.path is None:
            tools.logprint('No project folder defined. '
                           'Using default output folder.')
            self.path = default_path
        elif not os.path.exists(self.path):
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
                exit()


if __name__ == '__main__':
    path = '/Users/naorscheinowitz/Master Project/Beetle Project/Beetle '\
           'Classifier Robot/Experiments/Test'
    test = Experiment(name='test experiment', path=path, testflight=True)
