import pyvisa
import atexit
from sys import platform

try:
    import Modules.tools as tools
except ModuleNotFoundError:
    import tools


class ESP():
    READ_TERMINATION = '\r'
    WRITE_TERMINATION = '\r'
    TIMEOUT = 60000  # milliseconds
    COMMANDS = {
        'define home': 'DH',
        'move to absolute position': 'PA',
        'move to relative position': 'PR',
        'read error code': 'TE?',
        'search for home': 'OR',
        'set home search mode': 'OM',
        'set velocity': 'VA',
        'wait for motion stop': 'WS',
    }
    ERROR_CODES = {
        '7': 'PARAMETER OUT OF RANGE',
        '9': 'AXIS NUMBER OUT OF RANGE',
        '10': 'MAXIMUM VELOCITY EXCEEDED',
        '13': 'MOTOR NOT ENABLED',
        '37': 'AXIS NUMBER MISSING',
        '38': 'COMMAND PARAMETER MISSING'
    }

    def __init__(self, axis, MAX_POSITION=[-39, 110], velocity=5,
                 testflight=False):
        # Convert supplied parameters to their 'self' equivalents.
        for key in dir():
            if 'self' not in key:
                self.__setattr__(key, eval(key))

        if testflight:
            self.current_position = 0.0
            tools.logprint(f'Initializing {tools.bcolors.yellow("fictional")} '
                           'connection with ESP.')
        elif platform != "win32":
            self.current_position = 0.0
            tools.logprint('Not running on Windows platform. Turning on '
                           'testflight mode.', 'yellow')
            self.testflight = True
        else:
            tools.logprint('Initializing new connection with ESP '
                           f'axis {self.axis}.')
            rm = pyvisa.ResourceManager()
            resources = rm.list_resources()
            # print('Resource manager used:       ',rm)
            # print('Detected devices:          ',resources)
            if 'GPIB' in '\t'.join(resources):
                first = next(filter(
                    lambda resource: 'GPIB' in resource, resources), None)
                self.instrument = rm.open_resource(first)

                # Set universal ESP300 parameters
                self.instrument.read_termination = self.READ_TERMINATION
                self.instrument.write_termination = self.WRITE_TERMINATION
                self.instrument.timeout = self.TIMEOUT
                # Set device specific parameters
                self.send_command('set velocity', velocity)
                self.send_command('set home search mode', 0)

            else:
                raise IOError('Could not find GPIB device.', 'red')

        #atexit.register(self.move_home)

    def send_command(self, command, parameter=''):
        if self.testflight:
            return True
        else:
            full_command = f'{self.axis}{self.COMMANDS[command]}{parameter}'
            if 'move' in command:
                wait = f'{self.axis}{self.COMMANDS["wait for motion stop"]}0'
                full_command += ';' + wait
            self.instrument.write(full_command)
            err = self.instrument.query(self.COMMANDS['read error code'])
            if err[0] == str(self.axis) and len(err) > 2:
                err = err[1:]
            if err != '0':
                if err in self.ERROR_CODES:
                    tools.logprint(
                        f'Error code {err}: {self.ERROR_CODES[err]}', 'red')
                else:
                    tools.logprint(f'Error code {err}', 'red')
                return False
            else:
                return True

    def send_ASCII_command(self, command):
        if self.testflight:
            return True
        else:
            self.instrument.write(command)
            err = self.instrument.query(self.COMMANDS['read error code'])
            if err[0] == str(self.axis) and len(err) > 2:
                err = err[1:]
            if err != '0':
                if err in self.ERROR_CODES:
                    tools.logprint(
                        f'Error code {err}: {self.ERROR_CODES[err]}', 'red')
                else:
                    tools.logprint(f'Error code {err}', 'red')
                return False
            else:
                return True

    def get_current_position(self):
        if self.testflight:
            return self.current_position
        else:
            return float(self.instrument.query(str(self.axis)+'TP?')[:-1])

    def define_home(self, degrees=0):
        tools.logprint('New home sequence! \n', 'yellow')
        print(f'You are about to redefine current position '
              f' {self.get_current_position()} as {degrees}.')
        choice = input('Are you sure you want to make this change? (Y/N)')
        if choice in ['Y', 'y']:
            if self.send_command('define home'):
                tools.logprint('New home position saved.', 'green')
            else:
                tools.logprint('Failed to set new home position.', 'red')

    def move(self, degrees, mode, verbose=True):
        '''
        For some reason, if you set the home position through this code, it
        will not completely override the home position set in firmware.
        Therefore it is better to calculate manually how to move to return to
        home.
        '''
        if mode == 'home':
            self.send_command('search for home')
            return None
        elif mode == 'absolute':
            new_position = degrees
        elif mode == 'relative':
            new_position = self.get_current_position() + degrees

        allowed = self.MAX_POSITION[0] <= new_position <= self.MAX_POSITION[1]
        if allowed:
            if verbose: tools.logprint(f'Moving axis {self.axis} to position '
                                       f'{new_position} degrees.')
            self.send_command('move to absolute position', new_position)
        else:
            tools.logprint(f'Desired position {new_position} is out of '
                           f'operating bounds of axis {self.axis}.', 'red')

    # DEPRECATED ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def move_absolute(self, degrees, verbose=True):
        if self.MAX_POSITION[0] <= degrees <= self.MAX_POSITION[1]:
            if verbose: tools.logprint(f'Moving to position {degrees} degrees.')
            self.send_command('move to absolute position', degrees)
        else:
            tools.logprint(f'Desired position {degrees} is out of '
                           f'operating bounds of axis {self.axis}.', 'red')

    def move_relative(self, degrees, verbose=True):
        new_position = self.get_current_position() + degrees

        if self.MAX_POSITION[0] <= new_position <= self.MAX_POSITION[1]:
            if verbose:tools.logprint(f'Moving {degrees} degrees.')
            self.send_command('move to relative position', degrees)
        else:
            tools.logprint(f'Desired position {new_position} is out of '
                           f'operating bounds of axis {self.axis}.', 'red')

    def move_home(self, verbose=True):
        if verbose: tools.logprint('Moving to home position.')
        # self.send_command(f'{str(axis)}OR;{str(axis)}WS')
        # self.move_relative(degrees= (self.get_current_position()*-1) )
        self.move_absolute(0.00000,False)



if __name__ == '__main__':
    cam_arm = ESP(axis=1, testflight=True)
    # Print all available methods
    # import inspect
    # print([ m for m in dir(cam_arm) if not m.startswith('__')])

    # cam_arm.move_absolute(200)
