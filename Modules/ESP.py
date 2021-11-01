import pyvisa
try:
    import Modules.tools as tools
except:
    import tools

class ESP():
    READ_TERMINATION = '\n'
    WRITE_TERMINATION = '\n'
    TIMEOUT = 60000 #60 seconds

    def __init__(self, axis, MAX_POSITION = [-20,100], testflight = False):
        self.MAX_POSITION = MAX_POSITION
        if testflight:
            self.current_position = 0.0
            tools.logprint('*** Initializing fictional connection with ESP ***')
        else:
            tools.logprint('*** Initializing new connection with ESP ***')
            rm = pyvisa.ResourceManager()
            #print('Resource manager used:       ',rm)
            resources = rm.list_resources()
            #print('Detected devices:          ',resources)

            if 'GPIB' in '\t'.join(resources):
                first = next(filter(lambda resource: 'GPIB' in resource, resources), None)
                self.instrument = rm.open_resource(first)
                print('Connected instrument:        ',self.instrument.query("*IDN?"))

                #Set universal ESP300 parameters
                self.instrument.read_termination = self.READ_TERMINATION
                self.instrument.write_termination = self.WRITE_TERMINATION
                self.instrument.timeout = self.TIMEOUT

            else:
                raise IOError('Could not find GPIB device.')

    def send_command(self,command):
        if not self.testflight:
            self.instrument.write(command)
            err = self.instrument.query('TE?')
            if err != '0\r':
                print('Error code:      ',err)
                return False
            else:
                return True

    def current_position(self):
        if self.testflight:
            return self.current_position
        else:
            return float(self.instrument.query(str(self.axis)+'TP?')[:-1])

    def move_relative(self, degrees=5.0):
        new_position = self.current_position() + degrees

        if self.MAX_POSITION[0] <= new_position <= self.MAX_POSITION[1]:
            tools.logprint(f'Moving {str(degrees)} degrees.')
            self.send_command(f'{str(self.axis)}PR{str(degrees)};{str(self.axis)}WS')
        else:
            tools.logprint(f'Desired position {new_position} is out of operating bounds of axis {self.axis}.')

    def move_absolute(self, degrees=5.0):
        if self.MAX_POSITION[0] <= degrees <= self.MAX_POSITION[1]:
            tools.logprint(f'Moving to position {str(degrees)} degrees.')
            self.send_command(f'{str(self.axis)}PA{str(degrees)};{str(self.axis)}WS')
        else:
            tools.logprint(f'Cannot move to position {str(degrees)} as it is larger than the bounds of the motor.')

    def define_home(self):
        tools.logprint('***  NEW HOME SEQUENCE   ***')
        print('Current position:        ',self.current_position())
        input(f'Please move axis {self.axis} to desired position manually and press enter.')
        print('Selected position:        ',self.current_position())
        if input('Are you sure you want to use this position? (Y/N)') in ['Y','y']:
            if self.send_command(str(self.axis)+'DH'):
                print('New home position saved.')
            else:
                print('Failed to set new home position.')

    def move_home(self):
        '''
        For some reason, if you set the home position through this code, it will not completely override the
        home position set in firmware. Therefore it is better to calculate manually how to move to return to home.
        '''
        tools.logprint('Moving to home position.')
        #self.send_command(f'{str(axis)}OR;{str(axis)}WS')
        #self.move_relative(degrees= (self.current_position()*-1) )
        self.move_absolute( degrees = 0.0 )

if __name__ == '__main__':
    #Print all available methods
    import inspect
    cam_arm = ESP(axis=1, testflight=True)
    print([ m for m in dir(ESP) if not m.startswith('__')])

    cam_arm.move_absolute(200)
