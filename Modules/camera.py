import atexit
import matplotlib.pyplot as plt
import numpy as np

from pylablib.devices import uc480

try:
    import Modules.tools as tools
except ModuleNotFoundError:
    import tools


class Cam():

    def __init__(self, settings={}, testflight=False):
        # Make sure the connection to the camera is shutdown when the program
        # exits. Without this, the camera must be physically unplugged each
        # time you rerun the code.
        # WARNING: does not (currently) work with Jupyter notebook yet.
        # Please manually call Cam.shutdown()
        atexit.register(self.shutdown)

        # Convert supplied parameters to their 'self' equivalents.
        for key in dir():
            if 'self' not in key:
                self.__setattr__(key, eval(key))

        default_settings = {
            'pixel_rate': 5e6,
            'frame_period': 0.15,
            'exposure': 0.15,
            'gains': (1.0, 1.0, 1.0, 1.0),
            'color_mode': 'rgb8p',
            'binning': (1, 1),
            'roi': (400,900,300,700,1,1)
        }
        default_settings.update(settings)
        self.settings = default_settings

        if testflight:
            tools.logprint(f'Initializing {tools.bcolors.yellow("fictional")} '
                           'connection with camera.')
        else:
            tools.logprint('Initializing connection with camera')
            try:
                # Connect to first available camera
                self.instrument = uc480.UC480Camera(cam_id=0)
                self.set_settings(self.settings, False)
            except OSError:
                self.testflight = True

    def shutdown(self):
        tools.logprint('Disconnecting camera', 'yellow')
        if not self.testflight:
            self.instrument.close()

    def check_connection(self):
        if self.instrument.is_opened():
            tools.logprint('Camera is connected.', 'green')
            return True
        else:
            raise IOError(tools.bcolors.red('Camera not connected'))
            return False

    def get_settings(self, print=True):
        if self.testflight:
            return {}
        else:
            settings = self.instrument.get_settings()
            tools.print_dict(settings)
            return settings

    def set_settings(self, settings={}, print=True):
        if not type(settings) is dict:
            raise TypeError('Variable "settings" should be of type dict.')
        s = self.instrument.apply_settings(settings)
        if print:
            self.get_settings()

    def take_images(self, nframes=10, median=True, show=True):
        if self.testflight:
            img = self._random_image(nframes)
        else:
            self.instrument.start_acquisition()
            self.instrument.wait_for_frame(nframes=nframes)
            img = self.instrument.read_multiple_images()
            self.instrument.stop_acquisition()

        img = np.array(img)

        if median and nframes > 1:
            img = np.median(img, axis=0)

        if show:
            try:
                maps = ['autumn','summer','winter']
                for i in range(3):
                    plt.imshow(img[:,:,i], cmap=maps[i])
                    plt.colorbar()
                    plt.show()
            except TypeError:  # imshow can only plot 1 image
                plt.imshow(np.sum(img[-1],axis=2))  # plot last image
                plt.show()

        return img

    def _random_image(self, nframes=1):
        img = []
        for frame in range(nframes):
            x = y = np.linspace(-5, 5, 500)
            X, Y = np.meshgrid(x, y)
            noise = np.random.random((500, 500)) - 0.5
            f = np.sinc(np.hypot(X, Y)) + noise
            img.append(f)

        return(img)


if __name__ == '__main__':


    '''cam = Cam()
    print(cam.instrument.get_all_color_modes())
    print(cam.instrument.get_available_pixel_rates())
    print(cam.instrument.get_acquisition_parameters())

    img = cam.take_images(10, show=False, median=False)
    for im in img:
        plt.imshow(im)
        plt.show()
    cam.shutdown()'''

    cam = Cam()
    tools.print_dict(cam.instrument.get_full_info())
    #cam.get_settings()
    img = cam.take_images(10, show=True, median=True)
