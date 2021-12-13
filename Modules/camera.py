import atexit
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.axes_grid1 import make_axes_locatable
from pylablib.devices import uc480

try:
    import Modules.tools as tools
except ModuleNotFoundError:
    import tools


class Cam():

    def __init__(self, name, settings={}, testflight=False):
        # Convert supplied parameters to their 'self' equivalents.
        for key in dir():
            if 'self' not in key:
                self.__setattr__(key, eval(key))

        default_settings = {
            'pixel_rate': 5e6,
            'exposure': 0.2,
            'frame_period': 0.2,
            'gains': (1.0, 1.0, 1.0, 1.0),
            'color_mode': 'rgb8p',
            'roi': (0, 1280, 0, 1024, 1, 1)
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
            if print:
                tools.print_dict(settings)
            return settings

    def set_settings(self, settings={}, print=True):
        if not type(settings) is dict:
            raise TypeError('Variable "settings" should be of type dict.')
        s = self.instrument.apply_settings(settings)
        if print:
            self.get_settings(True)

    def take_images(self, nframes=20, median=True, show=True):
        if self.testflight:
            img = self._random_image(nframes)
        else:
            frame_period = self.instrument.get_frame_timings()[1]
            max_TO = frame_period * 2 + 1.0
            self.instrument.start_acquisition()
            self.instrument.wait_for_frame(
                nframes=nframes, timeout=(max_TO * nframes, max_TO))
            img = self.instrument.read_multiple_images()
            self.instrument.stop_acquisition()

        img = np.array(img)

        if median and nframes > 1:
            img = np.median(img, axis=0)

        # Following lines are to ensure output is of shape [frames, x, y, 3]
        if len(img.shape) < 4:
            img = np.expand_dims(img, axis=0)

        if show:
            maps = ['autumn', 'summer', 'winter']
            fig, ax = plt.subplots(ncols=3, figsize=(17, 5))
            frame = img[-1]
            for i in range(3):
                channel = frame[:, :, i]
                title = (f'min: {int(np.min(channel))}. '
                         + f'max: {int(np.max(channel))}.\n'
                         + f'mean: {np.round(np.mean(channel),2)}. '
                         + f'median: {np.round(np.median(channel),2)}.')
                im = ax[i].imshow(channel, cmap=maps[i], origin='lower')
                ax[i].set_title(title)
                # Colorbar stuff
                divider = make_axes_locatable(ax[i])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im, cax=cax, orientation='vertical')

            fig.tight_layout()
            plt.show()

        return img

    def _random_image(self, nframes=1):
        img = []
        for frame in range(nframes):
            x = y = np.linspace(-5, 5, 500)
            z = [1, 1, 1]
            X, Y, Z = np.meshgrid(x, y, z)
            noise = np.random.random((500, 500, 3)) - 0.5
            f = np.sinc(np.hypot(X, Y)) + noise
            img.append(f)

        return(img)

    def __del__(self):
        tools.logprint('Disconnecting camera', 'yellow')
        if not self.testflight:
            self.instrument.close()


if __name__ == '__main__':

    cam = Cam()
    tools.print_dict(cam.instrument.get_full_info())
    cam.get_settings()
    img = cam.take_images(5, show=True, median=True)
