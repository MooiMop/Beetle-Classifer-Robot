import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.axes_grid1 import make_axes_locatable
from pylablib.devices import uc480

try:
    import Modules.tools as tools
except ModuleNotFoundError:
    import tools


class Cam():

    def __init__(self, name='ThorCam', settings={}, testflight=False):
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

    def restart(self):
        if not self.testflight:
            self.instrument.close()
            self.__init__()

    def auto_expose(self, E_start=0.01, target=220, margin=20, show=True):
        E_min = 0.0001  # shortest possible exposure (in seconds)
        E_max = 10.0
        E = E_start
        dE = 1
        p_min = np.round(1 / 30, 3)  # camera can only go as low as 0.5 fps
        correctly_exposed = False
        tools.logprint('Starting auto exposure...')
        if self.testflight:
            return None
        else:
            while not correctly_exposed:
                E *= dE
                p = np.max((p_min, E * 1.5))
                if E < E_min:
                    tools.logprint('Minimum exposure reached before correctly '
                                   'exposed. Subject too dim.', 'red')
                    return None
                elif E > E_max:
                    tools.logprint('Maximum exposure reached before correctly '
                                   'exposed. Subject too bright.', 'red')
                    return None
                else:
                    tools.logprint(f'Trying exposure = {np.round(E,4)}')
                    self.instrument.set_frame_period(p)
                    self.instrument.set_exposure(E)
                    testframe = self.take_images(10, True, False)
                    diff = np.abs(np.max(testframe, axis=(0, 1)) - target)
                    # Check for overexposure
                    if testframe.max() >= 254:
                        dE = 0.5
                    elif diff.min() <= margin: # Check exposure of different channels
                        correctly_exposed = True
                    else:
                        dE = target / testframe.max()
                    tools.logprint(f'dE= {dE}')

            settings = self.get_settings(False)
            E = np.round(settings['exposure'] * 1000, 2)
            p = np.round(settings['frame_period'], 2)
            tools.logprint(f'Auto exposure yielded exposure {E}ms and '
                           f'frame period {p}s.')
            if show:
                self.take_images(5, True, True)

    def auto_roi(self, mode='peak', width=50, show=True):
        valid_modes = ['reset', 'peak']
        xmax = 1280
        ymax = 1024
        tools.logprint('Selecting ROI...')
        if self.testflight:
            return None
        else:
            if mode not in valid_modes:
                raise ValueError(f'Mode should be in {valid_modes}.')
            roi = (0, xmax, 0, ymax, 1, 1)
            self.set_settings({'roi': roi}, False)
            if mode == 'peak':
                testframe = self.take_images(3, True, False)[0]
                index = np.unravel_index(
                    testframe.argmax(), testframe.shape)
                roi = (index[0]-width, index[0]+width,
                       index[1]-width, index[1]+width,
                       1, 1)
                roi = np.clip(roi, 0, [xmax, xmax, ymax, ymax, 1, 1])
                roi = tuple(roi)
                self.set_settings({'roi': roi}, False)
                tools.logprint(f'ROI set to {roi}.')
            if show:
                self.take_images(5, True, True)

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

    def set_settings(self, settings={}, print=True, test_img=True):
        if not type(settings) is dict:
            raise TypeError('Variable "settings" should be of type dict.')
        s = self.instrument.apply_settings(settings)
        if print:
            self.get_settings(True)
        if test_img:
            self.take_images(nframes=1, show=False)

    def take_images(self, nframes=20, median=True, show=True):
        if self.testflight:
            img = self._random_image(nframes)
        else:
            frame_period = self.instrument.get_frame_timings()[1]
            max_TO = frame_period * 2 + 1.0
            self.instrument.start_acquisition()
            self.instrument.wait_for_frame(
                nframes = nframes, timeout=(max_TO * nframes, max_TO))
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
                im = ax[i].imshow(channel, cmap=maps[i])
                ax[i].set_title(title)
                # Colorbar stuff
                divider = make_axes_locatable(ax[i])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im, cax=cax, orientation='vertical')
            fig.tight_layout()
            plt.show()
            plt.imshow(frame)
            plt.title('RGB image')
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

    def __repr__(self):
        info = self.instrument.get_device_info()
        return f'UC480 camera with name {self.name}. \nDevice info:\n{info}'

    def __del__(self):
        tools.logprint('Disconnecting camera', 'yellow')
        if not self.testflight:
            self.instrument.close()


if __name__ == '__main__':

    cam = Cam(testflight=True)
    #tools.print_dict(cam.instrument.get_full_info())
    cam.get_settings()
    img = cam.take_images(5, show=True, median=True)
    cam.auto_expose()
    cam.auto_roi()
