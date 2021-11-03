import atexit
import matplotlib.pyplot as plt
import numpy as np

from pylablib.devices import uc480

try:
    import Modules.tools as tools
except ModuleNotFoundError:
    import tools


class Cam():

    def __init__(self, testflight=False):
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

        if testflight:
            tools.logprint(
                tools.bcolors.blue(
                    'Initializing fictional connection with camera'))
        else:
            tools.logprint(
                tools.bcolors.blue(
                    'Initializing connection with camera'))
            try:
                # Connect to first available camera
                self.instrument = uc480.UC480Camera(cam_id=0)
                # Next 4 lines might be unnecessary
                if self.instrument.is_opened():
                    tools.logprint(
                        tools.bcolors.green(
                            'Connected to camera succesfully.'))
                else:
                    raise IOError(
                        tools.bcolors.red(
                            'Camera not connected'))
            except OSError:
                self.testflight = True

    def shutdown(self):
        tools.logprint('Disconnecting camera')
        if not self.testflight:
            self.instrument.close()

    def get_settings(self):
        if not self.testflight:
            print(self.instrument.get_settings())

    def take_images(self, nframes=1, median=True, show=False):
        if self.testflight:
            img = self._random_image(nframes)
        else:
            img = self.instrument.grab(nframes)

        if median and nframes > 1:
            tools.logprint('Taking median of captured images.')
            img = np.median(img, axis=0)

        if show:
            try:
                plt.imshow(img)
            except TypeError:  # imshow can only plot 1 image
                plt.imshow(img[-1])  # plot last image
            plt.show()

        return img

    def _random_image(self, nframes=1):
        img = []
        for frame in range(nframes):
            # seed = sorted( np.random.randint(-2,2,2) )
            x = y = np.linspace(-5, 5, 500)
            X, Y = np.meshgrid(x, y)
            noise = 2 * np.random.random((500, 500)) - 1
            f = np.sinc(np.hypot(X, Y)) + noise
            img.append(f)
            
        return(img)


if __name__ == '__main__':

    cam = Cam(testflight=True)
    img = cam.take_images(50, show=True, median=True)
