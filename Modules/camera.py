from pylablib.devices import uc480
import matplotlib.pyplot as plt
import numpy as np
try:
    import Modules.tools as tools
except:
    import tools
import atexit

class Cam():
    def __init__(self, testflight = False):
        atexit.register(self.shutdown)
        self.testflight = testflight

        if testflight:
            tools.logprint('*** Initializing fictional connection with camera ***')
        else:
            tools.logprint('*** Initializing connection with camera ***')
            self.instrument = uc480.UC480Camera(cam_id = 0) #Connect to first available camera

            if self.instrument.is_opened():
                tools.logprint('Connected to camera succesfully.')
            else:
                raise IOError('Camera not connected')

    def shutdown(self):
        print('Closing camera')
        if not self.testflight:
            self.instrument.close()

    def get_settings(self):
        if not self.testflight:
            print(self.instrument.get_settings())

    def take_images(self,nframes = 1, median=True, show=False):
        if self.testflight:
            img = self.random_image(nframes)
        else:
            img = self.instrument.grab(nframes)

        if median and nframes > 1:
            tools.logprint('Taking median of captured images.')
            img = np.median(img,axis=0)

        if show:
            try:
                plt.imshow(img,cmap='cividis')
            except TypeError: #imshow can only plot 1 image
                plt.imshow(img[-1],cmap='cividis') #plot last image
            plt.axis('off')
            plt.show()

        return img

    def random_image(self,nframes = 1):
        img = []
        for frame in range(nframes):
            seed = sorted( np.random.randint(-20,20,2) )
            x = y = np.linspace(seed[0],seed[1],500)
            X,Y = np.meshgrid(x,y)
            f = np.sinc(np.hypot(X,Y))
            img.append(f)

        return(img)

if __name__ == '__main__':

    cam = Cam(testflight=True)
    img = cam.take_images(100,show=True,median=False)
