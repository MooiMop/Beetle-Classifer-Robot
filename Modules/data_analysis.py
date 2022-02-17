import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from tqdm import tqdm
from numba import jit, njit
import time

try:
    import Modules.tools as tools
    from Modules.HDF5manager import HDF5
except ModuleNotFoundError:
    import tools
    from HDF5manager import HDF5

@njit
def get_top_10percent(array, errors):
    args = np.argsort(array.flatten())
    N = int(len(array) * 0.1)
    output1 = array.flatten()[args]
    output2 = errors.flatten()[args]
    return output1[-N:], output2[-N:]

@njit
def calc_stokes_frames(input_data, input_data_err, angles):
    # assume dataset of shape [reflection angles, polarizer_angles, x, y]
    input_shape = input_data.shape
    R = input_shape[0]
    X = input_shape[2]
    Y = input_shape[3]
    output_shape = (R, X, Y, 4)
    stokes_frames = np.zeros(shape=output_shape)
    error = np.zeros(shape=output_shape)
    angles = angles / 180 * np.pi
    for r in range(R):
        for x in range(X):
            for y in range(Y):
                intensity = input_data[r, :, x, y]
                err = input_data_err[r, :, x, y]
                stokes_frames[r, x, y], error[r,x,y] = calc_stokes_params(intensity, err, angles)
    return stokes_frames, error

@njit
def calc_stokes_params(intensity, intensity_err, angles):
    N = len(angles)
    A = 1 / N * np.sum(intensity)
    B = 1 / N * np.sum(intensity * np.sin(2 * angles))
    C = 1 / N * np.sum(intensity * np.cos(4 * angles))
    D = 1 / N * np.sum(intensity * np.sin(4 * angles))
    S0 = 2 * (A - C)
    S1 = 4 * C
    S2 = 4 * D
    S3 = -2 * B
    sum1 = S1**2 + S2**2 + S3**2
    sum2 = S0**2

    if  sum1 > sum2:
        stokes_vector = [0.0, 0.0, 0.0, 0.0]
        stokes_err = [0.0, 0.0, 0.0, 0.0]
    else:
        stokes_vector = [S0, S1, S2, S3]
        # error calculation
        A_err = np.mean(intensity_err)
        B_err = 1 / N * np.sqrt(np.sum((intensity_err * np.sin(2 * angles))**2))
        C_err = 1 / N * np.sqrt(np.sum((intensity_err * np.cos(4 * angles))**2))
        D_err = 1 / N * np.sqrt(np.sum((intensity_err * np.sin(4 * angles))**2))
        S0_err = np.sqrt(A_err**2 + C_err**2)
        S1_err = 2 * C_err
        S2_err = 2 * D_err
        S3_err = B_err
        stokes_err = [S0_err, S1_err, S2_err, S3_err]
    return stokes_vector, stokes_err

@njit
def stokes_frames_normalize(stokes_frames, errors):
    # assume input of shape [reflection angles, x, y, 4]
    R = stokes_frames.shape[0]
    X = stokes_frames.shape[1]
    Y = stokes_frames.shape[2]
    output = np.zeros(shape=stokes_frames.shape)
    output_err = np.zeros(shape=stokes_frames.shape)
    for r in range(R):
        for x in range(X):
            for y in range(Y):
                norm = stokes_frames[r, x, y, 0]
                for i in range(4):
                    if norm == 0:
                        output[r, x, y, i] *= 0
                        output_err[r, x, y, i] *= 0
                    else:
                        output[r, x, y, i] = stokes_frames[r, x, y, i] / norm
                        output_err[r, x, y, i] = errors[r, x, y, i] / norm
    return output, output_err

@njit
def calc_DOP(stokes_frame):
    numerator = np.sqrt(np.sum(stokes_frame[:,:,1:]**2, axis=-1))
    denominator = stokes_frame[:, :, 0]
    output = numerator / denominator
    return output

# Useful functions when inspecting a HDF5 file or a complex frame array.
# Can be used without initializing DataAnalyzer
def frame_overview(arr):
    print(f'Shape:  {arr.shape}')
    print(f'Min:  {np.min(arr, axis=(0,1))}')
    print(f'Max:  {np.max(arr, axis=(0,1))}')
    print(f'Mean:  {np.mean(arr, axis=(0,1))}')
    print(f'Median:  {np.median(arr, axis=(0,1))}')

class DataAnalyzer():

    def __init__(self, datafilepath=None, force_calculation=False,
        verbose=True):
        # Get filename and set working directory
        if datafilepath == None:
            datafilepath = input('(Full) path to HDF5 file to consider: ')
        self.filename = os.path.basename(datafilepath)
        os.chdir(os.path.dirname(datafilepath))
        self.verbose = verbose

        # Open datafile using custom HDF5 module
        self.f = HDF5(self.filename, 'open', verbose=verbose)
        if self.verbose:
            self.f.print_structure()

        # Start correction sequence if file is raw file.
        if (not self.f.contains_key('Frames_err')) or force_calculation:
            if self.verbose:
                    tools.logprint('Correcting raw data and creating new file.')
            self.correction_sequence()

        if (not self.f.contains_key('Stokes_vectors')) or force_calculation:
            for mode in ['single frame', 'rgb' , None]:
                self.generate_stokes_frames(mode)

    # Functions for data reduction sequences
    def correction_sequence(self):
        # Create subdirectory for outputting files
        dir = self.filename[:-5] + ' corrected'
        try:
            os.chdir(dir)
        except FileNotFoundError:
            os.mkdir(dir)
            os.chdir(dir)

        with HDF5(dir, 'create', date=False, verbose=self.verbose) as corrected:
            # Transfer high-level metadata
            metadata = HDF5.read_metadata(self.f.file)
            HDF5.write_metadata(corrected.file, metadata)

            # Recreate group structure with for loop
            for groupname in self.f.file:
                group = self.f.file[groupname]
                try:
                    dset = group['Frames combined']
                except KeyError:
                    if self.verbose:
                            tools.logprint(
                                'Combining individual runs to single dataset.')
                    self.f.combine_datasets(parent=groupname)
                    dset = group['Frames combined']

                # Combine group and dataset metdata and write to new group
                metadata = HDF5.read_metadata(group) | HDF5.read_metadata(dset)
                corrected.group(groupname, metadata=metadata)

                # Set shape of corrected frames output
                reflection_angles = dset.shape[1]
                output_shape = dset.shape[1:]
                corrected_frames = np.zeros(output_shape)
                error = np.zeros(output_shape)
                for i in tqdm(range(reflection_angles)):
                    data = dset[:, i]

                    # Dark frame correction
                    try:
                        dark_frame = group['dark frame 0'][:]
                        dark_frame = np.median(dark_frame, axis=0)
                        data = data - dark_frame
                    except KeyError:
                        None

                    # Exposure correction
                    exposure = dset.attrs['exposure']
                    data = data / exposure

                    # Calculate final value and error
                    error[i] = np.std(data, axis=0) / np.sqrt(len(data))
                    corrected_frames[i] = np.mean(data, axis=0)

                # Create new datasets
                try:
                    angles_out = dset.attrs['angles_out']
                except KeyError:
                    angles_out = dset.attrs['positions']
                polarizer_angles = dset.attrs['polarizer_angles']
                corrected.create_dataset(
                    'Frames', corrected_frames, groupname, overwrite=True)
                corrected.create_dataset(
                    'Frames_err', error, groupname, overwrite=True)
                corrected.create_dataset(
                    'angles_out', angles_out, groupname, overwrite=True)
                corrected.create_dataset(
                    'polarizer_angles', polarizer_angles, groupname,
                    overwrite=True)

        self.filename = dir
        self.f = HDF5(self.filename, 'open', verbose=False)

    def generate_stokes_frames(self, mode=None):
        if self.verbose:
            tools.logprint(
                f'Generating Stokes Frames (mode={mode}).')
        for group_name in self.f.file:
            group = self.f.group(group_name)
            angles = group['polarizer_angles'][:]
            frames = group['Frames'][:]
            frames_err = group['Frames_err'][:]
            if mode == 'rgb':
                data = []
                data_err = []
                norm = []
                norm_err = []
                for channel in range(3):
                    selection = frames[:,:,:,:,channel]
                    selection2 = frames_err[:,:,:,:,channel]
                    d, e = calc_stokes_frames(selection, selection2, angles)
                    n, n_e = stokes_frames_normalize(d, e)
                    data.append(d)
                    norm.append(n)
                    data_err.append(e)
                    norm_err.append(n_e)
                stokes_frames = np.transpose(data, (1,2,3,0,4))
                error = np.transpose(data_err, (1,2,3,0,4))
                norm = np.transpose(norm, (1,2,3,0,4))
                norm_err = np.transpose(norm_err, (1,2,3,0,4))
                name = 'Stokes_frames_rgb'
            elif mode == 'single frame':
                stokes_frames = []
                error = []
                for channel in range(3):
                    data = []
                    data_err = []
                    f = frames[:,:,:,:,channel]
                    fe = frames_err[:,:,:,:,channel]

                    for i in range(len(frames)):
                        selection = f[i]
                        selection2 = fe[i]

                        # Correct for overexposure by setting those pixels to 0
                        mask = selection != 255
                        selection = selection * mask
                        selection2 = selection * mask

                        ## Get top 10 percent of pixel values
                        #input = []
                        #input_err = []
                        #for i in range(len(selection)):
                        #    top, err = get_top_10percent(selection[i], selection2[i])
                        #    input.append(np.sum(top))
                        #    input_err.append(np.sum(input_err))

                        #input = np.array(input)
                        #input_err = np.array(input_err)

                        input = selection.sum(axis=(1,2))
                        input_err = selection2.sum(axis=(1,2))

                        s, e = calc_stokes_params(input, input_err, angles)
                        data.append(s)
                        data_err.append(e)
                    stokes_frames.append(data)
                    error.append(data_err)
                stokes_frames = np.transpose(stokes_frames, (1, 0, 2))
                error = np.transpose(error, (1, 0, 2))
                name ='Stokes_vectors'
            else:
                frames = frames.sum(axis=-1)
                frames_err = frames_err.sum(axis=-1)
                t = time.time()
                stokes_frames, error = calc_stokes_frames(frames, frames_err, angles)
                norm, norm_err = stokes_frames_normalize(stokes_frames, error)
                name = 'Stokes_frames'

            self.f.create_dataset(
                name, stokes_frames, group_name, overwrite=True)
            self.f.create_dataset(
                name+'_error', error, group_name, overwrite=True)
            if mode != 'single frame':
                self.f.create_dataset(
                    name+'_normalized', norm, group_name, overwrite=True)
                self.f.create_dataset(
                    name+'_normalized_error', norm_err, group_name, overwrite=True)

    def _mask_overexposure(self, frame):
        mask = frame != 255
    # Plotting functions
    def single_frames_generator(self, name=None):
        for group_name in self.f.file:
            group = self.f.group(group_name)
            angles_out = group['angles_out'][:]
            angles = group['polarizer_angles'][:]
            for angle in angles_out:
                selection = np.argmax(angles_out == angle)
                frame = group['Frames'][selection]
                stokes_frame = group['Stokes_frames'][selection]
                stokes_frame_norm = group['Stokes_frames_normalized'][selection]
                err = group['Frames_err'][selection]
                stokes_err = group['Stokes_frames_error'][selection]
                name = f'Frame overview - {group_name[:2]}->{angle}.png'
                DataPlotter.single_frame(frame, stokes_frame, stokes_frame_norm,
                    angles, err, stokes_err, name)

    def stokes_vs_angle_generator(self, name=None):
        for group_name in self.f.file:
            group = self.f.group(group_name)
            stokes_vectors = group['Stokes_vectors'][:]
            stokes_err = group['Stokes_vectors_error'][:]
            angles = group['angles_out'][:]
            DataPlotter.stokes_vs_angle(
                stokes_vectors, angles, stokes_err, group_name)

class DataPlotter():
    def argmax2d(frame):
        index = np.unravel_index(frame.argmax(), frame.shape)
        return index

    def highlight_cell(x,y, width=10, ax=None, **kwargs):
        rect = plt.Rectangle((x-int(width/2), y-int(width/2)), width, width, fill=False, **kwargs)
        ax = ax or plt.gca()
        ax.add_patch(rect)
        return rect

    def intensity_curve(stokes_vector):
        # Generate expected intensity curve with given Stokes vector
        theta = np.linspace(0, np.pi, 100)
        I0 = stokes_vector[0]
        I1 = stokes_vector[1] * np.cos(2 * theta)**2
        I2 = stokes_vector[2] * np.cos(2 * theta) * np.sin(2 * theta)
        I3 = stokes_vector[3] * np.sin(2 * theta) * -1
        I = 0.5 * (I0 + I1 + I1 + I3)
        theta = theta / np.pi * 180
        return I, theta

    def single_frame(frame, stokes_frame, stokes_frame_norm, angles,
        err=None, stokes_err=None, name=None):
        # Assumes input frames are of shape [P, x, y, 3] where P is the number
        # of polarizer angles

        # Select brightest part
        arg = DataPlotter.argmax2d(frame)[0:3]
        pixel = frame[:, arg[1], arg[2]].sum(axis=-1)
        stokes_vector = stokes_frame[arg[1], arg[2]]
        if err is not None:
            pixel_err = err[:, arg[1], arg[2]].sum(axis=-1)
        if stokes_err is not None:
            stokes_vector_err = stokes_err[arg[1], arg[2]]

        # start plotting
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15,10))

        # RGB image
        RGB_img = frame[arg[0]]
        RGB_img = RGB_img / RGB_img.max()
        #axes[0,0].imshow(np.clip(RGB_img, 0, 50)/50)
        axes[0,0].imshow(RGB_img)
        axes[0,0].set_title('RGB image')

        # Image of degree op polarization
        P = calc_DOP(stokes_frame)
        im = axes[0,1].imshow(P, vmin=0, vmax=1, cmap='hot')
        fig.colorbar(im, ax=axes[0,1])
        axes[0,1].set_title('Degree of polarization')

        # Images of Stokes paramaters
        mini = np.min(stokes_frame_norm[:, :, 1:])
        maxi = np.min(stokes_frame_norm[:, :, 1:])
        v = np.abs(np.max([mini,maxi]))
        im1 = axes[1,0].imshow(stokes_frame_norm[:, :, 1], vmin=-v, vmax=v, cmap='bwr')
        im2 = axes[1,1].imshow(stokes_frame_norm[:, :, 2], vmin=-v, vmax=v, cmap='bwr')
        im3 = axes[2,0].imshow(stokes_frame_norm[:, :, 3], vmin=-v, vmax=v, cmap='bwr')
        axes[1,0].set_title(f'S1')
        axes[1,1].set_title(f'S2')
        axes[2,0].set_title(f'S3')
        fig.colorbar(im1, ax=axes[1,0])
        fig.colorbar(im2, ax=axes[1,1])
        fig.colorbar(im2, ax=axes[2,0])
        DataPlotter.highlight_cell(arg[2], arg[1], 30, ax=axes[2,0], color='green', linewidth=2)

        # Image of Fourier analysis Stokes paramater calculation
        if err is not None:
            axes[2,1].errorbar(angles, pixel, yerr=pixel_err, label='Measurement')
        else:
            axes[2,1].plot(angles, pixel, label='Measurement')
        I, theta = DataPlotter.intensity_curve(stokes_vector)
        axes[2,1].plot(theta, I, label='Expected', c='red')
        if stokes_err is not None:
            upper, dump= DataPlotter.intensity_curve(stokes_vector+stokes_vector_err)
            lower, dump= DataPlotter.intensity_curve(stokes_vector-stokes_vector_err)
            axes[2,1].fill_between(
                theta, upper, lower, facecolor='red', alpha=0.3,
                label=r'Error interval')
        axes[2,1].legend()
        axes[2,1].set_title('Fit of Stokes parameters')


        if name is None:
            plt.show()
        else:
            fig.suptitle(name)
            fig.savefig(name+'.png', dpi=300)
            plt.close(fig)

    def stokes_vs_angle(stokes_vectors, angles, stokes_err=None, name=None):
        fig, ax = plt.subplots(nrows=4, sharex=True, figsize=(5,10))
        colors = ['red', 'green', 'blue']
        markers = ['v', 'D', 'o']

        # Plot each component of S in a different ax
        norm = np.max(stokes_vectors[:, :, 0], axis=1)
        for S in range(4):
            for i, color in enumerate(colors):
                if S > 0:
                    stokes_vectors[:, i, S] = stokes_vectors[:, i, S] / norm
                    stokes_err[:, i, S] = stokes_err[:, i, S] / norm
                    ax[S].set_ylabel(f'S{S} (Normalized)')
                if S < 3:
                    ax[S].set_xticklabels([])
                if S == 3:
                    ax[S].set_xlabel('Angle out (degrees)')
                ax[S].errorbar(
                    angles, stokes_vectors[:, i, S],
                    yerr=stokes_err[:, i, S], ls = ':',
                    label=f'{color} channel', color=color, marker=markers[i])
            ax[S].set_ylabel(f'S{S}')
        ax[0].legend()
        fig.tight_layout()
        if name is None:
            plt.show()
        else:
            fig.suptitle(name)
            fig.savefig(name+'.png', dpi=300)
            plt.close(fig)

if __name__ == '__main__':
    path = '/Users/naorscheinowitz/Master Project/Beetle Project/Beetle Classifier Robot/Experiments/Beetle Hyperspectral/Beetle reflection spot corrected/Beetle reflection spot corrected.hdf5'
    DA = DataAnalyzer(path, verbose=False, force_calculation=False)
    DA.generate_stokes_frames(mode='single frame')
    DA.single_frames_generator()
