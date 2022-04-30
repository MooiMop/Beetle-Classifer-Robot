'''Data analysis for HDF5 files produced by BCR module.

Analysis and plotting is handled by the two classes. To improve processing
speed, separate numba-accelerated functions are implemented.

Classes
-------
    DataAnalyzer
    DataPlotter

Functions (numba accelerated)
-----------------------------
    calc_DOP
    calc_stokes_frames
    calc_stokes_params
    exposure_correction
    stokes_frames_normalize

Function (misc)
---------------
    frame_overview

Constants
---------
    MAX_PIXEL_VALUE
    MIN_PIXEL_VALUE
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py
import os
from tqdm import tqdm, trange
from numba import njit
import time
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    import Modules.tools as tools
    from Modules.HDF5manager import HDF5
except ModuleNotFoundError:
    import tools
    from HDF5manager import HDF5

# Constants
MAX_PIXEL_VALUE = 255
MIN_PIXEL_VALUE = 0
CUTOFF = 3

# Numba accelerated functions
@njit
def calc_DOP(stokes_frame):
    '''Calculate degree of polarization of each pixel in frame.'''
    numerator = np.sqrt(np.sum(stokes_frame[:, :, 1:]**2, axis=-1))
    denominator = stokes_frame[:, :, 0]
    output = numerator / denominator
    return output


@njit
def calc_stokes_frames(frame, frame_err, angles):
    '''Calculate stokes vector for each pixel in frame.

    Parameters
    ----------
    frame : float
        Numpy array of shape [reflection angles, polarizer_angles, x, y] with
        (corrected) pixel count values.
    frame_err : float
        Numpy array of same shape as frame with (corrected) pixel count errors.
    angles : int
        List or numpy array with values of wave-plate rotation angle.

    Returns
    -------
    stokes_frames : float
        Numpy array of shape [reflection_angles, x, y, 4] of pixel-specific
        Stokes vectors.
    error : float
        Numpy array of same shape as stokes_frames with error values for each
        Stokes vector.
    '''
    input_shape = frame.shape
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
                if frame[r, :, x, y].mean() <= CUTOFF:
                    stokes_frames[r, x, y] = [0, 0, 0, 0]
                    error[r, x, y] = [0, 0, 0, 0]
                else:
                    intensity, err = frame[r, :, x, y], frame_err[r, :, x, y]
                    vec, err = calc_stokes_params(intensity, err, angles)
                    stokes_frames[r, x, y], error[r, x, y] = vec, err
    return stokes_frames, error


@njit
def calc_stokes_params(intensity, intensity_err, angles):
    '''Calculate single stokes vector from series of wave-plate angle dependent
    intensities.

    This function checks if the calculated values make sense by checking if the
    squared sum of parameters S1-S3 is larger than S0. If this is the case, it
    outputs a Stokes vector consisting of zeros.

    Parameters
    ----------
    intensity : list float
        (Numpy) list with intensity values.
    intensity_err : list float
        (Numpy) list with intensity errors.
    angles : list float
        (Numpy) list with values of wave-plate rotation angle (radians or
        degrees).

    Returns
    ------
    stokes_vector : list float
        List with Stokes 4-vector.
    error  : list float
        List with Stokes 4-vector errors.
    '''
    # Convert degrees to radians if necessary
    convert = angles.max() > 10.0
    if convert:
        x = angles[1:] / 180 * np.pi
    else:
        x = angles[1:]
    y = intensity[1:]
    y[-1] = (intensity[0] + intensity[-1]) / 2
    yerr = intensity_err[1:]
    yerr[-1] = (intensity_err[0] + intensity_err[-1]) / 2

    N = len(angles) - 1
    A = 2 * np.mean(y)
    B = (y * np.sin(2 * x)).sum() / N * 4
    C = (y * np.cos(4 * x)).sum() / N * 4
    D = (y * np.sin(4 * x)).sum() / N * 4
    S0 = (A - C)
    S1 = 2 * C
    S2 = 2 * D
    S3 = -1 * B

    # Squared sum inequality check.
    sum1 = S1**2 + S2**2 + S3**2
    sum2 = S0**2
    if sum1 > sum2:
        stokes_vector = [0.0, 0.0, 0.0, 0.0]
        error = [0.0, 0.0, 0.0, 0.0]
    else:
        stokes_vector = [S0, S1, S2, S3]

        # error calculation
        A_err = np.sqrt(np.sum(yerr**2)) / N * 2
        B_err = np.sqrt(np.sum((yerr * np.sin(2 * x))**2)) / N * 4
        C_err = np.sqrt(np.sum((yerr * np.cos(4 * x))**2)) / N * 4
        D_err = np.sqrt(np.sum((yerr * np.sin(4 * x))**2)) / N * 4
        S0_err = np.sqrt(A_err**2 + C_err**2)
        S1_err = 2 * C_err
        S2_err = 2 * D_err
        S3_err = B_err
        error = [S0_err, S1_err, S2_err, S3_err]
    return stokes_vector, error

@njit
def stokes_frames_normalize(stokes_frames, errors):
    '''Normalize stokes vectors in data cube such that S0 is always 1.

    Parameters
    ----------
    stokes_frames : float
        Numpy array with Stokes vectors of shape
        [reflection angles, x, y, 4].
    errors : float
        Numpy array with Stokes vector errors of shape
        [reflection angles, x, y, 4]

    Returns
    -------
    output : float
        Numpy array with normalized Stokes vectors of shape
        [reflection angles, x, y, 4]
    error : float
        Numpy array with normalized errors of shape
        [reflection angles, x, y, 4]
    '''
    shape = stokes_frames.shape
    R, X, Y = shape[:3]
    output = np.zeros(shape)
    error = np.zeros(shape)
    for r in range(R):
        for x in range(X):
            for y in range(Y):
                norm = stokes_frames[r, x, y, 0]
                for s in range(4):
                    if norm == 0:
                        # Set entire vector to zero. This is to prevent divided
                        # by zero problems.
                        output[r, x, y, s] *= 0
                        error[r, x, y, s] *= 0
                    else:
                        output[r, x, y, s] = stokes_frames[r, x, y, s] / norm
                        error[r, x, y, s] = errors[r, x, y, s] / norm
    return output, error

# Misc functions
def frame_overview(arr):
    '''Outputs useful quantities of numpy array of single data frame.'''
    print(f'Shape:  {arr.shape}')
    print(f'Min:  {np.min(arr, axis=(0,1))}')
    print(f'Max:  {np.max(arr, axis=(0,1))}')
    print(f'Mean:  {np.mean(arr, axis=(0,1))}')
    print(f'Median:  {np.median(arr, axis=(0,1))}')


class DataAnalyzer():
    '''
    A class that represents a HDF5 data file.

    Upon initialization, the class will open a HDF5 file. If it detects that
    its a raw datafile, it will create a new HDF5 in a new subfolder with
    corrected frames. Likewise, if no HDF5 datasets are detected with Stokes
    vectors, it will calculate and generate these.

    Attributes
    ----------
    f : HDF5 object
        instance of custom HDF5 class that contains h5py instance of HDF5 file
    filename : str
        filename of the HDF5 file
    verbose : bool
        whether or not to print comments during execution

    Methods
    -------
    correction_sequence():
        Creates new HDF5 file with corrected data.
    generate_stokes_frames(mode):
        Creates a HDF5 dataset filled with frames of Stokes vectors.
    single_frames_generator():
        Generates a plot of each Stokes vector as a seperately colored frame.
    stokes_vs_angle_generator():
        Generates a plot of Stokes parameter vs reflection angle.
    '''

    def __init__(self, path=None, force_calculation=False, verbose=True):
        '''Initialize DataAnalyzer object.

        Opens a HDF5 file in write mode using custom HDF5 class from
        Modules.HDF5manager. If the file does not contain a dataset called
        'Frames_err', a new file is created and data correction is started.
        If the file does not contain a dataset called 'Stokes_vectors',
        calculation of the Stokes vectors is started.

        Parameters
        ----------
        path : str
            (full) path to HDF5 file
        force_calculation : bool
            if true, perform calculation regardless of existance of datasets
        verbose : bool
            whether or not to print comments during execution
        '''
        # Get filename and set working directory
        if path is None:
            path = input('(Full) path to HDF5 file to consider: ')
        self.file = os.path.realpath(path)
        self.filename = os.path.basename(self.file)
        os.chdir(os.path.dirname(self.file))
        self.verbose = verbose

        # Open datafile and check which datasets it contains
        with HDF5(self.file, 'read only', verbose=False) as f:
            if self.verbose:
                f.print_structure()
            keys = list(f.file.keys())
            raw = not f.contains_key('Frames_err')
            no_stokes = not f.contains_key('Stokes_vectors')

        # Check if the corrected file already exists.
        corrected_path = os.path.join(
            os.getcwd(),
            self.filename[:-5] + ' corrected',
            self.filename[:-5] + ' corrected.hdf5')
        corrected_exists = os.path.exists(corrected_path)

        # Correct raw frames
        if raw:
            if not corrected_exists or force_calculation:
                self.correction_sequence()
            elif corrected_exists:
                if self.verbose:
                    tools.logprint(
                    'Corrected file already exists. Switching to it.'
                    )
                self.file = os.path.realpath(corrected_path)
                self.filename = os.path.basename(self.file)
                os.chdir(os.path.dirname(self.file))

        if no_stokes or force_calculation:
            for mode in ['single frame', None]:
                self.generate_stokes_frames(mode)

    def correction_sequence(self):
        '''Convert raw data into corrected frames with error values.

        Uses 'with/as' form to create a new HDF5 file and write corrected
        data to it. After all the calculation, the raw data file is closed and
        replaced with the new one, such that self.f points to the new file.
        '''
        if self.verbose:
            tools.logprint('Correcting raw data and creating new file.')

        # Create subdirectory for outputting files
        dir = self.filename[:-5] + ' corrected'
        try:
            os.chdir(dir)
        except FileNotFoundError:
            os.mkdir(dir)
            os.chdir(dir)

        with HDF5(self.file, 'read only', verbose=False) as raw:
            with HDF5(dir, 'create', verbose=self.verbose, date=False) as corrected:
                # Transfer high-level metadata
                metadata = HDF5.read_metadata(raw.file)
                HDF5.write_metadata(corrected.file, metadata)

                # Recreate group structure with for loop
                for groupname in raw.file:
                    group = raw.file[groupname]
                    try:
                        dset = group['Frames combined']
                    except KeyError:
                        raw.combine_datasets(parent=groupname)
                        dset = group['Frames combined']

                    # Combine group and dataset metadata and write to new group
                    metadata = HDF5.read_metadata(group) | HDF5.read_metadata(dset)
                    corrected.group(groupname, metadata=metadata)

                    # Set shape of corrected frames output
                    reflection_angles = dset.shape[1]
                    output_shape = dset.shape[1:]
                    corrected_frames = np.zeros(output_shape)
                    error = np.zeros(output_shape)

                    pbar = trange(reflection_angles)
                    for i in pbar:
                        pbar.set_description("Loading data")
                        data = dset[:, i]  # zero-th dimension is the number of
                                           # repeats of the experiment
                        try:
                            dark_frame = group['dark frame 0'][:]
                            pbar.set_description('Subtracting dark frames')
                            dark_frame = np.median(dark_frame, axis=0).astype(int)
                            data = data - dark_frame
                            data = np.clip(data, MIN_PIXEL_VALUE, MAX_PIXEL_VALUE)
                        except KeyError:
                            None

                        pbar.set_description('Calculating errors')
                        error[i] = np.std(data, axis=0) / np.sqrt(len(data))
                        pbar.set_description('Calculating mean pixel values')
                        corrected_frames[i] = np.median(data, axis=0)

                    # Create new datasets
                    try:  # older versions used different key names for angle_out
                        angles_out = dset.attrs['angles_out']
                    except KeyError:
                        angles_out = dset.attrs['positions']
                    polarizer_angles = dset.attrs['polarizer_angles']

                    names = ['Frames', 'Frames_err', 'angles_out',
                             'polarizer_angles']
                    datas = [corrected_frames, error, angles_out,
                             polarizer_angles]
                    for n, d in zip(names, datas):
                        corrected.create_dataset(n, d, groupname, overwrite=True)

                self.filename = os.path.basename(corrected.filename)
                self.file = os.path.realpath(self.filename)

    def generate_stokes_frames(self, mode=None):
        '''Generate stokes frames using one of three different mode and save
        to new dataset in self.f.

        Parameters
        ----------
        mode : str
            selects mode to calculate stokes frames, can be set to:
                'rgb':
                    Calculate a stokes for every pixel in every frame,
                    seperated by rgb color channel.
                'single frame':
                    Calculate a single stokes vector for an entire frame by
                    summing all pixel count values for each frame. This
                    mode does calculate a stokes vector for each color
                    channel separately.
                None:
                    Calculate a stokes for every pixel in every frame.
            For modes 'rgb' and None, a seperate dataset with normalized
            stokes vectors is created.
        '''
        with HDF5(self.file, 'open', verbose=False) as f:
            if self.verbose:
                tools.logprint(
                    f'Generating Stokes Frames (mode={mode}).')
            for group_name in f.file:
                group = f.group(group_name)
                angles = group['polarizer_angles'][:]
                frames = group['Frames'][:]
                frames_err = group['Frames_err'][:]

                if mode == 'rgb':
                    data = []
                    data_err = []
                    norm = []
                    norm_err = []
                    for channel in range(3):
                        selection = frames[:, :, :, :, channel]
                        selection2 = frames_err[:, :, :, :, channel]
                        d, e = calc_stokes_frames(selection, selection2, angles)
                        n, n_e = stokes_frames_normalize(d, e)
                        data.append(d)
                        norm.append(n)
                        data_err.append(e)
                        norm_err.append(n_e)
                    stokes_frames = np.transpose(data, (1, 2, 3, 0, 4))
                    error = np.transpose(data_err, (1, 2, 3, 0, 4))
                    norm = np.transpose(norm, (1, 2, 3, 0, 4))
                    norm_err = np.transpose(norm_err, (1, 2, 3, 0, 4))
                    name = 'Stokes_frames_rgb'
                elif mode == 'single frame':
                    stokes_frames = []
                    error = []
                    for channel in range(3):
                        for i in range(len(frames)):
                            input = frames[i, :, :, :, channel].sum(axis=(1, 2))
                            err = frames_err[i, :, :, :, channel].sum(axis=(1, 2))
                            vec, vec_err = calc_stokes_params(input, err, angles)
                            stokes_frames.append(vec)
                            error.append(vec_err)
                    stokes_frames = np.reshape(stokes_frames, (len(frames), 3, 4))
                    error = np.reshape(error, (len(frames), 3, 4))
                    name = 'Stokes_vectors'
                else:
                    shape = frames.shape
                    frames, frames_err = frames.sum(axis=4), frames_err.sum(axis=4)
                    stokes_frames, error = calc_stokes_frames(
                        frames, frames_err, angles)
                    norm, norm_err = stokes_frames_normalize(
                        stokes_frames, error)
                    name = 'Stokes_frames'

                names, datas = [name, name+'_error'], [stokes_frames, error]
                if mode != 'single frame':
                    names += [name+'_normalized', name+'_normalized_error']
                    datas += [norm, norm_err]
                for n, d in zip(names, datas):
                    f.create_dataset(n, d, group_name, overwrite=True)

    def single_frames_generator(self):
        '''Generate a plot of each Stokes vector as a seperately colored frame
        for each reflection angle in every data group.
        '''
        with HDF5(self.file, 'read only', verbose=False) as f:
            for group_name in f.file:
                group = f.group(group_name)
                angles_out = group['angles_out'][:]
                angles = group['polarizer_angles'][:]
                for angle in angles_out:
                    selection = np.argmax(angles_out == angle)
                    frame = group['Frames'][selection]
                    stokes_frame = group['Stokes_frames'][selection]
                    stokes_frame_norm = group['Stokes_frames_normalized']
                    stokes_frame_norm = stokes_frame_norm[selection]
                    err = group['Frames_err'][selection]
                    stokes_err = group['Stokes_frames_error'][selection]
                    name = f'Frame overview - {group_name[:2]}--{angle}'
                    DataPlotter.single_frame(
                        frame, stokes_frame, stokes_frame_norm, angles, err,
                        stokes_err, name)

    def stokes_vs_angle_generator(self):
        '''Generate a plot of Stokes parameter vs reflection angle for each
        data group.
        '''
        with HDF5(self.file, 'read only', verbose=False) as f:
            for group_name in f.file:
                group = f.group(group_name)
                stokes_vectors = group['Stokes_vectors'][:]
                stokes_err = group['Stokes_vectors_error'][:]
                angles = group['angles_out'][:]
                DataPlotter.stokes_vs_angle(
                    stokes_vectors, angles, stokes_err, group_name)


class DataPlotter():
    '''
    A class that bundles plotting functions.

    This class has no 'programmatic' benefit other than to organize standard,
    often used plots together.

    Methods
    -------
    argmax_nd(frame):
        get the coordinate of an N-d array without flattening dimensions
    highlight_cell(x, y, width, ax, **kwargs):
        add square box to existing ax
    single_frame(frame, stokes_frame, stokes_frame_norm, angles, err=None,
                 stokes_err=None, name=None):
        plot 6 axes: rgb image, degree of polarization, S1-S3, and the Stokes
        vector fit on a single pixel
    stokes_vs_angle(stokes_vectors, angles, stokes_err=None, name=None):
        plot each Stokes vector as a function of angle of reflection separated
        by color channel
    '''

    COLORS = ['orangered', 'deepskyblue', 'yellowgreen', 'darkviolet']
    LINESTYLES = ['solid', 'dotted', 'dashed', 'dashdot']

    def argmax_nd(frame):
        '''get the coordinate of an N-d array without flattening dimensions'''
        index = np.unravel_index(frame.argmax(), frame.shape)
        return index

    def error_margin(x, upper, lower, ax=None, alpha=0.5,
            color='red', **kwargs):
        '''Plot shaded region between upper and lower.'''
        if ax is None:
            ax = plt.gca()
        shaded = ax.fill_between(
            x, upper, lower, alpha=alpha, color=color, **kwargs)
        return shaded

    def highlight_cell(x, y, width=50, ax=None, **kwargs):
        '''Add square box to existing ax.

        Parameters
        ----------
        x : int
            x-coordinate of box center
        y : int
            y-coordinate of box center
        width : int
            full width of box
        ax : matplotlib.axes object (default None)
            ax to add box to. If set to None, a new object is created or it
            is added to the open matplotlib session.
        **kwargs
            arguments to pass through to plt.Rectangle()

        Returns
        -------
        rect: plt.Rectangle() instance
        '''
        rect = plt.Rectangle(
            (x-int(width/2), y-int(width/2)), width, width,
            fill=False, **kwargs)
        ax = ax or plt.gca()
        ax.add_patch(rect)
        return rect

    def intensity_curve(stokes_vector, N=100):
        '''Generate intensity curve from Stokes vector.

        Parameters
        ----------
        stokes_vector : list int
            Stokes 4-vector
        N : int (default 100)
            y-coordinate of box center

        Returns
        -------
        intensity: list float
            intensity values
        theta: list float
            wave-plate rotation angles with dtheta = 180/N degrees
        '''
        theta = np.linspace(0, np.pi, N)
        I0 = stokes_vector[0]
        I1 = stokes_vector[1] * np.cos(2 * theta)**2
        I2 = stokes_vector[2] * np.cos(2 * theta) * np.sin(2 * theta)
        I3 = stokes_vector[3] * np.sin(2 * theta) * -1
        intensity = 0.5 * (I0 + I1 + I2 + I3)
        theta = theta / np.pi * 180
        return intensity, theta

    def stokes_image(stokes_frame_norm, S, v=None, ax=None,
            colorbar=True, cmap='bwr', **kwargs):
        '''Create color image of a stokes frame of a single Stokes parameter.
        Upper and lower bounds of color scale can be set with parameter v
        as either
            - a single value (symmetrical)
            - a 2-number list (asymmetrical)
            - None (automatic based on extreme value)
        '''
        if v is None:
            vmax = np.abs(stokes_frame_norm[:, :, S]).max()
            vmin = -1 * vmax
        elif type(v) in [np.ndarray, list]:
            if len(v) != 2:
                raise ValueError(
                    'v should be None, float or list with 2 elements.'
                )
            elif v[0] > v[1]:
                raise ValueError(
                    'v[0] should be smaller than v[1].'
                )
            else:
                vmin = v[0]
                vmax = v[1]
        else:
            vmin, vmax = -1 * v, v

        if ax is None:
            ax = plt.gca()
        fig = ax.get_figure()

        im = ax.imshow(
            stokes_frame_norm[:, :, S], vmin=vmin, vmax=vmax,
            cmap=cmap, **kwargs)

        ax.set_title('S' + str(S))
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax)
        return im

    def fourier(x, y, offset=0.0, ax=None, xlim=(0.2, 6.2), **kwargs):
        '''Create Fourier transform plot of x,y data.'''
        if ax is None:
            ax = plt.gca()
        N = len(y)
        dt = x[1] - x[0]
        transform = fft(y)
        norm = transform.max()
        transform = np.abs(transform / norm)
        freq = fftfreq(N, dt) * 360 - offset
        markerline, stemlines, baseline = ax.stem(
            freq[1:], transform[1:], markerfmt='D', **kwargs)

        # Aesthetics
        baseline.set_alpha(0.2)
        baseline.set_color('black')
        ax.set_title('Fourier Transform')
        ax.set_xlim(xlim)
        ax.set_xlabel(r'Fourier Frequency ($2\pi f$)')
        ax.set_ylabel('Fourier Amplitude')
        return transform, markerline, stemlines, baseline

    def single_frame(frame, stokes_frame, stokes_frame_norm, angles,
                     err=None, stokes_err=None, name=None):
        '''Plot 6 axes: rgb image, degree of polarization, S1-S3, and the
        Stokes vector fit on a single pixel. If name is given, plot is saved
        to current working directory. Otherwise it is shown.'''
        # Assumes input frames are of shape [P, x, y, 3] where P is the number
        # of polarizer angles

        # Select brightest part
        arg = DataPlotter.argmax_nd(frame)[0:3]
        pixel = frame[:, arg[1], arg[2]].sum(axis=-1)
        stokes_vector = stokes_frame[arg[1], arg[2]]
        if err is not None:
            pixel_err = err[:, arg[1], arg[2]].sum(axis=-1)
        if stokes_err is not None:
            stokes_vector_err = stokes_err[arg[1], arg[2]]

        # start plotting
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(13,15))

        # RGB image
        RGB_img = frame[arg[0]].astype('uint8')
        clip = RGB_img.max() * 0.8
        clipped = np.clip(RGB_img, 0, clip)
        axes[0, 0].imshow(clipped/clipped.max())
        axes[0, 0].set_title('RGB image')
        axes[0, 0].axes.xaxis.set_visible(False)
        axes[0, 0].axes.yaxis.set_visible(False)
        # Add a divider so that image fits nicely with following images
        divider = make_axes_locatable(axes[0, 0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cax.axis('off')

        # Images of Stokes paramaters
        mini = stokes_frame_norm[:, :, 1:].min()
        maxi = stokes_frame_norm[:, :, 1:].max()
        v = np.max(np.abs([mini, maxi]))
        for S, coord in enumerate([(0, 1), (1, 0), (1, 1)]):
            DataPlotter.stokes_image(
                stokes_frame_norm, S + 1, ax=axes[coord], v=v)
            DataPlotter.highlight_cell(arg[1], arg[2], ax=axes[coord],
                color='green', lw=5)

        # Image of Fourier analysis Stokes paramater calculation
        ax = axes[2, 0]
        # Data plot
        if err is not None:
            ax.errorbar(angles, pixel, yerr=pixel_err,
                        label='Measurement', color=DataPlotter.COLORS[0])
        else:
            ax.plot(
                angles, pixel, label='Measurement', color=DataPlotter.COLORS[0])
        # Stokes vector expected intensity plot
        I, theta = DataPlotter.intensity_curve(stokes_vector)
        ax.plot(theta, I, label='Expected', color=DataPlotter.COLORS[1])
        if stokes_err is not None:
            upper, *x = DataPlotter.intensity_curve(
                stokes_vector + stokes_vector_err)
            lower, *x = DataPlotter.intensity_curve(
                stokes_vector-stokes_vector_err)
            DataPlotter.error_margin(
                theta, upper, lower, ax, color=DataPlotter.COLORS[1])
        ax.legend()
        ax.set_xlabel('Wave-plate Rotation Angle (degrees)')
        ax.set_ylabel('Intensity (Pixel Count)')
        ax.set_title('Fit of Stokes parameters for Single Pixel')

        # Fourier transform
        ax = axes[2, 1]
        params = DataPlotter.fourier(
            angles, pixel, ax=ax, label='Measured')
        for param in params[1:3]:
            param.set_color(DataPlotter.COLORS[0])
        params = DataPlotter.fourier(
            theta, I, ax=ax, label='Calculated')
        for param in params[1:3]:
            param.set_color(DataPlotter.COLORS[1])
        ax.legend(loc='upper right')


        fig.suptitle(name)
        fig.tight_layout()
        if name is None:
            plt.show()
        else:
            fig.savefig(name + '.png', dpi=300, transparent=False,
                        bbox_inches='tight')
            plt.close(fig)

    def stokes_vs_angle(stokes_vectors, angles, stokes_err=None, name=None,
                        ax=None):
        '''Plot the different Stokes parameters as a function of angle of
        observation. If name is given, plot is saved to current working
        directory. Otherwise it is shown.'''
        if ax is None:
            fig, ax = plt.subplots(nrows=4, sharex=True, figsize=(5, 10))
            if name is not None:
                save = True
        else:
             save = False
        colors = ['red', 'green', 'blue']
        markers = ['v', 'D', 'o']

        # Plot each component of S in a different ax
        norm = np.max(stokes_vectors[:, :, 0], axis=1)
        for S in range(4):
            for i, color in enumerate(colors):
                if S > 0:
                    stokes_vectors[:, i, S] = stokes_vectors[:, i, S] / norm
                    stokes_err[:, i, S] = stokes_err[:, i, S] / norm
                if S < 3:
                    ax[S].axes.xaxis.set_visible(False)
                if S == 3:
                    ax[S].set_xlabel('Angle out (degrees)')
                ax[S].errorbar(
                    angles, stokes_vectors[:, i, S],
                    yerr=stokes_err[:, i, S], ls=':',
                    label=f'{color} channel', color=color, marker=markers[i])
            ax[S].set_ylabel(f'S{S}')
            ax[S].hlines(0,angles.min()-1,angles.max()+1, colors='black')
            ax[S].set_xlim(angles.min()-1,angles.max()+1)
        ax[0].legend()
        if save:
            fig.suptitle(name)
            fig.tight_layout()
            fig.savefig(name+'.png', dpi=300)
            plt.close(fig)
        return ax

if __name__ == '__main__':
    path = None
    DA = DataAnalyzer(path, verbose=True, force_calculation=True)
    DA.stokes_vs_angle_generator()
    DA.single_frames_generator()
