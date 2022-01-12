import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from tqdm import tqdm
from numba import jit

def frame_overview(arr):
    print(f'Shape:  {arr.shape}')
    print(f'Min:  {np.min(arr, axis=(0,1))}')
    print(f'Max:  {np.max(arr, axis=(0,1))}')
    print(f'Mean:  {np.mean(arr, axis=(0,1))}')
    print(f'Median:  {np.median(arr, axis=(0,1))}')

@jit(nopython=True)
def calc_stokes_params(intensity, theta):
    shape = intensity.shape[1:]
    N = len(theta)
    theta = theta / 180 * np.pi

    A = 2 / N * np.sum(intensity, axis=0)
    coef = create_array(np.sin(2 * theta), shape)
    B = 4 / N * np.sum(intensity * coef, axis=0)
    coef = create_array(np.cos(4 * theta), shape)
    C = 4 / N * np.sum(intensity * coef, axis=0)
    coef = create_array(np.sin(4 * theta), shape)
    D = 4 / N * np.sum(intensity * coef, axis=0)
    S1 = A - C
    S2 = 2 * C
    S3 = 2 * D
    S4 =  B
    vector = np.array([S1, S2, S3, S4])
    return vector.transpose(1, 2, 0)

def create_array(vec, shape):
    N = len(vec)
    a = np.empty((shape + (N,)))
    a[:] = vec
    a = a.transpose(2, 0, 1)
    return a

def DOP(stokes_vector):
    numerator = np.sqrt(np.sum(stokes_vector[:1]**2, axis=-1))
    denominator = stokes_vector[:, :, 0]
    return numerator/denominator

def argmax2d(frame):
    index = np.unravel_index(frame.argmax(), frame.shape)
    return index

def highlight_cell(x,y, width=10, ax=None, **kwargs):
    rect = plt.Rectangle((x-int(width/2), y-int(width/2)), width, width, fill=False, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect

@jit(nopython=True)
def correct_raw_data(filename_raw):
    # Get all frames and perform data correction
    # create new hdf5 file with corrected images
    filename_corrected = filename[:-5] + ' corrected' + '.hdf5'
    raw = h5py.File(filename_raw,'r')
    f = h5py.File(filename_corrected, "w")
    #with h5py.File(filename_corrected, "w") as f:
    # Transfer metadata
    for m in raw.attrs:
        f.attrs[m] = raw.attrs[m]
    #create group for each filter
    for filter_name in raw:
        #print(f'Correcting filter "{filter_name}"')
        print(f'Correcting...')
        filter = raw[filter_name]
        filter_corrected = f.create_group(filter_name)
        # Transfer group metadata
        metadata = filter.attrs
        for m in metadata:
            filter_corrected.attrs[m] = metadata[m]
        metadata = filter['Frames 0'].attrs
        for m in metadata:
            filter_corrected.attrs[m] = metadata[m]
        # Correct images
        try:
            dark_frame = filter['dark frame 0']
            dark_frame = np.median(dark_frame, axis=0)
        except:
            None
        #corrected_imgs = []
        for index, run in enumerate(tqdm(filter)):
            if 'dark frame' in run:
                continue
            imgs = filter[run][:]
            try:
                len(dark_frame)
                imgs = imgs - dark_frame
                imgs[imgs<0] = 0
            except:
                None
            imgs = np.expand_dims(imgs, axis=0)
            if index == 0:
                corrected_imgs = imgs
            else:
                corrected_imgs = np.concatenate((corrected_imgs, imgs))
            corrected_imgs.append(imgs)
            del imgs

        print('Creating new HDF5 file.')
        err = np.std(corrected_imgs, axis=0) / np.sqrt(len(corrected_imgs))
        filter_corrected.create_dataset('Frames_err', data=err, compression="gzip")
        del err

        corrected_imgs = np.mean(corrected_imgs, axis=0)
        filter_corrected.create_dataset('Frames', data=corrected_imgs, compression="gzip")
        #print(f'Shape of dataset: {corrected_imgs.shape}')
        del corrected_imgs

        positions = filter[run].attrs['positions']
        filter_corrected.create_dataset('positions', data=positions, compression="gzip")
        polarizer_angles = filter[run].attrs['polarizer_angles']
        filter_corrected.create_dataset('polarizer_angles', data=polarizer_angles, compression="gzip")
    raw.close()
    f.close()
    return filename_corrected

def print_metadata(file):
    print('\nFile metadata:')
    for m in file.attrs:
        print(f'{m}:  {file.attrs[m]}')

    print('\nGroups:')
    for key in file.keys():
        print(key)

    first_grp = list(file.keys())[0]
    print(f'\nDatasets of group "{first_grp}":')
    for key in file[first_grp].keys():
        print(key)

    print(f'\nMetadata of group "{first_grp}":')
    for m in file[first_grp].attrs:
        print(f'{m}:  {file[first_grp].attrs[m]}')

@jit(nopython=True)
def generate_stokes_frames(filename):
    with h5py.File(filename, 'a') as f:
        print('Generating Stokes frames.')
        for filter_name in f:
            group = f[filter_name]
            if 'Stokes_frames' in group:
                del group['Stokes_frames']
            theta = group['polarizer_angles'][:]
            for index, frame in enumerate(tqdm(group['Frames'][:])):
                stokes_frame = calc_stokes_params(frame.sum(axis=3), theta)
                stokes_frame = np.expand_dims(stokes_frame, axis=0)
                if index == 0:
                    stokes_frames = stokes_frame
                else:
                    stokes_frames = np.concatenate(stokes_frames, stokes_frame)
            group.create_dataset(
                'Stokes_frames', data=stokes_frames, compression="gzip")

filepath = input('(Full) path to HDF5 file to consider: ')
#filepath = '/Users/naorscheinowitz/Master Project/Beetle Project/Beetle Classifier Robot/Experiments/Beetle Hyperspectral/2022.01.10 Beetle Hyperspectral single position high accuracy 2 corrected.hdf5'
filename = os.path.basename(filepath)
os.chdir(os.path.dirname(filepath))

if 'corrected' not in filename:
    print('Correcting raw data and creating new file.')
    filename = correct_raw_data(filename)

generate_stokes_frames(filename)
exit()
file = h5py.File(filename,'r')

frame = file['No filter/Frames'][0]
stokes_frame = file['No filter/Stokes_frames'][0]
angles = file['No filter/polarizer_angles'][:]

# Select interesting pixel
arg = argmax2d(stokes_frame[600:,600:])[:2]
pixel = frame[:, arg[0]+600, arg[1]+600].sum(axis=-1)
pixel_err = file['No filter/Frames_err'][0, :, arg[0]+600, arg[1]+600].sum(axis=-1)
stokes_vector = stokes_frame[arg[0]+600, arg[1]+600]

# Generate expected intensity curve with given Stokes vector
theta = np.linspace(0, np.pi, 100)
I1 = stokes_vector[0]
I2 = stokes_vector[1] * np.cos(2 * theta)**2
I3 = stokes_vector[2] * np.cos(2 * theta) * np.sin(2 * theta)
I4 = stokes_vector[3] * np.sin(2 * theta)
I = 0.5 * (I1 + I2 + I3 + I4)


# Plotting
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15,10))
stokes_frame = stokes_frame / stokes_frame.max()

# Image of Beetle
#axes[0,0].imshow(frame[0]/frame[0].max(), vmin=0, vmax=0.05)
axes[0,0].imshow(np.clip(frame[0], 0, 10)/10)
axes[0,0].set_title('Image of beetle')

# Image of degree op polarization
cutoff = 2 * np.median(stokes_frame[:,:,0])
P = DOP(stokes_frame)
P[stokes_frame[:,:,0] <= cutoff] = 0
im = axes[0,1].imshow(P, vmin=0, vmax=1)
fig.colorbar(im, ax=axes[0,1])
axes[0,1].set_title('Degree of polarization')

# Images of linear polarization
horizontal = np.sqrt((stokes_frame[:, :, 0] + stokes_frame[:, :, 1]) / 2)
vertical = np.sqrt((stokes_frame[:, :, 0] - stokes_frame[:, :, 1]) / 2)
im2 = axes[1,0].imshow(stokes_frame[:, :, 1])#, vmin=-0.1, vmax=0.1)
im3 = axes[1,1].imshow(stokes_frame[:, :, 2])#, vmin=-0.1, vmax=0.1)
fig.colorbar(im2, ax=axes[1,0])
fig.colorbar(im3, ax=axes[1,1])
axes[1,0].set_title('S1')
axes[1,1].set_title('S2')

# Image of circular polarization
im4 = axes[2,0].imshow(stokes_frame[:, :, 3])#, vmin=-0.1, vmax=0.1)
highlight_cell(arg[1]+600, arg[0]+600, 50, ax=axes[2,0], color='red')
fig.colorbar(im2, ax=axes[2,0])
axes[2,0].set_title('S4')

# Image of fit
axes[2,1].plot(angles, pixel, label='Measurement')
axes[2,1].plot(theta/np.pi*180, I,label='Expected')
axes[2,1].errorbar(angles, pixel, yerr=pixel_err)
axes[2,1].legend()
axes[2,1].set_title('Fit of Stokes parameters')


plt.savefig('plaatjes.png', dpi=300)
plt.show()
file.close()
