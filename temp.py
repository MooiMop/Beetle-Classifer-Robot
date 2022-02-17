'''



@njit
def normalize_stokes_frames(stokes_frames, errors):
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
def DOP(stokes_frame):
    numerator = np.sqrt(np.sum(stokes_frame[:,:,1:]**2, axis=-1))
    denominator = stokes_frame[:, :, 0]
    output = numerator / denominator
    return output



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

def plot_single_frame(frame, stokes_frame, stokes_frame_norm, angles,
    err=None, stokes_err=None, name=None):

    # Select brightest part
    arg = argmax2d(frame)[0:3]  # only want x and y coordinates on frame
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
    axes[0,0].imshow(np.clip(RGB_img, 0, 50)/50)
    axes[0,0].set_title('RGB image')

    # Image of degree op polarization
    P = DOP(stokes_frame)
    im = axes[0,1].imshow(P, vmin=0, vmax=1, cmap='hot')
    fig.colorbar(im, ax=axes[0,1])
    axes[0,1].set_title('Degree of polarization')

    # Images of Stokes paramaters
    mini = np.min(stokes_frame_norm[:, :, 1:])
    maxi = np.min(stokes_frame_norm[:, :, 1:])
    v = np.abs(np.max([mini,maxi]))
    V = 1
    im1 = axes[1,0].imshow(stokes_frame_norm[:, :, 1], vmin=-v, vmax=v, cmap='bwr')
    im2 = axes[1,1].imshow(stokes_frame_norm[:, :, 2], vmin=-v, vmax=v, cmap='bwr')
    im3 = axes[2,0].imshow(stokes_frame_norm[:, :, 3], vmin=-v, vmax=v, cmap='bwr')
    axes[1,0].set_title(f'S1')
    axes[1,1].set_title(f'S2')
    axes[2,0].set_title(f'S3')
    fig.colorbar(im1, ax=axes[1,0])
    fig.colorbar(im2, ax=axes[1,1])
    fig.colorbar(im2, ax=axes[2,0])
    highlight_cell(arg[2], arg[1], 30, ax=axes[2,0], color='green', linewidth=2)

    # Image of Fourier analysis Stokes paramater calculation
    if err is not None:
        axes[2,1].errorbar(angles, pixel, yerr=pixel_err, label='Measurement')
    else:
        axes[2,1].plot(angles, pixel, label='Measurement')
    I, theta = intensity_curve(stokes_vector)
    axes[2,1].plot(theta, I, label='Expected', c='red')
    if stokes_err is not None:
        upper, dump= intensity_curve(stokes_vector+stokes_vector_err)
        lower, dump= intensity_curve(stokes_vector-stokes_vector_err)
        axes[2,1].fill_between(
            theta, upper, lower, facecolor='red', alpha=0.3,
            label=r'Error interval')
    axes[2,1].legend()
    axes[2,1].set_title('Fit of Stokes parameters')


    if name is not None:
        fig.savefig(name, dpi=300)
        plt.close(fig)
    else:
        plt.show()


'''
