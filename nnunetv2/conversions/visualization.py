import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

def load_nii_data(file_path):
    return nib.load(file_path).get_fdata()

class ScrollableVolumeViewer:
    def __init__(self, volume1, volume2, window_level=40, window_width=400):
        self.volume1 = volume1
        self.volume2 = volume2
        self.window_level = window_level
        self.window_width = window_width
        self.slice_index = volume1.shape[2] // 2  # Start from the middle slice
        self.fig, self.axes = plt.subplots(1, 2)
        self.update()

    def connect(self):
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def on_key(self, event):
        if event.key == 'up':
            self.slice_index = min(self.slice_index + 1, self.volume1.shape[2] - 1)
            self.update()
        elif event.key == 'down':
            self.slice_index = max(self.slice_index - 1, 0)
            self.update()

    def apply_windowing(self, slice):
        lower_bound = self.window_level - (self.window_width / 2)
        upper_bound = self.window_level + (self.window_width / 2)
        windowed_slice = np.clip(slice, lower_bound, upper_bound)
        windowed_slice = (windowed_slice - lower_bound) / self.window_width
        return windowed_slice

    def update(self):
        # Clear the current axes
        self.axes[0].clear()
        self.axes[1].clear()

        # Apply windowing to the first volume's slice
        slice1_windowed = self.apply_windowing(self.volume1[self.slice_index, :, :])

        # Update the slice in each subplot
        self.axes[0].imshow(slice1_windowed.T, cmap='gray', origin='lower')
        self.axes[1].imshow(self.volume2[self.slice_index, :, :].T, cmap='gray', origin='lower')

        # Set titles
        self.axes[0].set_title('Volume 1 - Slice {}'.format(self.slice_index))
        self.axes[1].set_title('Volume 2 - Slice {}'.format(self.slice_index))

        for ax in self.axes:
            ax.axis('off')

        # Draw the updated figures
        self.fig.canvas.draw()


# Load your volumes
volume1 = load_nii_data('C:/MyPythonProjects/XipeAI/test_data/prediction_input/nifti/unnamed_8/volume_008_0000.nii.gz')
volume2 = load_nii_data('C:/MyPythonProjects/XipeAI/test_data/prediction_output_pp/unnamed_8/volume_008.nii.gz')

# Initialize and connect the viewer
viewer = ScrollableVolumeViewer(volume1, volume2)
viewer.connect()

# Display the matplotlib window
plt.show()
