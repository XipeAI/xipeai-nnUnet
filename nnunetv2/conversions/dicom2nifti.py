import os
import pydicom
import numpy as np
import nibabel as nib

def dicom_series_to_nifti(input_folder, output_folder):
    # Extract the number after the last underscore in the last folder name and pad it to three digits
    last_folder_name = os.path.basename(os.path.normpath(input_folder))
    number_part = last_folder_name.split('_')[-1]  # Get the part after the last underscore
    formatted_number = number_part.zfill(3)  # Pad it to three digits

    # Initialize an empty list to hold the DICOM images
    slices = []

    # Read all DICOM files in the input folder
    for s in os.listdir(input_folder):
        try:
            # Attempt to read the DICOM file
            path = os.path.join(input_folder, s)
            slices.append(pydicom.dcmread(path))
        except:
            # If an error occurs, skip the file
            continue

    # Ensure that slices are sorted by their Instance Number
    slices.sort(key=lambda x: int(x.InstanceNumber))

    # Stack the DICOM slices and get the pixel array
    image_data = np.stack([s.pixel_array for s in slices])

    # Convert to float and scale pixel values if necessary (e.g., for CT images)
    image_data = image_data.astype(np.float32)
    if hasattr(slices[0], 'RescaleSlope') and hasattr(slices[0], 'RescaleIntercept'):
        image_data = image_data * slices[0].RescaleSlope + slices[0].RescaleIntercept

    # Create a NIfTI image (you might need to adjust the affine matrix for correct orientation)
    nifti_img = nib.Nifti1Image(image_data, np.eye(4))

    # Define the output file path
    output_file_path = os.path.join(output_folder, f'volume_{formatted_number}_0000.nii.gz')

    # Save the NIfTI image
    nib.save(nifti_img, output_file_path)

    print(f'Saved NIfTI file to {output_file_path}')

if __name__ == '__main__':
    # Example usage
    dicom_series_to_nifti('C:/MyPythonProjects/XipeAI/test_data/prediction_input/dicom/unnamed_8', 'C:/MyPythonProjects/XipeAI/test_data/prediction_input/nifti/unnamed_8')