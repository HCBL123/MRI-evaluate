import pydicom  # for reading DICOM files
import numpy as np

def dicom_to_numpy(dicom_folder, plane):
    """Convert DICOM series to numpy array"""
    # Read all DICOM files in the folder
    dicom_files = sorted([f for f in os.listdir(dicom_folder) if f.endswith('.dcm')])
    
    # Stack all slices into a 3D array
    slices = []
    for f in dicom_files:
        dcm = pydicom.read_file(os.path.join(dicom_folder, f))
        slices.append(dcm.pixel_array)
    
    # Convert to numpy array
    volume = np.stack(slices)
    
    # Resize to 256x256 if needed
    from skimage.transform import resize
    if volume.shape[1] != 256 or volume.shape[2] != 256:
        resized_slices = []
        for slice_img in volume:
            resized_slices.append(resize(slice_img, (256, 256), preserve_range=True))
        volume = np.stack(resized_slices)
    
    return volume