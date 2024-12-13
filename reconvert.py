#Convert the Convert 3D numpy array to a single multi-frame DICOM file

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
import datetime
import os

def numpy_to_multiframe_dicom(array_3d, output_file, patient_name="Anonymous", patient_id="0000"):
    """
    Convert 3D numpy array to a single multi-frame DICOM file
    
    Args:
        array_3d: 3D numpy array (slices, height, width)
        output_file: output DICOM filename
        patient_name: patient name for DICOM metadata
        patient_id: patient ID for DICOM metadata
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # Create file meta dataset
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.4.1'  # Enhanced MR Image Storage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    # Create dataset
    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)

    # Patient info
    ds.PatientName = patient_name
    ds.PatientID = patient_id
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    # Study and Series info
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.Modality = "MR"
    ds.SeriesNumber = 1
    ds.InstanceNumber = 1

    # Image info
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0

    # Set dimensions
    ds.NumberOfFrames = array_3d.shape[0]
    ds.Rows = array_3d.shape[1]
    ds.Columns = array_3d.shape[2]

    # Dates and times
    dt = datetime.datetime.now()
    ds.ContentDate = dt.strftime('%Y%m%d')
    ds.ContentTime = dt.strftime('%H%M%S.%f')

    # Convert array to uint16 if needed
    if array_3d.dtype != np.uint16:
        array_3d = ((array_3d - array_3d.min()) * (65535.0 / (array_3d.max() - array_3d.min()))).astype(np.uint16)

    # Combine all slices into one pixel array
    ds.PixelData = array_3d.tobytes()

    # Save the file
    ds.save_as(output_file)

def main():
    # Load numpy arrays
    axial_array = np.load('./data/train/axial/0000.npy')
    coronal_array = np.load('./data/train/coronal/0000.npy')
    sagittal_array = np.load('./data/train/sagittal/0000.npy')
    
    print("Converting arrays with shapes:")
    print(f"Axial: {axial_array.shape}")
    print(f"Coronal: {coronal_array.shape}")
    print(f"Sagittal: {sagittal_array.shape}")

    # Convert each plane to multi-frame DICOM
    for plane, array in [('axial', axial_array), 
                        ('coronal', coronal_array), 
                        ('sagittal', sagittal_array)]:
        output_file = f'output_dicoms/{plane}_multiframe.dcm'
        print(f"\nConverting {plane} plane to DICOM...")
        numpy_to_multiframe_dicom(array, output_file)
        print(f"Saved to {output_file}")

if __name__ == "__main__":
    main()