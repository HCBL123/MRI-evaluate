import os
import sys
from pathlib import Path

# Add the directory containing acl_classifier.py to the Python path
sys.path.append(Path(__file__).parent.resolve())

from acl_classifier import GlobalACLClassifier 

def predict_acl_tear(dicom_root_folder):
    """
    Predict ACL tear from DICOM files
    Args:
        dicom_root_folder: folder containing 'sagittal', 'coronal', 'axial' subfolders with DICOM files
    """
    # Initialize classifier
    classifier = GlobalACLClassifier(
        models_dir='models/',
        data_dir='data/'
    )
    
    # Load trained classifier
    classifier.load_global_classifier('models/global_classifier.pkl')
    
    # Convert DICOM to numpy for each plane
    scans = {}
    for plane in ['sagittal', 'coronal', 'axial']:
        dicom_folder = os.path.join(dicom_root_folder, plane)
        if os.path.exists(dicom_folder):
            scans[plane] = dicom_to_numpy(dicom_folder, plane)
        else:
            raise ValueError(f"Missing {plane} plane folder")
    
    # Get prediction
    probability = classifier.predict(
        axial_scan=scans['axial'],
        coronal_scan=scans['coronal'],
        sagittal_scan=scans['sagittal']
    )
    
    return probability

result = predict_acl_tear('/dicom')
print(f"Probability of ACL tear: {result:.3f}")
print(f"Diagnosis: {'ACL tear' if result > 0.5 else 'No ACL tear'}")