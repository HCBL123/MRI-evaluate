import requests

def test_api():
    url = "http://localhost:8000/predict_acl"
    
    # Replace with your DICOM file paths
    files = {
        'axial_file': ('axial.dcm', open('./output_dicoms/axial_multiframe.dcm', 'rb')),
        'coronal_file': ('coronal.dcm', open('./output_dicoms/coronal_multiframe.dcm', 'rb')),
        'sagittal_file': ('sagittal.dcm', open('./output_dicoms/sagittal_multiframe.dcm', 'rb'))
    }
    
    response = requests.post(url, files=files)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Final ACL Tear Probability: {result['final_probability']:.3f}")
        print("\nProbabilities by plane:")
        for plane, prob in result['plane_probabilities'].items():
            print(f"{plane}: {prob:.3f}")
        
        # Save CAM visualizations
        for plane in ['axial', 'coronal', 'sagittal']:
            with open(f'cam_{plane}.png', 'wb') as f:
                f.write(response.content)
            print(f"Saved CAM visualization for {plane} plane")
    else:
        print(f"Error: {response.text}")

if __name__ == "__main__":
    test_api()