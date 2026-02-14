import numpy as np
import cv2

def calculate_gaze_vector(keypoints):
    """
    Calculate the gaze vector based on keypoints.
    Definition: Vector from the midpoint of the eyes to the nose.
    
    Args:
        keypoints (dict): Dictionary with keys 'Nose', 'Left Eye', 'Right Eye' (or lowercase).
                          Values should be (x, y) tuples or numpy arrays.
                          
    Returns:
        np.array: Normalized gaze vector (x, y).
        tuple: Start point of the vector (midpoint of eyes).
    """
    # Normalize keys to lowercase for robustness
    kpts = {k.lower(): v for k, v in keypoints.items()}
    
    # Check for required keypoints
    required = ['nose', 'lefteye', 'righteye'] # Adjust based on actual DLC model output (SuperAnimal usually has these)
    # Common variations in SuperAnimal: 'Nose', 'Left Eye', 'Right Eye' -> 'nose', 'lefteye', 'righteye'
    # Let's handle 'left eye' vs 'lefteye'
    
    # Map common variations
    name_map = {
        'left eye': 'lefteye',
        'right eye': 'righteye',
        'nose': 'nose'
    }
    
    # Remap keys
    clean_kpts = {}
    for k, v in kpts.items():
        clean_k = name_map.get(k, k)
        clean_kpts[clean_k] = np.array(v[:2], dtype=float) # Ensure only x,y and float
        
    try:
        if 'lefteye' in clean_kpts and 'righteye' in clean_kpts and 'nose' in clean_kpts:
            left_eye = clean_kpts['lefteye']
            right_eye = clean_kpts['righteye']
            nose = clean_kpts['nose']
            
            # Midpoint of eyes
            eyes_mid = (left_eye + right_eye) / 2.0
            
            # Vector from eyes to nose
            gaze = nose - eyes_mid
            
            # Normalize
            norm = np.linalg.norm(gaze)
            if norm > 0:
                gaze = gaze / norm
                
            return gaze, eyes_mid
        else:
            return None, None
    except Exception as e:
        print(f"Error calculating gaze: {e}")
        return None, None

def draw_gaze(image, start_point, vector, color=(0, 0, 255), thickness=2, length=50):
    """
    Draw the gaze vector on the image.
    
    Args:
        image (np.array): Image to draw on.
        start_point (tuple): (x, y) coordinates of the vector start.
        vector (np.array): Normalized (x, y) direction vector.
        length (int): Length of the arrow to draw in pixels.
    """
    if start_point is None or vector is None:
        return image
        
    start_point_int = tuple(start_point.astype(int))
    end_point = start_point + vector * length
    end_point_int = tuple(end_point.astype(int))
    
    cv2.arrowedLine(image, start_point_int, end_point_int, color, thickness, tipLength=0.3)
    return image
