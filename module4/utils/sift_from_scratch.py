import cv2
import numpy as np

def compute_gradients(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    angle = cv2.phase(gx, gy, angleInDegrees=True)
    return mag, angle

def detect_keypoints(img, num_points=500):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    kp = np.argwhere(dst > 0.01*dst.max())
    keypoints = [cv2.KeyPoint(float(x[1]), float(x[0]), 1) for x in kp]
    if len(keypoints) > num_points:
        keypoints = keypoints[:num_points]
    return keypoints

def compute_descriptors(img, keypoints):
    mag, angle = compute_gradients(img)
    descriptors = []
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        patch_mag = mag[max(0,y-8):y+8, max(0,x-8):x+8]
        patch_angle = angle[max(0,y-8):y+8, max(0,x-8):x+8]
        if patch_mag.shape != (16,16):
            patch_mag = cv2.copyMakeBorder(patch_mag,0,16-patch_mag.shape[0],0,16-patch_mag.shape[1],cv2.BORDER_CONSTANT,0)
            patch_angle = cv2.copyMakeBorder(patch_angle,0,16-patch_angle.shape[0],0,16-patch_angle.shape[1],cv2.BORDER_CONSTANT,0)
        descriptor = []
        for i in range(0,16,4):
            for j in range(0,16,4):
                mag_cell = patch_mag[i:i+4,j:j+4].flatten()
                angle_cell = patch_angle[i:i+4,j:j+4].flatten()
                hist = np.zeros(8)
                for k in range(len(mag_cell)):
                    bin_idx = int(np.floor(angle_cell[k]/45.0)) % 8
                    hist[bin_idx] += mag_cell[k]
                descriptor.extend(hist)
        descriptor = np.array(descriptor)
        descriptor /= (np.linalg.norm(descriptor)+1e-7)
        descriptors.append(descriptor)
    return np.array(descriptors, dtype=np.float32)
