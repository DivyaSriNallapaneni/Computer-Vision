import cv2
import numpy as np
import os

def main():
    # Base folder: module2_corr_single_scene/
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    in_path   = os.path.join(base_dir, "task2", "L.jpg")
    blur_path = os.path.join(base_dir, "task2", "L_b.jpg")
    rec_path  = os.path.join(base_dir, "task2", "L_restored.jpg")

    if not os.path.exists(in_path):
        print("❌ Input image L.jpg not found at", in_path)
        return

    # 1. Load original image L as grayscale float [0,1]
    L_u8 = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
    if L_u8 is None:
        print("❌ Failed to read L.jpg")
        return

    L = L_u8.astype(np.float32) / 255.0
    h, w = L.shape
    print(f"[INFO] Loaded L: shape={L.shape}, min={L.min():.4f}, max={L.max():.4f}")

    # 2. Apply Gaussian blur (this creates L_b)
    # You can tweak ksize and sigma.
    ksize = 15        # must be odd
    sigma = 3.0

    L_b = cv2.GaussianBlur(L, (ksize, ksize), sigma)
    cv2.imwrite(blur_path, (L_b * 255).astype(np.uint8))
    print(f"[INFO] Saved blurred image to {blur_path}")

    # 3. Build the same Gaussian kernel (PSF), but let FFT pad it to image size
    g1d = cv2.getGaussianKernel(ksize, sigma)
    g2d = g1d @ g1d.T         # 2D Gaussian
    g2d = g2d / g2d.sum()     # normalize

    # IMPORTANT:
    # Instead of manually padding + shifting, let np.fft.fft2 handle padding.
    # This avoids the "four quadrants" artifact.
    H = np.fft.fft2(g2d, s=L.shape)

    # 4. FFT of blurred image
    Fb = np.fft.fft2(L_b)

    # 5. Regularized inverse filtering in the frequency domain
    eps = 1e-3   # regularization; increase => smoother, decrease => sharper

    H_conj = np.conj(H)
    denom  = (np.abs(H) ** 2) + eps
    F_est  = (H_conj / denom) * Fb

    # 6. Back to spatial domain
    L_rec = np.fft.ifft2(F_est)
    L_rec = np.real(L_rec)

    # 7. Normalize to [0,1] then [0,255] for display
    min_val, max_val = L_rec.min(), L_rec.max()
    print(f"[INFO] Restored raw range: min={min_val:.4f}, max={max_val:.4f}")

    if max_val - min_val < 1e-6:
        L_rec_norm = np.zeros_like(L_rec, dtype=np.float32)
    else:
        L_rec_norm = (L_rec - min_val) / (max_val - min_val)

    L_rec_u8 = (L_rec_norm * 255.0).clip(0, 255).astype(np.uint8)
    cv2.imwrite(rec_path, L_rec_u8)
    print(f"[INFO] Saved restored image to {rec_path}")

if __name__ == "__main__":
    main()
