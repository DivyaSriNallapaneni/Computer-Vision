import cv2
import numpy as np

def cylindrical_projection(img, f):
    """
    Projects an image onto a cylinder (vertical axis). 
    f: focal length in pixels (approx = image width or tuned value)
    """
    h, w = img.shape[:2]
    # center
    cx = w / 2.0
    cy = h / 2.0

    # prepare output canvas (same size)
    cyl = np.zeros_like(img)
    for y in range(h):
        for x in range(w):
            # coordinates relative to center
            x_rel = x - cx
            y_rel = y - cy
            # cylindrical coordinates
            theta = np.arctan2(x_rel, f)
            h_ = np.sqrt(x_rel**2 + f**2)
            y_c = y_rel * f / h_
            X = f * theta
            Y = y_c
            u = X + cx
            v = Y + cy
            if 0 <= int(np.round(v)) < h and 0 <= int(np.round(u)) < w:
                cyl[int(np.round(v)), int(np.round(u))] = img[y, x]

    # inpainting to fill black holes (simple)
    gray = cv2.cvtColor(cyl, cv2.COLOR_BGR2GRAY)
    mask = (gray == 0).astype('uint8') * 255
    if np.any(mask):
        cyl = cv2.inpaint(cyl, mask, 3, cv2.INPAINT_TELEA)
    return cyl

def _blend_images(base, warped, offset_x, offset_y):
    """
    Simple linear blending for overlapping region.
    base and warped share canvas coordinates, both same size.
    offset_x, offset_y: top-left where base was placed (usually 0,0)
    """
    h, w = base.shape[:2]
    # create weight masks
    base_gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    base_mask = (base_gray > 0).astype('float32')
    warped_mask = (warped_gray > 0).astype('float32')

    # compute overlap
    overlap = (base_mask * warped_mask) > 0
    result = np.zeros_like(base)

    # linear blend in overlap
    for c in range(3):
        b = base[:, :, c].astype('float32')
        wimg = warped[:, :, c].astype('float32')
        channel = np.where(overlap,
                           (b * 0.5 + wimg * 0.5),
                           np.where(warped_mask > 0, wimg, b))
        result[:, :, c] = np.clip(channel, 0, 255).astype('uint8')
    return result

def stitch_images(img1, img2, H, max_width=4000, max_height=2000):
    """
    Warp img2 into img1 coordinate using H and return a blended result.
    Limits canvas size to max_width x max_height to avoid OOM.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # corners of img2 transformed
    corners2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    trans_corners2 = cv2.perspectiveTransform(corners2, H)

    # corners of img1
    corners1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)

    all_corners = np.vstack((corners1, trans_corners2))
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # clamp canvas size
    canvas_w = xmax - xmin
    canvas_h = ymax - ymin
    if canvas_w > max_width:
        # center crop region horizontally
        center_x = (xmax + xmin) // 2
        xmin = center_x - max_width//2
        xmax = center_x + max_width//2
        canvas_w = max_width
    if canvas_h > max_height:
        center_y = (ymax + ymin) // 2
        ymin = center_y - max_height//2
        ymax = center_y + max_height//2
        canvas_h = max_height

    # translation to keep everything positive
    translation = [-xmin, -ymin]
    H_trans = np.array([[1,0,translation[0]],[0,1,translation[1]],[0,0,1]])

    # warp img2 into canvas
    try:
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype='uint8')
        warped = cv2.warpPerspective(img2, H_trans.dot(H), (canvas_w, canvas_h))
    except Exception as e:
        print("Warp failed:", e)
        return img1

    # place img1 on canvas
    x_off = translation[0]
    y_off = translation[1]
    # clip where img1 goes on canvas
    x1s = max(0, x_off)
    y1s = max(0, y_off)
    x1e = min(canvas_w, x_off + w1)
    y1e = min(canvas_h, y_off + h1)
    canvas[y1s:y1e, x1s:x1e] = img1[(y1s - y_off):(y1e - y_off), (x1s - x_off):(x1e - x_off)]

    # blend the warped image and the current canvas
    result = _blend_images(canvas, warped, x_off, y_off)
    return result
