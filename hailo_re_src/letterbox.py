"""Ultralytics-identical letterbox (ultralytics.data.augment.LetterBox, auto=False, center=True, gray 114).

Shared by calibration, QAT, evaluation and deployment so every stage sees the same pixels.
"""
import cv2


def letterbox(img, shape, color=(114, 114, 114)):
    """img: HxWx3 uint8. shape: (H, W). Returns (padded, ratio, (pad_left, pad_top))."""
    h0, w0 = img.shape[:2]
    r = min(shape[0] / h0, shape[1] / w0)
    nw, nh = int(round(w0 * r)), int(round(h0 * r))
    dw, dh = (shape[1] - nw) / 2, (shape[0] - nh) / 2
    if (w0, h0) != (nw, nh):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (left, top)
