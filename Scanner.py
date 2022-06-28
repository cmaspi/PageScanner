"""
Scanner
-------

Provides
1. grayscaling an image
2. blur an image
3. make a binary mask using canny edge detector
4. find the biggest contour (perimeter)
5. Converts the gray scale image to a binary threshold image
"""

from typing import Tuple
import cv2
import numpy as np


def grayScale(img : np.ndarray) -> np.ndarray:
    """
    Converts an image to grayscale

    Args:
    -----
        img (np.ndarray): input image

    Returns:
    --------
        np.ndarray: grayscale image
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def blur(img : np.ndarray, k: int = 3, std : int = 0) -> np.ndarray:
    """
    blurs the image using gaussian blur

    Args:
    -----
        img (np.ndarray): input image (grayscale)
        k (int, optional): kernel size. Defaults to 3.

    Returns:
    --------
        np.ndarray: denoised image

    Theory:
    -------
    Gaussian blur uses a kernel and convoles
    it with the input image, it reduces the 
    salt and paper noises in the image
    """
    return cv2.GaussianBlur(img, (k,k), std)

def binary(img : np.ndarray) -> np.ndarray:
    """
    finds the edges in the image, and make a binary
    mask, 255 for edges, 0 for rest

    Args:
    -----
        img (np.ndarray): input image (grayscale)

    Returns:
    --------
        np.ndarray: binary image with edges highlighted
    
    Theory:
    -------
    Canny edge detector uses Sobel method to find the
    edges. Sobel method essentially find the gradient
    of intensity in x and y direction.
    """
    return cv2.Canny(img, 100, 255)

def retBiggestContour(img : np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    returns the biggest contour and width, height
    of the minimum area rectangle that bounds it

    Args:
    -----
        img (np.ndarray): input image (B&W from canny edge)

    Returns:
    --------
        Tuple[np.ndarray, Tuple[int, int]]
    """
    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnt = sorted(contours, key = cv2.contourArea)[-1]
    _, (w,h), angle = cv2.minAreaRect(cnt)
    w,h = int(w), int(h)
    if angle > 45 or angle < -45:
        w,h = h,w
    return cnt, (w,h)


def extract(img : np.ndarray, 
            cnt : np.ndarray, 
            w : int, 
            h : int
            ) -> np.ndarray:
    """
    crops out the page from the entire image

    Args:
    -----
        img (np.ndarray): input image
        cnt (np.ndarray): contour
        w (int): width of min area rect
        h (int): hieght of min area rect

    Returns:
    --------
        np.ndarray: the cropped out image
    """
    cnt = cnt.reshape(cnt.shape[0], cnt.shape[-1])
    # the corners have extreme x, y coordinates
    s1 = sorted(cnt, key = lambda x : (x[0], x[1]))
    s2 = sorted(cnt, key = lambda x : (x[1], x[0]))
    corner1, corner3 = s1[0], s1[-1]
    corner2, corner4 = s2[0], s2[-1]

    corners = np.array([corner1, corner2, corner3, corner4])
    target_corners = np.array([(0,0), (w,0), (w,h), (0,h)])

    H, _ = cv2.findHomography(corners, target_corners, params = None)

    transformed_image = cv2.warpPerspective(
    img, H, (img.shape[1], img.shape[0]))

    transformed_image = transformed_image[:h, :w]

    return transformed_image

def transform(img : np.ndarray) -> np.ndarray:
    """
    converts the image to B&W

    Args:
    -----
        img (np.ndarray): input image 

    Returns:
        np.ndarray: B&W image
    """
    T = cv2.GaussianBlur(img, (11,11),0)-10
    return (img > T).astype(np.uint8) * 255

if __name__ == "__main__":
    img = cv2.imread("image.jpg")
    gray = grayScale(img)
    blurred = blur(gray)
    edged = binary(blurred)
    cnt, (w,h) = retBiggestContour(edged)
    img = extract(gray, cnt, w, h)
    img = transform(img)
    cv2.imwrite("transformed.jpg", img)

