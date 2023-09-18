import cv2
import numpy as np
from datetime import datetime

from ..utils.utils import printwtime

##############################
# Find Homography
##############################


def _find_homography_from_descriptors(target, kp_target, des_target,
                                      source, kp_source, des_source,
                                      keep_match=0.75, mode="homography",
                                      verbose=False):
    """
    Find homography, given SIFT descriptors of two images.
    """
    if verbose:
        print(datetime.now().strftime("%H:%M:%S"), "- Match descriptors")
    matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
    matches = matcher.knnMatch(des_source, des_target, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < keep_match * n.distance:
            good_matches.append([m])

    if verbose:
        printwtime(f"Found {len(good_matches)} good matches")

    if len(good_matches) < 5:
        printwtime("Found too few good matches! Returnining None.")
        return None, None

    if verbose:
        printwtime("Find Homography")
    source_matched_kpts = np.float32([kp_source[m[0].queryIdx].pt
                                      for m in good_matches])
    target_matched_kpts = np.float32([kp_target[m[0].trainIdx].pt
                                      for m in good_matches])

    if mode == "homography":
        to_target, _ = cv2.findHomography(source_matched_kpts,
                                          target_matched_kpts,
                                          cv2.RANSAC)
        from_target = np.linalg.inv(to_target)
        from_target = from_target / from_target[2, 2]
        return to_target, from_target

    elif mode == "partialaffine":
        to_target, _ = cv2.estimateAffinePartial2D(source_matched_kpts,
                                                   target_matched_kpts,
                                                   method=cv2.RANSAC,
                                                   ransacReprojThreshold=5)
        from_target, _ = cv2.estimateAffinePartial2D(target_matched_kpts,
                                                     source_matched_kpts,
                                                     method=cv2.RANSAC,
                                                     ransacReprojThreshold=5)
        return to_target, from_target
    elif mode == "affine":
        to_target, _ = cv2.estimateAffine2D(source_matched_kpts,
                                            target_matched_kpts,
                                            method=cv2.RANSAC,
                                            ransacReprojThreshold=5)
        from_target, _ = cv2.estimateAffine2D(target_matched_kpts,
                                              source_matched_kpts,
                                              method=cv2.RANSAC,
                                              ransacReprojThreshold=5)
        return to_target, from_target
    else:
        assert False


def find_homography(target, source, keep_match=0.75, mode="homography",
                    verbose=False):
    """
    Find homography for two images, uses SIFT.
    Can use general homography, affine homography (no position
    dependent scaling), or partial affine homography (also no shear).
    """
    if mode not in ["homography", "partialaffine", "affine"]:
        raise ValueError("Unknown mode!")
    if verbose:
        printwtime("Find descriptors")

    descriptor = cv2.SIFT_create()

    source = (source/source.max()*255).astype('uint8')
    kp_source, des_source = descriptor.detectAndCompute(source, None)
    target = (target/target.max()*255).astype('uint8')
    kp_target, des_target = descriptor.detectAndCompute(target, None)

    return _find_homography_from_descriptors(target, kp_target, des_target,
                                             source, kp_source, des_source,
                                             keep_match, mode, verbose)


def find_homographies(target, sources, keep_match=0.75, mode="homography",
                      verbose=False):
    """
    Same as find_homography, but for multiple sources and reuses 
    target descriptors.
    """
    if mode not in ["homography", "partialaffine", "affine"]:
        raise ValueError("Unknown mode!")

    if verbose:
        printwtime("Find descriptors")

    descriptor = cv2.SIFT_create()
    target = (target/target.max()*255).astype('uint8')
    kp_target, des_target = descriptor.detectAndCompute(target, None)

    homographies = []
    for i, source_ in enumerate(sources):
        if verbose:
            printwtime(f"Apply to source {i} of {len(sources)}.")
        source = (source_/source_.max()*255).astype('uint8')
        kp_source, des_source = descriptor.detectAndCompute(source, None)
        homographies.append(_find_homography_from_descriptors(
            target, kp_target, des_target,
            source, kp_source, des_source,
            keep_match, mode, verbose
        ))

    return homographies

##############################
# Homography Utils
##############################


def scale_homography(homography, scale_target, scale_source):
    """
    Scale homography to target, if registration was done by downsampling
    with scale_target and scale_source.
    """
    if homography.shape[0] == 3:  # Homography
        scaler_target = np.array([scale_target, scale_target, 1])[:, None]
        scaler_source = np.array([1/scale_source, 1/scale_source, 1])[None, :]
        return scaler_target * homography * scaler_source

    elif homography.shape[0] == 2:  # Affine Homography
        scaler_target = np.array([scale_target, scale_target])[:, None]
        scaler_source = np.array([1/scale_source, 1/scale_source, 1])[None, :]
        return scaler_target * homography * scaler_source
    else:
        raise ValueError("Invalid homography!")


def homography_shift_target(homography, xshift=0, yshift=0):
    """
    Return homography, with origin of the target shifted.
    """
    homography_shifted = homography.copy()
    homography_shifted[0, 2] += xshift
    homography_shifted[1, 2] += yshift
    return homography_shifted

##############################
# Apply Homography
##############################


def warp_image(target, source, homography):
    """
    Warp image with homography.
    """
    if homography.shape[0] == 3:  # Homography
        warped_image = cv2.warpPerspective(source, homography,
                                           (target.shape[1], target.shape[0]))
        return warped_image

    elif homography.shape[0] == 2:  # Affine Homography
        warped_image = cv2.warpAffine(source, homography,
                                      (target.shape[1], target.shape[0]))
        return warped_image

    else:
        raise ValueError("Invalid homography!")


def transform_coordinate(homography, x, y):
    """
    Transform single point with homography.
    Ignores z, only appropriate if angles are very small!
    """
    xp = homography[0, 0]*x + homography[0, 1]*y + homography[0, 2]
    yp = homography[1, 0]*x + homography[1, 1]*y + homography[1, 2]

    if homography.shape[0] == 3:  # Homography
        sf = homography[2, 0]*x + homography[2, 1]*y + homography[2, 2]
        return xp/sf, yp/sf

    elif homography.shape[0] == 2:  # Affine Homography
        return xp, yp

    else:
        raise ValueError("Invalid homography!")


def get_transformed_corners(source, homography):
    """
    After image source was registered to some target with homography,
    get the points its corners transform to.
    Sorted and with duplicates such that plotting them yields a rectangle.
    x coordinate in [:,0], y in [:,1]
    """
    corners = np.reshape(
        np.asarray([[transform_coordinate(homography, i, j)
                     for j in [0, source.shape[0]]]
                    for i in [0, source.shape[1]]]),
        (4, 2)
    )
    corners = corners[[0, 1, 3, 2, 0]]
    return corners
