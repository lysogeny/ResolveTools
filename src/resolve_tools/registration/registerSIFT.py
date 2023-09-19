import cv2
import numpy as np
import copy

from ..utils.utils import printwtime


def _prepare_image(img):
    return (img/img.max()*255).astype("uint8")


def _find_matches(target, kp_target, des_target,
                  source, kp_source, des_source,
                  keep_match):
    matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
    matches = matcher.knnMatch(des_source, des_target, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < keep_match * n.distance:
            good_matches.append([m])
    return good_matches


class AbstractKeypointObject:
    @property
    def kps_source(self):
        return self.kps[0]

    @property
    def kps_target(self):
        return self.kps[1]

    @property
    def des_source(self):
        return self.des[0]

    @property
    def des_target(self):
        return self.des[1]

    @property
    def kps_match_source(self):
        return self.kps_match[0]

    @property
    def kps_match_target(self):
        return self.kps_match[1]


class AbstractHomography(AbstractKeypointObject):
    """
    Homography abstraction.
    """

    def __init__(self, target, source):
        self.target = _prepare_image(target)
        self.source = _prepare_image(source)
        # TODO: Do I actually do something with the scales here?
        self.descriptor = cv2.SIFT_create()
        self.good = []
        self.kps = (None, None)
        self.kps_match = (None, None)
        self.des = (None, None)
        self.to_target = None
        self.to_source = None

    def find_descriptors(self):
        source = self.descriptor.detectAndCompute(self.source, None)
        target = self.descriptor.detectAndCompute(self.target, None)
        self.kps = (source[0], target[0])
        self.des = (source[1], target[1])

    @property
    def homography(self):
        return self.to_target, self.to_source

    def match_descriptors(self, keep_match=0.75):
        matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
        matches = matcher.knnMatch(self.des_source, self.des_target, k=2)
        self.good = []
        for m, n in matches:
            if m.distance < keep_match * n.distance:
                self.good.append([m])
        # TODO: Handle few matches
        kps_source_match = np.float32([self.kps_source[m[0].queryIdx].pt
                                       for m in self.good])
        kps_target_match = np.float32([self.kps_target[m[0].trainIdx].pt
                                       for m in self.good])
        self.kps_match = (kps_source_match, kps_target_match)

    def find_homography(self):
        self.to_target = self.estimate_forward()
        self.to_source = self.estimate_backward()

    def estimate_forward(self):
        raise NotImplementedError

    def estimate_backward(self):
        raise NotImplementedError

    def transform(self):
        raise NotImplementedError

    def warp_image(self, img):
        raise NotImplementedError

    def scale(self, target, source):
        raise NotImplementedError

    @property
    def transformed_corners(self):
        """
        After image source was registered to some target with homography,
        get the points its corners transform to.
        Sorted and with duplicates such that plotting them yields a rectangle.
        x coordinate in [:,0], y in [:,1]
        """
        corners = np.reshape(
            np.asarray([[self.transform_coordinate(self.to_target, i, j)
                         for j in [0, self.source.shape[0]]]
                        for i in [0, self.source.shape[1]]]),
            (4, 2)
        )
        corners = corners[[0, 1, 3, 2, 0]]
        return corners

    @property
    def backtransformed_corners(self):
        """
        After image source was registered to some target with homography,
        get the points its corners transform to.
        Sorted and with duplicates such that plotting them yields a rectangle.
        x coordinate in [:,0], y in [:,1]
        """
        corners = np.reshape(
            np.asarray([[self.transform_coordinate(self.to_source, i, j)
                         for j in [0, self.target.shape[0]]]
                        for i in [0, self.target.shape[1]]]),
            (4, 2)
        )
        corners = corners[[0, 1, 3, 2, 0]]
        return corners


class GeneralHomography(AbstractHomography):
    def estimate_forward(self):
        transform, _ = cv2.findHomography(self.kps_match_source,
                                          self.kps_match_target,
                                          cv2.RANSAC)
        return transform

    def estimate_backward(self):
        transform = np.linalg.inv(self.to_target)
        transform = transform / transform[2, 2]
        return transform

    def warp_image(self, img):
        warped_image = cv2.warpPerspective(img, self.to_target,
                                           (self.target.shape[1],
                                            self.target.shape[0]))
        return warped_image

    def scale(self, scale_target, scale_source, inplace=True):
        if not inplace:
            self = copy.copy(self)
        scaler_target = np.array([scale_target, scale_target, 1])[:, None]
        scaler_source = np.array([1/scale_source, 1/scale_source, 1])[None, :]
        self.to_target = scaler_target * self.to_target * scaler_source
        self.from_target = scaler_source * self.from_target * scaler_target
        if not inplace:
            return self

    def transform(self, x, y):
        homography = self.to_target
        xp = homography[0, 0]*x + homography[0, 1]*y + homography[0, 2]
        yp = homography[1, 0]*x + homography[1, 1]*y + homography[1, 2]
        sf = homography[2, 0]*x + homography[2, 1]*y + homography[2, 2]
        return xp/sf, yp/sf


class AbstractAffineHomography(AbstractHomography):
    args = dict(method=cv2.RANSAC, ransacReprojThreshold=5)

    def estimate_forward(self):
        raise NotImplementedError

    def estimate_backward(self):
        raise NotImplementedError

    def warp_image(self, img):
        warped_image = cv2.warpAffine(img, self.to_target,
                                      (self.target.shape[1],
                                       self.target.shape[0]))
        return warped_image

    def scale(self, scale_target, scale_source, inplace=True):
        if not inplace:
            self = copy.copy(self)
        scaler_target = np.array([scale_target, scale_target])[:, None]
        scaler_source = np.array([1/scale_source, 1/scale_source, 1])[None, :]
        self.to_target = scaler_target * self.to_target * scaler_source
        self.from_target = scaler_source * self.from_target * scaler_target
        if not inplace:
            return self

    def transform(self, x, y):
        homography = self.to_target
        xp = homography[0, 0]*x + homography[0, 1]*y + homography[0, 2]
        yp = homography[1, 0]*x + homography[1, 1]*y + homography[1, 2]
        return xp, yp


class AffineHomography(AbstractAffineHomography):
    def estimate_forward(self):
        transform, _ = cv2.estimateAffine2D(self.kps_match_source,
                                            self.kps_match_target,
                                            *self.args)
        return transform

    def estimate_backward(self):
        transform, _ = cv2.estimateAffine2D(self.kps_match_source,
                                            self.kps_match_target,
                                            *self.args)
        return transform


class PartialAffineHomography(AbstractHomography):
    def estimate_forward(self):
        transform, _ = cv2.estimateAffinePartial2D(self.kps_match_source,
                                                   self.kps_match_target,
                                                   *self.args)
        return transform

    def estimate_backward(self):
        transform, _ = cv2.estimateAffinePartial2D(self.kps_match_source,
                                                   self.kps_match_target,
                                                   *self.args)
        return transform


def find_homography(target, source, keep_match=0.75, mode="homography",
                    verbose=False):
    """
    Legacy Interface.
    Use GeneralHomography, AffineHomography and PartialAffineHomography instead
    """
    if mode == "Homography":
        homography = GeneralHomography(target, source)
    elif mode == "affine":
        homography = AffineHomography(target, source)
    elif mode == "partialaffine":
        homography = PartialAffineHomography(target, source)
    else:
        raise ValueError("Unknown method!")
    if verbose:
        printwtime("Find descriptors")
    homography.find_descriptors()
    if verbose:
        printwtime("Match descriptors")
    homography.match_descriptors(keep_match=keep_match)
    if verbose:
        printwtime("Matched descriptors")
    homography.find_homography()
    if verbose:
        printwtime(f"Etsimated {mode} homography")
    return homography.homography
