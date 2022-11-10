import cv2
import numpy as np
from datetime import datetime

def find_homography(target, source, keep_match=0.75, mode="homography", verbose=False, convertto256=True, N=5000, method="SIFT"):
    if not mode in ["homography", "partialaffine", "affine"]: raise ValueError("Unknown mode!")
    if not method in ["ORB", "SIFT"]: raise ValueError("Unknown mode!")
    
    if verbose: print(datetime.now().strftime("%H:%M:%S"),"- Find Descriptors")
    if method == "SIFT":
        descriptor = cv2.SIFT_create()
    else:
        descriptor = cv2.ORB_create(N)
    source = (source/source.max()*255).astype('uint8')
    kp1, des1 = descriptor.detectAndCompute(source, None)
    target = (target/target.max()*255).astype('uint8')
    kp2, des2 = descriptor.detectAndCompute(target, None)
    
    if verbose: print(datetime.now().strftime("%H:%M:%S"),"- Match descriptors")
    matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck = False)
    matches = matcher.knnMatch(des1, des2, k = 2)
    
    good_matches = []
    for m,n in matches:
        if m.distance < keep_match * n.distance:
            good_matches.append([m])
    if verbose: print(datetime.now().strftime("%H:%M:%S"),"- Found",len(good_matches),"good matches.")
    
    if verbose: print(datetime.now().strftime("%H:%M:%S"),"- Find Homography")
    ref_matched_kpts = np.float32([kp1[m[0].queryIdx].pt for m in good_matches])
    sensed_matched_kpts = np.float32([kp2[m[0].trainIdx].pt for m in good_matches])
    
    if mode=="homography":
        homography_toTarget, _ = cv2.findHomography(ref_matched_kpts, sensed_matched_kpts, cv2.RANSAC)
        # Invert to use them interchangeably
        #homography_toRes, _ = cv2.findHomography(ref_matched_kpts, sensed_matched_kpts, cv2.RANSAC)
        homography_fromTarget = np.linalg.inv(homography_toTarget)
        homography_fromTarget = homography_fromTarget/homography_fromTarget[2,2]
        return homography_toTarget, homography_fromTarget
    
    elif mode=="partialaffine":
        homography_toTarget, _  = cv2.estimateAffinePartial2D(ref_matched_kpts,
            sensed_matched_kpts, method = cv2.RANSAC, ransacReprojThreshold = 5)
        homography_fromTarget, _  = cv2.estimateAffinePartial2D(sensed_matched_kpts,
            ref_matched_kpts, method = cv2.RANSAC, ransacReprojThreshold = 5)
        return homography_toTarget, homography_fromTarget
    elif mode=="affine":
        homography_toTarget, _  = cv2.estimateAffine2D(ref_matched_kpts,
            sensed_matched_kpts, method = cv2.RANSAC, ransacReprojThreshold = 5)
        homography_fromTarget, _  = cv2.estimateAffine2D(sensed_matched_kpts,
            ref_matched_kpts, method = cv2.RANSAC, ransacReprojThreshold = 5)
        return homography_toTarget, homography_fromTarget
    else:
        assert False

def warp_image(target, source, homography, mode="homography"):
    if not mode in ["homography", "partialaffine", "affine"]:
        raise ValueError("Unknown mode!")
    
    if mode=="homography":
        warped_image = cv2.warpPerspective(source, homography, (target.shape[1], target.shape[0]))
        return warped_image
    
    elif mode in ["partialaffine", "affine"]:
        warped_image = cv2.warpAffine(source, homography, (target.shape[1], target.shape[0]))
        return warped_image
        
    else:
        assert False

def transform_coordinate(homography, x, y, mode="homography"):
    if not mode in ["homography", "partialaffine", "affine"]:
        raise ValueError("Unknown mode!")
    
    xp = homography[0,0]*x + homography[0,1]*y + homography[0,2]
    yp = homography[1,0]*x + homography[1,1]*y + homography[1,2]
    
    if mode=="homography":
        sf = homography[2,0]*x + homography[2,1]*y + homography[2,2]
        return xp/sf, yp/sf
    
    elif mode in ["partialaffine", "affine"]:
        return xp, yp
        
    else:
        assert False
