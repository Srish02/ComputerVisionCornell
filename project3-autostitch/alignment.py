import math
import random

import cv2
import numpy as np
from scipy.spatial import distance

eTranslate = 0
eHomography = 1


def computeHomography(f1, f2, matches, A_out=None):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        A_out -- ignore this parameter. If computeHomography is needed
                 in other TODOs, call computeHomography(f1,f2,matches)
    Output:
        H -- 2D homography (3x3 matrix)
        Takes two lists of features, f1 and f2, and a list of feature
        matches, and estimates a homography from image 1 to image 2 from the matches.
    '''
    num_matches = len(matches)

    # Dimensions of the A matrix in the homogenous linear
    # equation Ah = 0
    num_rows = 2 * num_matches
    num_cols = 9
    A_matrix_shape = (num_rows,num_cols)
    A = np.zeros(A_matrix_shape)

    for i in range(len(matches)):
        m = matches[i]
        (a_x, a_y) = f1[m.queryIdx].pt
        (b_x, b_y) = f2[m.trainIdx].pt

        #BEGIN TODO 2
        #Fill in the matrix A in this loop.
        #Access elements using square brackets. e.g. A[0,0]
        #TODO-BLOCK-BEGIN
    
        # xi, yi, 1
        xy_term = [a_x, a_y, 1]
        # -xi'xi, -xi'yi, -xi'
        xy_xprime_term = [-a_x*b_x, -a_y*b_x, -b_x]
        # -yi'xi, -yi'yi, -yi'
        xy_yprime_term = [-a_x*b_y, -a_y*b_y, -b_y]

        # first row
        A[2*i, :3] = xy_term
        A[2*i, 6:] = xy_xprime_term
        # second row
        A[(2*i)+1, 3:6] = xy_term
        A[(2*i)+1, 6:] = xy_yprime_term
        
        #TODO-BLOCK-END
        #END TODO

    U, s, Vt = np.linalg.svd(A)

    if A_out is not None:
        A_out[:] = A

    #s is a 1-D array of singular values sorted in descending order
    #U, Vt are unitary matrices
    #Rows of Vt are the eigenvectors of A^TA.
    #Columns of U are the eigenvectors of AA^T.

    #Homography to be calculated
    H = np.eye(3)

    #BEGIN TODO 3
    #Fill the homography H with the appropriate elements of the SVD
    #TODO-BLOCK-BEGIN
    # s is in descending order
    # so smallest eigenvalue is the last one, 
    # and corresponding eigenvector is the last row in Vt

    H = Vt[-1].reshape((3,3))
    # H = H / H[2,2]
    #TODO-BLOCK-END
    #END TODO

    return H

def alignPair(f1, f2, matches, m, nRANSAC, RANSACthresh):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        m -- MotionModel (eTranslate, eHomography)
        nRANSAC -- number of RANSAC iterations
        RANSACthresh -- RANSAC distance threshold

    Output:
        M -- inter-image transformation matrix
        Repeat for nRANSAC iterations:
            Choose a minimal set of feature matches.
            Estimate the transformation implied by these matches
            count the number of inliers.
        For the transformation with the maximum number of inliers,
        compute the least squares motion estimate using the inliers,
        and return as a transformation matrix M.
    '''

    #BEGIN TODO 4
    #Write this entire method.  You need to handle two types of
    #motion models, pure translations (m == eTranslation) and
    #full homographies (m == eHomography).  However, you should
    #only have one outer loop to perform the RANSAC code, as
    #the use of RANSAC is almost identical for both cases.

    #Your homography handling code should call compute_homography.
    #This function should also call get_inliers and, at the end,
    #least_squares_fit.
    #TODO-BLOCK-BEGIN

    if(m == eTranslate):
        s = 2
    elif(m == eHomography):
        s = 4
    else:
        s = 0

    models = []
    num_inliers = []

    for n in range(nRANSAC):
        # compute transform for eTranslation
        transform = np.eye(3)
        k_t = 0
        if(m == eHomography):
            feat_matches=random.sample(matches,s) 
            transform_next = computeHomography(f1, f2, feat_matches)
        else:
            feat_matches=random.sample(matches,s)  
            transform_next=leastSquaresFit(f1,f2,feat_matches,m,[0,1])
            # get the first match
            # (a_x, a_y) = f1[matches[feat_matches[0]].queryIdx].pt
            # (b_x, b_y) = f2[matches[feat_matches[0]].trainIdx].pt
            # # since its a translation, 
            # # the movement is just the x2 - x1 and y2 - y1 terms
            # transform[0,2] = b_x - a_x
            # transform[1,2] = b_y - a_y
            
        in_next = getInliers(f1, f2, matches, transform_next, RANSACthresh)

        if len(in_next) > len(models) :
            models = in_next
            transform = transform_next
        # models.append(transform)
        # num_inliers.append(len(getInliers(f1, f2, feat_matches, transform, RANSACthresh)))
   
    M = leastSquaresFit(f1, f2, matches, m, models)
    M = M/M[2][2]
    # max_inlier_ind = num_inliers.index(max(num_inliers))
    # M = models[max_inlier_ind]
    #TODO-BLOCK-END
    #END TODO
    return M

def getInliers(f1, f2, matches, M, RANSACthresh):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        M -- inter-image transformation matrix
        RANSACthresh -- RANSAC distance threshold

    Output:
        inlier_indices -- inlier match indices (indexes into 'matches')

        Transform the matched features in f1 by M.
        Store the match index of features in f1 for which the transformed
        feature is within Euclidean distance RANSACthresh of its match
        in f2.
        Return the array of the match indices of these features.
    '''

    inlier_indices = []

    for i in range(len(matches)):
        #BEGIN TODO 5
        #Determine if the ith matched feature f1[id1], when transformed
        #by M, is within RANSACthresh of its match in f2.
        #If so, append i to inliers
        #TODO-BLOCK-BEGIN
        id1 = matches[i].queryIdx
        id2 = matches[i].trainIdx
        f1_mat = [f1[id1].pt[0], f1[id1].pt[1], 1]
        transformed_f1 = np.dot(M, f1_mat)
        transformed_f1 /= transformed_f1[-1]
        f2_mat = [f2[id2].pt[0], f2[id2].pt[1], 1]
        if(distance.euclidean(transformed_f1, f2_mat) <= RANSACthresh):
            inlier_indices.append(i)
        #TODO-BLOCK-END
        #END TODO
    
    print(len(inlier_indices))
    print(inlier_indices)

    print(inlier_indices)
    print(len(inlier_indices))
    return inlier_indices

def leastSquaresFit(f1, f2, matches, m, inlier_indices):
    '''
    Input:
        f1 -- list of cv2.KeyPoint objects in the first image
        f2 -- list of cv2.KeyPoint objects in the second image
        matches -- list of cv2.DMatch objects
            DMatch.queryIdx: The index of the feature in the first image
            DMatch.trainIdx: The index of the feature in the second image
            DMatch.distance: The distance between the two features
        m -- MotionModel (eTranslate, eHomography)
        inlier_indices -- inlier match indices (indexes into 'matches')

    Output:
        M - transformation matrix

        Compute the transformation matrix from f1 to f2 using only the
        inliers and return it.
    '''

    # This function needs to handle two possible motion models,
    # pure translations (eTranslate)
    # and full homographies (eHomography).

    M = np.eye(3)

    if m == eTranslate:
        #For spherically warped images, the transformation is a
        #translation and only has two degrees of freedom.
        #Therefore, we simply compute the average translation vector
        #between the feature in f1 and its match in f2 for all inliers.

        u = 0.0
        v = 0.0

        for i in range(len(inlier_indices)):
            #BEGIN TODO 6
            #Use this loop to compute the average translation vector
            #over all inliers.
            #TODO-BLOCK-BEGIN
            inlier = matches[inlier_indices[i]]
            u = u + f2[inlier.trainIdx].pt[0] - f1[inlier.queryIdx].pt[0]
            v = v + f2[inlier.trainIdx].pt[1] - f1[inlier.queryIdx].pt[1]
            #TODO-BLOCK-END
            #END TODO

        u /= len(inlier_indices)
        v /= len(inlier_indices)

        M[0,2] = u
        M[1,2] = v

    elif m == eHomography:
        #BEGIN TODO 7
        #Compute a homography M using all inliers.
        #This should call computeHomography.
        #TODO-BLOCK-BEGIN
        match_inliers = [matches[ind] for ind in inlier_indices]
        M = computeHomography(f1, f2, match_inliers)
        M = M/M[2][2]
        #TODO-BLOCK-END
        #END TODO

    else:
        raise Exception("Error: Invalid motion model.")

    return M

