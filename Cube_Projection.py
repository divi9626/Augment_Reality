import numpy as np
import cv2 as cv
import copy
import imutils
####################################################
def get_corners(contour):
    epsilon = 0.05
    count = 0
    while True:
        perimeter = cv.arcLength(contour,True)
        perimeter = epsilon*perimeter
        if perimeter > 100 or perimeter < 1:
            return None
        approx = cv.approxPolyDP(contour,perimeter,True)
        print(perimeter)
        hull = cv.convexHull(approx)
        if len(hull) == 4:
            return hull
        else:
            if len(hull) > 4:
                epsilon += 0.01
            else:
                epsilon -= 0.01
        if count > 10:
            return []
###########################################################       
def ar_tag_contours(contours, contour_hierarchy):
    paper_contours_ind = []
    ar_tag_contours = []
    for ind, contour in enumerate(contour_hierarchy[0]):
        if contour[3] == 0:
            paper_contours_ind.append(ind)
            
    if (len(paper_contours_ind) > 3):
        return None
    for ind in paper_contours_ind:
        ar_tag_contour_ind = contour_hierarchy[0][ind][2]
        ar_tag_contours.append(contours[ar_tag_contour_ind])
        
    return ar_tag_contours
###############################################################
def arrange(corners):
    corners = corners.reshape((4, 2))
    new = np.zeros((4, 1, 2), dtype=np.int32)
    add = corners.sum(1)
    
    new[0] = corners[np.argmin(add)]
    new[2] =corners[np.argmax(add)]
    diff = np.diff(corners, axis=1)
    new[1] =corners[np.argmin(diff)]
    new[3] = corners[np.argmax(diff)]
    return new
###########################################################
def homograph(src_plane, dest_plane):
    A = []

    for i in range(0, len(src_plane)):
        x, y = src_plane[i][0], src_plane[i][1]
        xp, yp = dest_plane[i][0], dest_plane[i][1]
        A.append([-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp])
        A.append([0, 0, 0, -x, -y, -1, x * yp, y * yp, yp])
        
    A = np.asarray(A) 
    
    U, S, VT = np.linalg.svd(A)
    
    # normalizing
    l = VT[-1, :] / VT[-1, -1]
    
    H = l.reshape(3,3)

    return H
##################################################################
def project_cube(dest_points, contour, src_img):
    
    K = np.array([[1406.08415449821,0,0],
              [2.20679787308599, 1417.99930662800,0],
              [1014.13643417416, 566.347754321696,1]]).T
    
    src_points = np.float32([[0, 0], [200, 0],[200, 200], [0, 200]])
    
    H = homograph(src_points, dest_points)
    R_mat, t_vec = get_rotation_and_translation_matrix(K, H)
    axis_points = np.float32([[0, 0, 0], 
                       [200, 0, 0], 
                       [200, 200, 0], 
                       [0, 200, 0], 
                       [0, 0, -200], 
                       [200, 0, -200], 
                       [200, 200, -200],
                       [0, 200, -200]])
    
    proj_corner_points, jacobian = cv.projectPoints(axis_points, R_mat, t_vec, K, np.zeros((1, 4)))
    dest_img = draw_cube(src_img, contour, proj_corner_points)
    return dest_img
################################################################
def get_rotation_and_translation_matrix(K, H):
    K_inv = np.linalg.inv(K)
    lam = (np.linalg.norm(np.dot(K_inv, H[:,0])) + np.linalg.norm(np.dot(K_inv, H[:,1])))/2
    lam = 1/lam

    B_tilde = np.dot(K_inv, H)
    B = lam*B_tilde

    r1 = lam*B[:,0]
    r2 = lam*B[:,1]
    r3 = np.cross(r1,r2)/lam
    t = np.array([lam*B[:,2]]).T
    R = np.array([r1,r2,r3]).T
    M = np.hstack([R, t])

    #P = np.dot(K, M)
    return R, t
#########################################################################3
def draw_cube(img, bottom_contour, corner_points):
    corner_points = np.int32(corner_points).reshape(-1,2)
    
    # draw bottom_contour
    img = cv.drawContours(img, bottom_contour, -1, (0,255,0), 3)
    
    # draw lines to join bottom and top corners
    for i,j in zip(range(4),range(4,8)):
        img = cv.line(img, tuple(corner_points[i]), tuple(corner_points[j]), (255), 2)
    
    # draw top_contour
    img = cv.drawContours(img, [corner_points[4:]], -1, (0,0,255), 2)
    return img
###############################################################
cap = cv.VideoCapture('multipleTags.mp4')
if cap.isOpened() == False:
    print("Error opening the image")
    
img = None
count = 0

while cap.isOpened():
    count += 1
    ret, frame = cap.read()
    if ret == False:
        break
    try:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    except:
        break
    gray = cv.resize(gray,(600,600))
    frame = cv.resize(frame,(600,600))
    _,thresh = cv.threshold(gray,200,255, cv.THRESH_BINARY_INV)
    
    img_copy_for_cube = copy.deepcopy(frame)
    image = thresh
    img = copy.deepcopy(image)
    image_c = gray
    img_c = copy.deepcopy(image_c)
    #if count == 50:
        #img_c = frame
        #img = thresh
        #img_c = gray
        
    width, height = 80,80
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    get_ar_tag_contours = ar_tag_contours(contours,hierarchy)
    warp_test = img_c
    if get_ar_tag_contours is not None:

        for contour in get_ar_tag_contours:
            #cv.drawContours(img_c, contours, -1, (0,0,255), 1)

            New = get_corners(contour)
            if New is not None:# get corners of the tag
                source  = arrange(New)  # arrange the corners in proper orientation (for warping)
                #show_corners(New)
                source = np.float32(source)
                cor = []
                for i in source:
                    cor.append(i[0])
                cor = np.asarray(cor)
                
                cor = np.float32(cor)
                img_copy_for_cube = project_cube(cor, contour, img_copy_for_cube)
                cube_display_img = copy.deepcopy(img_copy_for_cube)
    
                cv.imshow('Cube projection Image', cube_display_img)
    
    if cv.waitKey(1) == 27:
        break
#img = np.float32(img)
print(gray.shape)
cap.release()
cv.destroyAllWindows()