import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

query_path = './matchimg/cap_result/point_1.jpg' # queryImage
source_path = './matchimg/DJI_0167.MP4_000296462.jpg' # trainImage

def matcher_BRISK_BF(im0, GCP_list) :
    
    MIN_MATCH_COUNT = 5
    box_point_MD_list = []

    for query_path in GCP_list:
        img1 = cv2.imread(query_path,0)
        img2 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
        img3 = im0

        ## --------------------------------------------------- master brisk/bf strat
        # start = time.time()  # 시작 시간 저장

        brisk = cv2.BRISK_create(thresh=60) # 밝기 값을 통하여 설정(corner의 후보 수 결정), default : 30
        (kp1,des1) = brisk.detectAndCompute(img1, None)
        (kp2,des2) = brisk.detectAndCompute(img2, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING) #, crossCheck=True)
        matches = bf.knnMatch(des1,des2,k=2)

        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
        if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0) # 투영 유도 방법, 객체를 찾아줌
            matchesMask = mask.ravel().tolist()
            
            h,w = img1.shape
            d = 1
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M) # 3*3 행렬로 이미지의 키포인트를 찾아준다(이미지를 와핑해준다.).
            # print('dst : ')
            # print(dst)
            img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        else:   
            print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
            matchesMask = None


        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                        singlePointColor = None,
                        matchesMask = matchesMask, # inliers 만 그려라
                        flags = 2)
        # print()
        # print('len(matchesMask)')
        # print(len(matchesMask))
        # print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
        # img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
        # plt.imshow(img3, 'gray')
        # plt.show()
        box_point_LH = tuple(list(map(int,sum(np.array(dst[0]).tolist(),[]))))
        box_point_RL = tuple(list(map(int,sum(np.array(dst[2]).tolist(),[]))))
        box_point_MD = [(list(box_point_LH)[0]+list(box_point_RL)[0])/2, (list(box_point_LH)[1]+list(box_point_RL)[1])/2]
        box_point_MD = list(map(int,box_point_MD ))
        # print('box_point_LH : ')
        # print(box_point_LH)
        # print('box_point_RL : ')
        # print(box_point_LH)
        # print('box_point_MD : ')
        # print(box_point_MD)

        cv2.circle(im0, tuple(box_point_MD),3, (0, 255, 0) , cv2.FILLED, cv2.LINE_4)
        cv2.rectangle(im0,box_point_LH, box_point_RL, (255,255,255), 3)
        box_point_MD_list.append(tuple(box_point_MD))
        # cv2.imshow('rec_img', rec_img)
        # cv2.imshow('point_img', img3)
        # cv2.waitKey(0)
    return box_point_MD_list
## --------------------------------------------------- master brisk/bf end

'''
## --------------------------------------------------- test sift/flann start
start = time.time()  # 시작 시간 저장
sift = cv2.xfeatures2d.SIFT_create() # SIFT 특성 이용

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0) # 투영 유도 방법 
    matchesMask = mask.ravel().tolist()
    
    h,w = img1.shape
    d = 1
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
else:   
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None


draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # inliers 만 그려라
                   flags = 2)

print()
print('len(matchesMask)')
print(len(matchesMask))
print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간

img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
plt.imshow(img3, 'gray')
plt.show()
## --------------------------------------------------- test sift/flann end
'''

'''
## --------------------------------------------------- test sift/bf start
start = time.time()  # 시작 시간 저장
sift = cv2.xfeatures2d.SIFT_create() # SIFT 특성 이용

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

bf = cv2.BFMatcher(cv2.NORM_L1)

matches = bf.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0) # 투영 유도 방법 
    matchesMask = mask.ravel().tolist()
    
    h,w = img1.shape
    d = 1
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
else:   
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None


draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # inliers 만 그려라
                   flags = 2)

print()
print('len(matchesMask)')
print(len(matchesMask))
print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간

img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
plt.imshow(img3, 'gray')
plt.show()
## --------------------------------------------------- test sift/bf end
'''
'''
## --------------------------------------------------- test BRISK/FLANN start
start = time.time()  # 시작 시간 저장

brisk = cv2.BRISK_create()
kp1,des1 = brisk.detectAndCompute(img1,None)
kp2,des2 = brisk.detectAndCompute(img2, None)


FLANN_INDEX_LSH = 6
index_params =  dict ( algorithm  =  FLANN_INDEX_LSH , 
                    table_number  =  6 ,  # 12 
                    key_size  =  12 ,      # 20 
                    multi_probe_level  =  1 )  # 2
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)

good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)
if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0) # 투영 유도 방법 
    matchesMask = mask.ravel().tolist()
    
    h,w = img1.shape
    d = 1
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
else:   
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None


draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # inliers 만 그려라
                   flags = 2)
print()
print('len(matchesMask)')
print(len(matchesMask))
print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
plt.imshow(img3, 'gray')
plt.show()
## --------------------------------------------------- test BRISK/FLANN end
'''
'''
## --------------------------------------------------- test ORB/BF start
start = time.time()  # 시작 시간 저장
# Initiate ORB detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# des1 = np.float32(des1)
# des2 = np.float32(des2)

bf = cv2.BFMatcher(cv2.NORM_HAMMING)

matches = bf.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0) # 투영 유도 방법 
    matchesMask = mask.ravel().tolist()
    
    h,w = img1.shape
    d = 1
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
else:   
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None


draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # inliers 만 그려라
                   flags = 2)

# print()
# print('len(matchesMask)')
# print(len(matchesMask))
print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간

img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
plt.imshow(img3, 'gray')
plt.show()
## --------------------------------------------------- test ORB/BF end
'''
'''
## --------------------------------------------------- test ORB/FLANN start
start = time.time()  # 시작 시간 저장
# Initiate ORB detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

FLANN_INDEX_LSH = 6
index_params =  dict ( algorithm  =  FLANN_INDEX_LSH , 
                    table_number  =  6 ,  # 12 
                    key_size  =  12 ,      # 20 
                    multi_probe_level  =  2 )  # 2
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0) # 투영 유도 방법 
    matchesMask = mask.ravel().tolist()
    
    h,w = img1.shape
    d = 1
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
else:   
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None


draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # inliers 만 그려라
                   flags = 2)

# print()
# print('len(matchesMask)')
# print(len(matchesMask))
print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간

img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
plt.imshow(img3, 'gray')
plt.show()
## --------------------------------------------------- test ORB/FLANN end
'''
'''
## --------------------------------------------------- test SURF/BF end
start = time.time()  # 시작 시간 저장
# Initiate ORB detector
surf = cv2.xfeatures2d_SURF.SURF_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = surf.detectAndCompute(img1,None)
kp2, des2 = surf.detectAndCompute(img2,None)

# des1 = np.float32(des1)
# des2 = np.float32(des2)

bf = cv2.BFMatcher(cv2.NORM_HAMMING)

matches = bf.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0) # 투영 유도 방법 
    matchesMask = mask.ravel().tolist()
    
    h,w = img1.shape
    d = 1
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
else:   
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None


draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # inliers 만 그려라
                   flags = 2)

# print()
# print('len(matchesMask)')
# print(len(matchesMask))
print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간

img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
plt.imshow(img3, 'gray')
plt.show()

--> 막힘

'''


