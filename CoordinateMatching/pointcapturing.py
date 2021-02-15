import utils as ut


# import the necessary packages
import argparse
import cv2
# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
folder_path = './matchimg/cap_source'
file_name = 'point_5.jpg'
# cv2.imshow("image" , image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

refPt = []
cropping = False

def click_and_crop(event, x, y, flags, param):
	# grab references to the global variablesc
	global refPt, cropping
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
    
	# performed : 현재는 512*512 이미지로 매칭
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x-256, y+256)]
		cropping = True

	elif event == cv2.EVENT_LBUTTONUP:
    		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x+256, y-256))
		cropping = False
	# check to see if the left mouse button was released
		# draw a rectangle around the region of interest
		cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
		cv2.imshow("image", image)
		print()
		print('x, y : ')
		print(x)
		print(y)

        	# check to see if the left mouse button was released

image = cv2.imread(ut.get_filepath_list_pair(folder_path))
clone = image.copy()
cv2.namedWindow("image", cv2.WINDOW_NORMAL)
image = cv2.resize(image, (3840, 2160))
cv2.setMouseCallback("image", click_and_crop)
cv2.moveWindow("image", 00,00) # [00, 00], [00,-150]
cv2.imshow("image", image)
cv2.waitKey(0)
# keep looping until the 'q' key is pressedc

while True:
	# display the image and wait for a keypress
	cv2.imshow("image", image)
	key = cv2.waitKey(1) & 0xFF
	# if the 'r' key is pressed, reset the cropping region
	if key == ord("r"):
		image = clone.copy()
	# if the 'c' key is pressed, break from the loop
	elif key == ord("c"):
		break
# if there are two reference points, then crop the region of interest
# from teh image and display it
# print(refPt)
if len(refPt) == 2:
	roi = clone[refPt[1][1]:refPt[0][1], refPt[0][0]:refPt[1][0]]
	cv2.imshow("ROI", roi)
	cv2.waitKey(0)
cv2.imwrite(folder_path+'/cap_result/'+file_name,roi)
# close all open windows
cv2.destroyAllWindows()