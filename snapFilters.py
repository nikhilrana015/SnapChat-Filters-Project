import cv2
import os
import imutils
import numpy as np 
import dlib as db 
import time
from imutils import face_utils
from keras.preprocessing.image import img_to_array


path_to_filter_sg = os.path.join(os.getcwd(), "Filters", "sunglasses.png")
path_to_filter_moustache = os.path.join(os.getcwd(), "Filters", "moustache.png")


face_detector = db.get_frontal_face_detector()
landmark_predict = db.shape_predictor('shape_predictor_68_face_landmarks.dat')



'''
The shape of the sunglasses image contains 4 channel, 3 channel
becomes same that is RGB and last channel is alpha channel.
Alpha channel represents the transparency level of the pixel achieved
through the cv2.IMREAD_UNCHANGED.
Here's how the transparency channel works: the lower the value, 
the more transparent, or see-through, the pixel will become. The lower bound (completely transparent) 
is zero here, so any pixels set to 0 will not be seen; these look like white background pixels in the image
, but they are actually totally transparent.

'''




def applying_sunglasses(landmarks_lst, frame_img):

	sunglasses = cv2.imread(path_to_filter_sg, cv2.IMREAD_UNCHANGED)

	x, y = landmarks_lst[17]

	w = abs(int(landmarks_lst[17,0] - landmarks_lst[26,0]))
	h = abs(int(landmarks_lst[27,1] - landmarks_lst[33,1]))

	# Resize the sunglasses on the basis of person facial landmarks.
	new_sunglasses = cv2.resize(sunglasses, (w,h), cv2.INTER_CUBIC)


	# getting the region of interest for the sunglasses to be placed.
	roi = img_to_array(frame_img[y:y+h, x:x+w])


	# finding the non-transparent pts in sunglasses
	# 0 means transparent while greater than 0 means move towards non-transparency

	idx = np.argwhere(new_sunglasses[:,:,3]>0)
	# print(f'Shape of Idx: {idx.shape}')

	'''
	What we are doing is replacing the pixels of roi with the non-transparent
	pixels of the sunglasses in all the 3 channels along all the x and y values.
	
	'''

	for i in range(3):
		roi[idx[:,0], idx[:,1], i] = new_sunglasses[idx[:,0], idx[:,1], i]


	frame_img[y:y+h, x:x+w] = roi


	return frame_img 


def applying_moustache(landmarks_lst, frame_img):

	moustache = cv2.imread(path_to_filter_moustache, cv2.IMREAD_UNCHANGED)

	y = landmarks_lst[33,1]
	x = landmarks_lst[41,0] 

	h = abs(int(landmarks_lst[33,1] - landmarks_lst[62,1]))
	w = abs(int(landmarks_lst[41,0] - landmarks_lst[46,0]))

	new_moustache = cv2.resize(moustache, (w,h), cv2.INTER_CUBIC)

	roi = img_to_array(frame_img[y:y+h, x:x+w])

	idx = np.argwhere(new_moustache[:,:,3]>0)
	# print(f'Shape of Idx: {idx.shape}')


	for i in range(3):
		roi[idx[:,0], idx[:,1], i] = new_moustache[idx[:,0], idx[:,1], i]


	frame_img[y:y+h, x:x+w] = roi

	return frame_img








coords_landmarks_lst = []
cap = cv2.VideoCapture(0)
time.sleep(3.0)
writer = None

while True:

	_, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	coords = face_detector(gray, 2)

	for points in coords:

		landmarks = landmark_predict(gray,points)
		coords_landmarks_lst = face_utils.shape_to_np(landmarks)

	if len(coords_landmarks_lst) < 68:
		continue


	# Extractin the top-left coordinates for sunglasses which is
	# edge of left-eyebrow.
	# print(f'Length of Landmarks: {len(coords_landmarks_lst)}')

	frame = applying_sunglasses(coords_landmarks_lst, frame)
	frame = applying_moustache(coords_landmarks_lst, frame)
	
	


	cv2.imshow("frame", frame)
	if cv2.waitKey(10) & 0xFF == ord('q'):
		break

	if writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter("output.avi", fourcc, 25, (frame.shape[1], frame.shape[0]), True)

	if writer is not None:
		writer.write(frame)



cap.release()
writer.release()
cv2.destroyAllWindows()







