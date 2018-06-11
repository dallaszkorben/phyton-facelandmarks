import face_recognition
import cv2
from PIL import Image, ImageDraw
import numpy as np

video_capture = cv2.VideoCapture(0)

process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:

        #face_locations = face_recognition.face_locations(small_frame)

        face_landmarks_list = face_recognition.face_landmarks(small_frame)

    #process_this_frame = not process_this_frame




    for face_landmarks in face_landmarks_list:

        face_landmarks['left_eyebrow'] = [(i[0]*4, i[1]*4) for i in face_landmarks['left_eyebrow']]
        pts = np.array(face_landmarks['left_eyebrow'], np.int32)
        cv2.polylines(frame, [pts], False, (0, 0, 255), 2)
        face_landmarks['right_eyebrow'] = [(i[0]*4, i[1]*4) for i in face_landmarks['right_eyebrow']]
        pts = np.array(face_landmarks['right_eyebrow'], np.int32)
        cv2.polylines(frame, [pts], False, (0, 0, 255), 2)


        face_landmarks['top_lip'] = [(i[0]*4, i[1]*4) for i in face_landmarks['top_lip']]
        pts = np.array(face_landmarks['top_lip'], np.int32)
        cv2.polylines(frame, [pts], True, (0, 0, 255), 2)
        face_landmarks['bottom_lip'] = [(i[0]*4, i[1]*4) for i in face_landmarks['bottom_lip']]
        pts = np.array(face_landmarks['bottom_lip'], np.int32)
        cv2.polylines(frame, [pts], True, (0, 0, 255), 2)

        face_landmarks['left_eye'] = [(i[0]*4, i[1]*4) for i in face_landmarks['left_eye']]
        pts = np.array(face_landmarks['left_eye'], np.int32)
        cv2.polylines(frame, [pts], True, (0, 0, 255), 2)
        face_landmarks['right_eye'] = [(i[0]*4, i[1]*4) for i in face_landmarks['right_eye']]
        pts = np.array(face_landmarks['right_eye'], np.int32)
        cv2.polylines(frame, [pts], True, (0, 0, 255), 2)

        face_landmarks['nose_bridge'] = [(i[0]*4, i[1]*4) for i in face_landmarks['nose_bridge']]
        pts = np.array(face_landmarks['nose_bridge'], np.int32)
        cv2.polylines(frame, [pts], True, (0, 0, 255), 2)

        face_landmarks['nose_tip'] = [(i[0]*4, i[1]*4) for i in face_landmarks['nose_tip']]
        pts = np.array(face_landmarks['nose_tip'], np.int32)
        cv2.polylines(frame, [pts], True, (0, 0, 255), 2)

        face_landmarks['chin'] = [(i[0]*4, i[1]*4) for i in face_landmarks['chin']]
        pts = np.array(face_landmarks['chin'], np.int32)
        cv2.polylines(frame, [pts], False, (0, 0, 255), 2)

        face_landmarks['nose_tip'] = [(i[0]*4, i[1]*4) for i in face_landmarks['nose_tip']]
        pts = np.array(face_landmarks['nose_tip'], np.int32)
        cv2.polylines(frame, [pts], True, (0, 0, 255), 2)




#    # Display the results
#    for (top, right, bottom, left) in face_locations:
#        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
#        top *= 4
#        right *= 4
#        bottom *= 4
#        left *= 4
#
#        # Draw a box around the face
#        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
#
#        ## Draw a label with a name below the face
#        #cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
#        #font = cv2.FONT_HERSHEY_DUPLEX
#        #cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


    # Display the resulting image
    cv2.imshow('Video', frame)

        #pil_image.show()

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()