import cv2


def viola_jhones():
    img_rgb = cv2.imread('Stock/d4.jpg')
    img_rgb = cv2.resize(img_rgb, (480, 640))
    face_casd = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Find faces
    faces = face_casd.detectMultiScale(img_rgb, scaleFactor=1.5, minNeighbors=1, minSize=(40, 40))
    for (x, y, w, h) in faces:
        # Draw rectangles
        cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Print result
    cv2.imshow('Detected', img_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()