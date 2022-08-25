import cv2

i = 0
flag = True
captures = []

while( flag ):
    capture = cv2.VideoCapture(i)
    ret, frame = capture.read()
    flag = ret
    if flag:
       i += 1
       captures.append( capture )

while(True):
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        break

    for i, capture in enumerate( captures ):
        ret, frame = capture.read()
        cv2.imshow( 'frame' + str(i), frame )


capture.release()
cv2.destroyAllWindows() 