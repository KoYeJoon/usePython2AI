import cv2
vidcap = cv2.VideoCapture('/home/kite/yejoon/mot_neural_solver/video2frame/dance2.mp4')
success,image = vidcap.read()
count = 1
while success:
  cv2.imwrite("/home/kite/yejoon/mot_neural_solver/video2frame/img2/%06d.jpg" % count, image)     # save frame as JPEG file
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1

print("finish! convert video to frame")

import cv2
import os


print("resize about image")
list = os.listdir('/home/kite/yejoon/mot_neural_solver/video2frame/img2')
file_count = len(list)
print(file_count)
count = 1

while (True):
    # img = cv2.imread('/home/kite/yejoon/FairMOT/demos/frame/%05d.jpg' % int(annot[count][0]))
    img = cv2.imread('/home/kite/yejoon/mot_neural_solver/video2frame/img2/%06d.jpg' % count)
    new_img = cv2.resize(img,dsize=(1920,1080), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite('/home/kite/yejoon/mot_neural_solver/video2frame/img2/%06d.jpg'%count,new_img)

    count += 1
    if count > file_count:
        print("finish!!")
        break

# 메모리를 해제시켜준다.
cv2.destroyAllWindows()