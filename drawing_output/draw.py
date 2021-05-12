import cv2
import numpy as np

annot = np.loadtxt('/home/kite/yejoon/mot_neural_solver/output/experiments/04-23_14:50_evaluation/mot_files/MOT17-14-FRCNN.txt',delimiter=',',skiprows=0,dtype= float)
#annot = np.loadtxt('/home/kite/yejoon/FairMOT/demos/results.txt',delimiter=',',skiprows=0,dtype= float)
print(annot.shape)

np.random.seed(0)
bbox_color = [list(np.random.random(size=3) * 256) for _ in range(3000)]
frame_num = 1
count = 0

while (True):
    # frame 별로 capture 한다
    temp = annot[count][0]
    if frame_num != int(temp) :
        img = cv2.imread('/home/kite/yejoon/mot_neural_solver/data/MOT17Det/test/MOT17-14/img1/%06d.jpg' %frame_num)
        cv2.imwrite('/home/kite/yejoon/mot_neural_solver/drawing_output/custom2_labelremove/%06d.jpg' % frame_num, img)
        frame_num += 1
        continue

    img = cv2.imread('/home/kite/yejoon/mot_neural_solver/data/MOT17Det/test/MOT17-14/img1/%06d.jpg' % frame_num)

    while temp == annot[count][0]:
        img = cv2.rectangle(img, (int(annot[count][2]),int(annot[count][3])), (int(annot[count][2] + annot[count][4]), int(annot[count][3] + annot[count][5])), bbox_color[int(annot[count][1])], 3)
        cv2.putText(img, str(annot[count][1]),  (int(annot[count][2] + annot[count][4]), int(annot[count][3] + annot[count][5])), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
        count += 1
        if count > len(annot) - 1:
            break

    # cv2.imshow('img',img)
    # cv2.waitKey(0)
    cv2.imwrite('/home/kite/yejoon/mot_neural_solver/drawing_output/custom2_labelremove/%06d.jpg'%frame_num,img)
    frame_num += 1
    if count > len(annot)-1 :
        print("finish drawing!!")
        break


# 메모리를 해제시켜준다.
cv2.destroyAllWindows()

import os
os.system("ffmpeg -f image2 -r 28 -i /home/kite/yejoon/mot_neural_solver/drawing_output/custom2_labelremove/%06d.jpg -vcodec"
          " mpeg4 -y /home/kite/yejoon/mot_neural_solver/drawing_output/custom2_labelremove.mp4")

