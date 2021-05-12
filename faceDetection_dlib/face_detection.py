import dlib
import cv2
from imutils import face_utils
import argparse
from utils import *
import os
import glob

def parse_args():
    desc = "Data preprocessing for Deep3DRecon."
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--img_path', type=str, default='./custom/train/refined_train', help='original images folder')
    parser.add_argument('--save_path', type=str, default='./custom_processed_data/train/', help='custom path to save proccessed images and labels')
    parser.add_argument('--opt', type=str, default='train', help='train/test mode')

    return parser.parse_args()

def midpoint(p1, p2):
    coords = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
    return [int(x) for x in coords]

def preprocessing():
    args = parse_args()
    image_path = args.img_path
    save_path = args.save_path
    opt = args.opt

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    img_list = sorted(glob.glob(image_path + '/' + '*.png'))
    img_list += sorted(glob.glob(image_path + '/' + '*.jpg'))

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    file_n = 1
    for file in img_list:
        img = cv2.imread(file)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 1하니까 갑자기 여러명 detect 됨
        rects = detector(gray, 1)

        count =1

        if len(rects) > 0:
            for rect in rects:
                x = rect.left()
                y = rect.top()
                w = rect.right()
                h = rect.bottom()

                shape = predictor(img, rect)
                shape_np = face_utils.shape_to_np(shape).tolist()
                left_eye = midpoint(shape_np[36], shape_np[39])
                right_eye = midpoint(shape_np[42], shape_np[45])
                features = [left_eye, right_eye, shape_np[33], shape_np[48], shape_np[54]]

                print(file.split('/')[-1][:-4])

                if opt == 'train':
                    cv2.imwrite('%s%06d.jpg' % (save_path, file_n), img)

                    with open('%s%06d.txt' % (save_path, file_n), "a") as f:
                        for i in features:
                            print(str(i[0]) + ' ' + str(i[1]), file=f)
                else :
                    cv2.imwrite('%s%06d-%03d.jpg'%(save_path,file_n,count),img)

                    with open('%s%06d-%03d.txt'%(save_path,file_n,count), "a") as f:
                        for i in features:
                            print(str(i[0]) + ' ' + str(i[1]), file=f)
                count += 1
            file_n += 1

if __name__ == '__main__':
	preprocessing()
