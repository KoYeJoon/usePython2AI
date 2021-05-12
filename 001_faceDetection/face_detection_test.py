import dlib
import cv2
import argparse
import os
import glob
from mtcnn import MTCNN

def parse_args():
    desc = "Data preprocessing for Deep3DRecon."
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--img_path', type=str, default='../data', help='original images folder')
    parser.add_argument('--save_path', type=str, default='../temp_data/', help='custom path to save proccessed images and labels')
    parser.add_argument('--model',type = str, default = 'dlib', help = 'set face detection model (dlib, mtcnn)')


    return parser.parse_args()

def midpoint(p1, p2):
    coords = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
    return [int(x) for x in coords]

def draw_rectangle_dlib(file,rects,save_path,count):
    img = cv2.imread(file)
    for rect in rects :
        x = rect.left()
        y = rect.top()
        w = rect.right()
        h = rect.bottom()
        img = cv2.rectangle(img, (x,y),(w,h),(255,0,0),3)

    cv2.imwrite('%s%06d_dlib.jpg' % (save_path, count), img)


def preprocessing_with_dlib(args):
    image_path = args.img_path
    save_path = args.save_path

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    img_list = sorted(glob.glob(image_path + '/' + '*.png'))
    img_list += sorted(glob.glob(image_path + '/' + '*.jpg'))

    detector = dlib.get_frontal_face_detector()
    count = 1

    for file in img_list:
        img = cv2.imread(file)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 1하니까 갑자기 여러명 detect 됨
        rects = detector(gray, 1)

        if len(rects) > 0:
            draw_rectangle_dlib(file,rects,save_path,count)
            count += 1

def draw_rectangle_mtcnn(file,result_list,save_path,count):
    img = cv2.imread(file)
    for result in result_list :
        x,y,w,h = result['box']
        img = cv2.rectangle(img, (x,y),(x+w,y+h),(255,0,0),3)

    cv2.imwrite('%s%06d_mtcnn.jpg' % (save_path, count), img)


def preprocessing_with_mtcnn(args):
    image_path = args.img_path
    save_path = args.save_path

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    img_list = sorted(glob.glob(image_path + '/' + '*.png'))
    img_list += sorted(glob.glob(image_path + '/' + '*.jpg'))

    count = 1
    for file in img_list:
        img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
        detector = MTCNN()
        faces = detector.detect_faces(img)

        if len(faces) > 0:
            draw_rectangle_mtcnn(file,faces,save_path,count)
            count += 1




if __name__ == '__main__':
    args = parse_args()
    if args.model=='dlib':
        preprocessing_with_dlib(args)
    else :
        preprocessing_with_mtcnn(args)
