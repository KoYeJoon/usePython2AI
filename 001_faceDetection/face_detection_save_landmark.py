import dlib
import cv2
from imutils import face_utils
import argparse
import os
import glob
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn import MTCNN

def parse_args():
    desc = "Data preprocessing for Deep3DRecon."
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--img_path', type=str, default='../data', help='original images folder')
    parser.add_argument('--save_path', type=str, default='../temp_data/', help='custom path to save proccessed images and labels')
    parser.add_argument('--model',type = str, default = 'dlib', help = 'set face detection model (dlib, mtcnn)')
    parser.add_argument('--opt', type=str, default='train', help='train/test mode')

    return parser.parse_args()

def midpoint(p1, p2):
    coords = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
    return [int(x) for x in coords]


def preprocessing_with_dlib(args):
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

# draw an image with detected objects
def draw_image_with_boxes(filename, result_list):
    # load the image
    data = pyplot.imread(filename)
    # plot the image
    pyplot.imshow(data)
    # get the context for drawing boxes
    ax = pyplot.gca()
    # plot each box
    for result in result_list:
        # get coordinates
        x, y, width, height = result['box']
        # create the shape
        rect = Rectangle((x, y), width, height, fill=False, color='red')
        # draw the box
        ax.add_patch(rect)
        # draw the dots
        for key, value in result['keypoints'].items():
            # create and draw dot
            dot = Circle(value, radius=2, color='red')
            ax.add_patch(dot)
    # show the plot
    pyplot.show()

def preprocessing_with_mtcnn(args):
    image_path = args.img_path
    save_path = args.save_path
    opt = args.opt

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    img_list = sorted(glob.glob(image_path + '/' + '*.png'))
    img_list += sorted(glob.glob(image_path + '/' + '*.jpg'))

    # 1. matplotlib 활용 image upload
    filename = 'custom/test/000001.jpg'
    # load image from file
    pixels = pyplot.imread(filename)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    faces = detector.detect_faces(pixels)
    # display faces on the original image
    draw_image_with_boxes(filename, faces)

    # 2. openCV 활용 image upload
    # img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    # faces = detector.detect_faces(img)
    # # display faces on the original image
    # draw_image_with_boxes(filename, faces)

if __name__ == '__main__':
    args = parse_args()
    if args.model=='dlib':
        preprocessing_with_dlib(args)
    else :
        preprocessing_with_mtcnn(args)
