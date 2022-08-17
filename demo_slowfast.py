# -*- coding: utf-8 -*-

from utils import *
from torch_utils import *
from darknet2pytorch import Darknet
import torch
# import argparse
import cv2

class YOLOv4(object):
    def __init__(self, cfgfile, weightfile, namesfile):
        self.m = Darknet(cfgfile)
        self.m.print_network()
        self.m.load_weights(weightfile)
        print('Loading weights from %s... Done!' % (weightfile))
        self.use_cuda = True
        if self.use_cuda:
            self.m.cuda()
        print('Starting YOLOv4...')
        self.class_names = load_class_names(namesfile)
        
    def do_detect_1(self, model, img, conf_thresh, nms_thresh, use_cuda=1):
        model.eval()
        with torch.no_grad():
            t0 = time.time()

            if type(img) == np.ndarray and len(img.shape) == 3:  # cv2 image
                img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
            elif type(img) == np.ndarray and len(img.shape) == 4:
                img = torch.from_numpy(img.transpose(0, 3, 1, 2)).float().div(255.0)
            else:
                print("unknow image type")
                exit(-1)

            if use_cuda:
                img = img.cuda()
            img = torch.autograd.Variable(img)

            t1 = time.time()

            output = model(img)

            t2 = time.time()

            print('-----------------------------------')
            print('           Preprocess : %f' % (t1 - t0))
            print('      Model Inference : %f' % (t2 - t1))
            print('-----------------------------------')

            return utils.post_processing(img, conf_thresh, nms_thresh, output)

    def plot_boxes_cv2_1(self, img, boxes, savename=None, class_names=None, color=None):
        img = np.copy(img)
        colors = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)

        def get_color(c, x, max_val):
            ratio = float(x) / max_val * 5
            i = int(math.floor(ratio))
            j = int(math.ceil(ratio))
            ratio = ratio - i
            r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
            return int(r * 255)

        width = img.shape[1]
        height = img.shape[0]
        for i in range(len(boxes)):
            box = boxes[i]
            x1 = int(box[0] * width)
            y1 = int(box[1] * height)
            x2 = int(box[2] * width)
            y2 = int(box[3] * height)
            bbox_thick = int(0.6 * (height + width) / 600)
            if color:
                rgb = color
            else:
                rgb = (255, 0, 0)
            if len(box) >= 7 and class_names:
                cls_conf = box[5]
                cls_id = box[6]
                print('%s: %f' % (class_names[cls_id], cls_conf))
                classes = len(class_names)
                offset = cls_id * 123457 % classes
                red = get_color(2, offset, classes)
                green = get_color(1, offset, classes)
                blue = get_color(0, offset, classes)
                if color is None:
                    rgb = (red, green, blue)
                msg = str(class_names[cls_id])+" "+str(round(cls_conf,3))
                t_size = cv2.getTextSize(msg, 0, 0.7, thickness=bbox_thick // 2)[0]
                c1, c2 = (x1,y1), (x2, y2)
                c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
                # cv2.rectangle(img, (x1,y1), (np.float32(c3[0]), np.float32(c3[1])), rgb, -1)
                # img = cv2.putText(img, msg, (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0,0,0), bbox_thick//2,lineType=cv2.LINE_AA)
                img = cv2.putText(img, class_names[cls_id], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, rgb, 2)
            
            img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 2)
        if savename:
            print('save result to %s' % savename)
            cv2.imwrite(savename, img)
        return img
    
    def detect_cv2_camera(self):
        cap = cv2.VideoCapture(0)
        # cap = cv2.VideoCapture("./test.mp4")
        cap.set(3, 1280)
        cap.set(4, 720)
        print("Starting the YOLO loop...")
        
        while True:
            ret, img = cap.read()
            sized = cv2.resize(img, (self.m.width, self.m.height))
            sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

            start = time.time()
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            boxes = self.do_detect_1(self.m, sized, 0.4, 0.6, self.use_cuda)
            # x1 = boxes[0][0][0] * width
            # y1 = boxes[0][0][1] * height
            # x2 = boxes[0][0][2] * width
            # y2 = boxes[0][0][3] * height
            # confidence = boxes[0][0][5]
            # object_class = boxes[0][0][6]
            finish = time.time()
            print('Predicted in %f seconds.' % (finish - start))
            print('boxes --> ', boxes)
            print('************************************************************************')

            result_img = self.plot_boxes_cv2_1(img, boxes[0], class_names=self.class_names)
            cv2.imshow('Yolo demo', result_img)
            key = cv2.waitKey(1) & 0xff

            if key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def detect(self, img):
        sized = cv2.resize(img, (self.m.width, self.m.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
        for i in range(2):
            boxes = self.do_detect_1(self.m, sized, 0.4, 0.6, self.use_cuda)
        
        self.plot_boxes_cv2_1(img, boxes[0], savename='predictions.jpg', class_names=self.class_names)

if __name__ == '__main__':
    # detect_cv2_camera()
    cfgfile = '/home/cir-4/Desktop/NTNU/SlowFast/tools/YOLOv4/cfg/yolo-obj-assemble_4class.cfg'
    # weightfile = '/home/cir-4/Desktop/NTNU/SlowFast/tools/YOLOv4/yolo-obj-assemble_best.weights'
    weightfile = '/home/cir-4/Desktop/NTNU/SlowFast/tools/YOLOv4/yolo_weights_2nd/yolo-obj-assemble_7000.weights'
    namesfile = '/home/cir-4/Desktop/NTNU/SlowFast/tools/YOLOv4/data/obj_assemble.names'

    yolo = YOLOv4(cfgfile=cfgfile, weightfile=weightfile, namesfile=namesfile)
    # yolo.detect_cv2_camera()
    input_img = cv2.imread('/home/cir-4/Desktop/NTNU/SlowFast/tools/YOLOv4/yolo_example.png')
    yolo_result = yolo.detect(input_img)