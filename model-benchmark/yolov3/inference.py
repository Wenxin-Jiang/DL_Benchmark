import cv2
import numpy as np
from onnx import numpy_helper
import os, statistics
import onnx
import onnxruntime as rt
from PIL import Image
from matplotlib.pyplot import imshow, imsave
import matplotlib.pyplot as plt
from scipy import special
import colorsys
import random
import time
import argparse
from pycocotools.coco import COCO
from tqdm import tqdm

# ======= Command-line Arguments =======
parser = argparse.ArgumentParser(description='Run the model on given imgae dataset for object detection. acc and FPS are reported.')
parser.add_argument('-m', '--model', metavar='MODEL', required=True,
                    help='path of the model')
parser.add_argument('-i', '--input', metavar='INPUT', required=True,
                    help='path to the input image folder')
parser.add_argument('-a', '--annotation', metavar='ANNOTATION', required=True,
                    help='path to the annotation json file')
parser.add_argument('-s', '--stop', metavar='STOP', type=int, default=5000,
                    help='set a breaking point to stop early for testing')
parser.add_argument('-q', '--quantized', type=str,
                    help='set to decide whether we used a quantized model or not')
parser.add_argument('--save', metavar='SAVE',
                    help='path to save the visual output')
parser.add_argument('--gpu', action='store_true',
                    help='Use GPU if available')
args = parser.parse_args()

DTYPE = None
if args.quantized == 'fp32': DTYPE = np.float32
elif args.quantized == 'fp16': DTYPE = np.float16
else: print('dtype not available.')


# this function is from yolo3.utils.letterbox_image
def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def preprocess(img):
    model_image_size = (416, 416)
    boxed_image = letterbox_image(img, tuple(reversed(model_image_size)))
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.transpose(image_data, [2, 0, 1])
    image_data = np.expand_dims(image_data, 0)
    return image_data

def categoryID2name(coco_annotation, query_id):
    query_annotation = coco_annotation.loadCats([query_id])[0]
    query_name = query_annotation["name"]
    query_supercategory = query_annotation["supercategory"]
    # print(" - Category ID -> Category Name:")
    # print(f" - Category ID: {query_id}, Category Name: {query_name}, Supercategory: {query_supercategory}")
    return query_name

def main():

    # ======= Inference =======
    # Start from ORT 1.10, ORT requires explicitly setting the providers parameter if you want to use execution providers
    # other than the default CPU provider (as opposed to the previous behavior of providers getting set/registered by default
    # based on the build flags) when instantiating InferenceSession.
    # For example, if NVIDIA GPU is available and ORT Python package is built with CUDA, then call API as following:
    # rt.InferenceSession(path/to/model, providers=['CUDAExecutionProvider'])

    # Load gdourntruth annotations in to a dictionary
    # https://leimao.github.io/blog/Inspecting-COCO-Dataset-Using-COCO-API/
    coco_annotation_file_path = args.annotation
    coco_annotation = COCO(annotation_file=coco_annotation_file_path)

    # Category IDs and # All categories..
    cat_ids = coco_annotation.getCatIds()
    cats = coco_annotation.loadCats(cat_ids)
    cat_names = [cat["name"] for cat in cats]
    # print(f"Number of Unique Categories: {len(cat_ids)}")
    # print(f"Category IDs: {cat_ids}") # The IDs are not necessarily consecutive.
    # print(f"Categories Names: {cat_names}")

    # Get the ID of all the images containing the object of the category.
    img_ids = coco_annotation.getImgIds()[-args.stop:]
    print(f"Number of Images to be processed: {len(img_ids)}\n")


    # rt.get_device()
    # sess = rt.InferenceSession(args.model, providers=['CUDAExecutionProvider'])
    if args.gpu:
        sess = rt.InferenceSession(args.model, providers=['CUDAExecutionProvider'])
        print('- Device: GPU')
    else:
        sess = rt.InferenceSession(args.model, providers=['CPUExecutionProvider'])
        print('- Device: CPU')

    # model = onnx.load(args.model)
    # sess = backend.prepare(model, device='CUDA:0')

    input_size = 416

    acc_list, conf_list, FPS = [], [], []
    for img_id in tqdm(img_ids):

        # ======= Read the original image =======
        img_info = coco_annotation.loadImgs([img_id])[0]
        img_path = os.path.join(args.input, img_info['file_name'])
        image = Image.open(img_path)
        # input
        image_data = preprocess(image)
        image_size = np.array([image.size[1], image.size[0]], dtype=np.int32). reshape(1, 2)
        # print(" - Preprocessed image shape:",image_data.shape) # shape of the preprocessed input
        # imsave("sample.jpg", np.asarray(original_image))

        # ======= Processing =======
        annIds = coco_annotation.getAnnIds(imgIds=[img_id],  iscrowd=None)
        anns = coco_annotation.loadAnns(annIds)
        start = time.time()
        outputs = sess.get_outputs()
        output_names = list(map(lambda output: output.name, outputs))
        input_name = sess.get_inputs()[0].name
        detections = sess.run(output_names, {input_name: image_data})
        end = time.time()
        # print(" - Output shape:", list(map(lambda detection: detection.shape, detections)))

        # ======= Start Post-processing =======
        out_boxes, out_scores, out_classes = [], [], []
        for idx_ in indices:
            x


        pred_bbox = postprocess_bbbox(detections, ANCHORS, STRIDES, XYSCALE)
        bboxes = postprocess_boxes(pred_bbox, original_image_size, input_size, 0.25)
        bboxes = nms(bboxes, 0.213, method='nms')
        image, conf, names = draw_bbox(original_image, bboxes)

        # ======= Scores & Output =======
        output_im = Image.fromarray(image)
        gth_im = cv2.imread(img_path)
        detect_count = 0
        for ann in anns: # gth
            query_id = ann['category_id']
            query_name = categoryID2name(coco_annotation, query_id)
            [x,y,w,h] = [int(i) for i in ann['bbox']]
            boo = False
            for i, bbox in enumerate(bboxes): # output
                bbox_gth = np.array([x, y, x+w, y+h,])
                if bboxes_iou(bbox_gth, bbox[:4]) > 0.5 and query_name == names[i]:
                    boo = True
            detect_count += 1 if boo == True else 0

            # Draw the boxes
            cv2.rectangle(gth_im, (x,y), (x+w, y+h), (0, 255, 0), 2)
            gth_im = cv2.putText(gth_im, query_name, (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)


        if args.save:
            final = np.concatenate((np.asarray(output_im), np.asarray(gth_im)), axis = 0)
            cv2.imwrite(f'{args.save}/{img_id}.jpg', final)

        conf = round(conf, 6)
        fps = round(1/(end-start), 6)
        acc = round(detect_count / (len(anns) + 1e-8), 6)
        conf_list.append(conf)
        FPS.append(fps)
        acc_list.append(acc)

        # print(f' - Processing {img_path}')
        # print(' - mConf: {:.4f}; FPS: {:.4f}; acc: {:.4f}'.format(conf, fps, acc))

    print(f' - model: {args.model.split("/")[-1]}, mConfidence: {statistics.mean(conf_list)}, mFPS: {statistics.mean(FPS)}, mAcc: {statistics.mean(acc_list)}\n')

if __name__ == '__main__':
    main()
