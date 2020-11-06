# -*- coding: utf-8 -*-

# import the necessary packages
# from object_detection.utils import label_map_util
import tensorflow as tf
import numpy as np
import cv2
import find_finger as ff


args = {
    "model": "./model/export_model_008/frozen_inference_graph.pb",
    # "model":"/media/todd/38714CA0C89E958E/147/yl_tmp/readingbook/model/export_model_015/frozen_inference_graph.pb",
    "labels": "./record/classes.pbtxt",
    # "labels":"record/classes.pbtxt" ,
    "num_classes": 1,
    "min_confidence": 0.6,
    "class_model": "../model/class_model/p_class_model_1552620432_.h5"}

COLORS = np.random.uniform(0, 255, size=(args["num_classes"], 3))


def scale_bb(center, _bs):
    # 顔枠のサイズ
    boxsize = [int(b * 1.5) for b in _bs]
    # 顔枠の左上を原点とした中心までの座標
    # import pdb; pdb.set_trace()
    xy = [c - b // 2 for c, b in zip(center, boxsize)]
    x1, y1 = xy
    x2, y2 = x1 + boxsize[0], y1 + boxsize[1]

    return [x1, y1, x2, y2]

def clip_bb_img(bb, img, ms):
    x1, y1, x2, y2 = bb[0], bb[1], bb[2], bb[3]
    try:
        height, width, _ = img.shape
    except Exception as e:
        import pdb
        pdb.set_trace()
    # 枠の左上 or 画像の左上縁
    dx = max(0, -x1)
    dy = max(0, -y1)
    x1 = max(0, x1)
    y1 = max(0, y1)

    # 枠の右下 or 画像の右下縁
    edx = max(0, x2 - width)
    edy = max(0, y2 - height)
    x2 = min(width, x2)
    y2 = min(height, y2)

    # 顔枠で切り抜き
    imgT = img[y1:y2, x1:x2]
    if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
        # 画像をコピーし周りに境界を作成
        imgT = cv2.copyMakeBorder(
            imgT, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)
    # 顔枠で切り抜き
    msT = ms[y1:y2, x1:x2]
    if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
        # 画像をコピーし周りに境界を作成
        msT = cv2.copyMakeBorder(
            msT, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)

    return imgT, msT

if __name__ == '__main__':
    model = tf.Graph()

    with model.as_default():
        print("> ====== loading NAIL frozen graph into memory")
        graphDef = tf.GraphDef()

        with tf.gfile.GFile(args["model"], "rb") as f:
            serializedGraph = f.read()
            graphDef.ParseFromString(serializedGraph)
            tf.import_graph_def(graphDef, name="")
        # sess = tf.Session(graph=graphDef)
        print(">  ====== NAIL Inference graph loaded.")
        # return graphDef, sess


    with model.as_default():
        with tf.Session(graph=model) as sess:
            imageTensor = model.get_tensor_by_name("image_tensor:0")
            boxesTensor = model.get_tensor_by_name("detection_boxes:0")

            # for each bounding box we would like to know the score
            # (i.e., probability) and class label
            scoresTensor = model.get_tensor_by_name("detection_scores:0")
            classesTensor = model.get_tensor_by_name("detection_classes:0")
            numDetections = model.get_tensor_by_name("num_detections:0")
            drawboxes = []

            img_name = "4.jpg"
            image = cv2.imread(img_name)
            mask = cv2.imread("_" + img_name)
            (H, W) = image.shape[:2]
            # print("H,W:", (H, W))
            output = image.copy()
            output_mask = mask.copy()
            img_ff, bin_mask, res = ff.find_hand_old(image.copy())
            image = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
            image = np.expand_dims(image, axis=0)

            (boxes, scores, labels, N) = sess.run(
                [boxesTensor, scoresTensor, classesTensor, numDetections],
                feed_dict={imageTensor: image})
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            labels = np.squeeze(labels)
            boxnum = 0
            box_mid = (0, 0)
            # print("scores_shape:", scores.shape)
            idx = 0
            for (box, score, label) in zip(boxes, scores, labels):
                # print(int(label))
                # if int(label) != 1:
                #     continue
                if score < args["min_confidence"]:
                    continue
                # scale the bounding box from the range [0, 1] to [W, H]
                boxnum = boxnum + 1
                (startY, startX, endY, endX) = box
                startX = int(startX * W)
                startY = int(startY * H)
                endX = int(endX * W)
                endY = int(endY * H)
                X_mid = startX + int(abs(endX - startX) / 2)
                Y_mid = startY + int(abs(endY - startY) / 2)
                box_mid = (X_mid, Y_mid)
                # get scaling bb
                # import pdb; pdb.set_trace()
                bs = (endX - startX, endY - startY)
                new_bb = scale_bb(box_mid, bs)
                # crop bb img
                newimg, newmask = clip_bb_img(new_bb, output.copy(), output_mask)
                # draw the prediction on the output image
                label_name = 'nail'
                # idx = int(label["id"]) - 1
                # idx = 0
                label = "{}: {:.2f}".format(label_name, score)
                idx += 1
                #cv2.rectangle(output, (new_bb[0], new_bb[1]), (new_bb[2], new_bb[3]),
                #                COLORS[0], 2)
                #cv2.rectangle(output_mask, (new_bb[0], new_bb[1]), (new_bb[2], new_bb[3]),
                #                COLORS[0], 2)
                #cv2.imwrite(str(idx) + "_finger.jpg", newimg)
                #cv2.imwrite(str(idx) + "_mask.jpg", newmask)
    #cv2.imwrite("labeld.jpg", output)
    #cv2.imwrite("labeld_mask.jpg", output_mask)
