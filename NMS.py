'''
非极大值抑制的流程如下：

1. 根据置信度得分进行排序
2. 选择置信度最高的边界框添加到最终输出列表中，将其从边界框列表中删除
3. 计算所有边界框的面积
4. 计算置信度最高的边界框与其它候选框的IoU。
5. 删除IoU大于阈值的边界框
重复上述过程，直至边界框列表为空。
'''

import cv2
import numpy as np

'''
@param list: Object candidate bounding boxes
@param list: Confidence score of bounding boxes
@param float IoU threshold

@return: Rest boxes after nms operation
'''
def nms(bouding_boxes, confidence_score, threshold):
    if len(bounding_boxes) == 0:
        return [], []
    
    boxes = np.array(bounding_boxes)
    min_xs = boxes[:, 0]
    min_ys = boxes[:, 1]
    max_xs = boxes[:, 2]
    max_ys = boxes[:, 3]

    scores = np.array(confidence_score)

    picked_boxes = []
    picked_scores = []

    # Compute all the areas
    areas = (max_xs - min_xs + 1) * (max_ys - min_ys + 1)

    # Sort by confidence score
    order = np.argsort(scores)

    # Iterate each box
    while order.size > 0:
        # the index of largest confidence score
        index = order[-1]

        picked_boxes.append(bounding_boxes[index])
        picked_scores.append(confidence_score[index])

        #compute coordinates of intersection area
        x1s = np.maximum(min_xs[index], min_xs[order[:-1]])
        x2s = np.minimum(max_xs[index], max_xs[order[:-1]])
        y1s = np.maximum(min_ys[index], min_ys[order[:-1]])
        y2s = np.minimum(max_ys[index], max_ys[order[:-1]])
        
        # compute the w and h of interesction areas
        w = np.maximum(0.0, x2s - x1s + 1)
        h = np.maximum(0.0, y2s - y1s + 1)
        intersection_areas = w * h

        ratio = intersection_areas / (areas[index] + areas[order[:-1]] - intersection_areas)

        picked = np.where(ratio < threshold)
        order = order[picked]

    return picked_boxes, picked_scores

image_name = 'nms.jpg'

bounding_boxes = [(187, 82, 337, 317), (150, 67, 305, 282), (246, 121, 368, 304)]
confidence_score = [0.9, 0.75, 0.8]

image = cv2.imread(image_name)
original = image.copy()

#box line parameters
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
thickness = 2

#IoU threshold
threshold = 0.4

# Draw bouding boxes and confidence score
for (min_x, min_y, max_x, max_y), conf in zip(bounding_boxes, confidence_score):
    (w, h), baseline = cv2.getTextSize(str(conf), font, font_scale, thickness)
    cv2.rectangle(original, (min_x, min_y-(2*baseline+5)), (min_x + w, min_y), (0, 255, 255), -1)
    cv2.rectangle(original, (min_x, min_y), (max_x, max_y), (0, 255, 255), 2)
    cv2.putText(original, str(conf), (min_x, min_y), font, font_scale, (0, 0, 0), thickness)

picked_boxes, picked_score = nms(bounding_boxes, confidence_score, threshold)

# Draw bounding boxes and confidence score after non-maximum supression
for (start_x, start_y, end_x, end_y), confidence in zip(picked_boxes, picked_score):
    (w, h), baseline = cv2.getTextSize(str(confidence), font, font_scale, thickness)
    cv2.rectangle(image, (start_x, start_y - (2 * baseline + 5)), (start_x + w, start_y), (0, 255, 255), -1)
    cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 255, 255), 2)
    cv2.putText(image, str(confidence), (start_x, start_y), font, font_scale, (0, 0, 0), thickness)

cv2.imshow('Original', original)
cv2.imshow('NMS', image)
cv2.waitKey(0)