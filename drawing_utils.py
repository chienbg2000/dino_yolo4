
import cv2

image_path = 't_rex.png'

image = cv2.imread(image_path)
image = cv2.resize(image, (100, 100))

def draw_bounding_boxes(frame, boxes, class_ids, confidences, labels, indexes, roi_x, roi_y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = labels[class_ids[i]]
            confidence = confidences[i]
            color = (0, 255, 0)  # Màu xanh lá cây
            cv2.rectangle(frame, (x + roi_x, y + roi_y), (x + w + roi_x, y + h + roi_y), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f} {x + roi_x} {y + roi_y}", (x + roi_x, y - 10 + roi_y), font, 0.6, color, 2)


def draw_score(frame, score):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "Score: " + str(score)
    text_size, _ = cv2.getTextSize(text, font, 1, 2)
    cv2.putText(frame, text, (frame.shape[1] - text_size[0] - 10, text_size[1] + 10), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

def draw_game_over(frame):
    text = "Game Over"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    color = (0, 0, 255)
    thickness = 5

    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

    x = int((frame.shape[1] - text_width) / 2)
    y = int((frame.shape[0] + text_height) / 2)

    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)


