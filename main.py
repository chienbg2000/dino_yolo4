import cv2
from dino import Dino
from video_handler import VideoHandler
from drawing_utils import draw_bounding_boxes, draw_score, draw_game_over

config_path = "yolov4-tiny.cfg"
weights_path = "yolov4-tiny.weights"

labels_path = "coco.names"
with open(labels_path, "r") as f:
    labels = f.read().strip().split("\n")

net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

youtube_url = "https://www.youtube.com/watch?v=Aib8nj_i16w"

video_handler = VideoHandler(youtube_url, net, output_layers, labels)

roi_x = 400  # Tọa độ x
roi_y = 250  # Tọa độ y
roi_width = 1200  # Chiều rộngl
roi_height = 300  # Chiều cao
dino = Dino(600,390)
gameOver = False
gameStart = False
isDrawBoundingBox = False

while not gameOver:
    frame = video_handler.read_frame()

    roi = video_handler.crop_roi(frame, roi_x, roi_y, roi_width, roi_height)

    max_y = dino.roi_y - roi_y + dino.roi_height + 20  #tạo filter chỉ lấy những xe cùng hàng với dino
    class_ids, confidences, boxes = video_handler.detect_objects(roi,max_y)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    draw_score(frame, dino.score // 10)

    if cv2.waitKey(1) == ord(" "):
        if gameStart == False:
            gameStart = True;
        if dino.isJumping == False:
            dino.isJumping = True
            dino.move_y = -1 * dino.jumpSpeed

    dino.update()

    if isDrawBoundingBox:
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 0, 255), 2)

    draw_bounding_boxes(frame, boxes, class_ids, confidences, labels, indexes, roi_x, roi_y)
    dino.draw(frame)
    video_handler.show_frame(frame)



    if(gameStart == True):
        for box in boxes:
            x, y, w, h = box
            box_x = x + roi_x
            box_y = y + roi_y
            box_w = w
            box_h = h
            dino_x = dino.roi_x
            dino_y = dino.roi_y2
            dino_w = dino.roi_width
            dino_h = dino.roi_height
            if (dino_x < box_x + box_w and
                    dino_x + dino_w > box_x and
                    dino_y < box_y + box_h and
                    dino_y + dino_h > box_y):
                gameOver = True
    if gameOver:
        draw_game_over(frame)
    video_handler.show_frame(frame)

input()



