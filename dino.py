import cv2

def crop_image(image_path, num_crops):
    images = []
    image_path = image_path
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    frame_width = image.shape[1] // num_crops
    frame_height = image.shape[0]

    for i in range(num_crops):
        top_left = (i * frame_width, 0)
        bottom_right = ((i + 1) * frame_width, frame_height)

        frame = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        images.append(frame)
    return images

class Dino():


    def __init__(self,roi_x,roi_y):
        self.images = crop_image("dino.png",5)
        self.image = self.images[0]
        self.roi_width,self.roi_height , _ = self.image.shape
        self.index = 0
        self.score = 0
        self.isJumping = False
        self.max_jumping_height = 350
        self.jumpSpeed = 45
        self.roi_x = roi_x
        self.roi_y = roi_y
        self.roi_y2 = roi_y
        self.move_y = 0
        self.gravity = 3

    def draw(self, frame):
        height_small, width_small, channels_small = self.image.shape

        x1 = self.roi_x
        y1 = self.roi_y2
        x2 = x1 + width_small
        y2 = y1 + height_small

        overlay = self.image[:, :, :3]
        alpha = self.image[:, :, 3] / 255.0
        frame[y1:y2, x1:x2] = (1.0 - alpha)[:, :, None] * frame[y1:y2, x1:x2] + alpha[:, :, None] * overlay

    def update(self):
        self.score = self.score+1
        self.image = self.images[self.index]
        if self.isJumping and self.roi_y2 != self.roi_y:
            self.index = 0
        else:
            self.index = (self.index + 1)%2 + 2

        if self.isJumping:
            self.move_y = self.move_y + self.gravity

        self.roi_y2 += self.move_y
        if self.roi_y - self.roi_y2 >= self.max_jumping_height:
            self.roi_y2 = self.roi_y - self.max_jumping_height

        if self.roi_y2 >= self.roi_y:
            self.roi_y2 = self.roi_y
            self.move_y = 0
            self.isJumping = False








