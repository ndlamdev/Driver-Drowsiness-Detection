from model.DriverDrowsinessDetection import DriverDrowsinessDetection
from model.VideoDriverDrowsinessDetection import VideoDriverDrowsinessDetection


model = DriverDrowsinessDetection("data/cnncat2.keras")
game = VideoDriverDrowsinessDetection(model)
game.start()