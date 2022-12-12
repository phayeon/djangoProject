from PIL import Image
import cv2 as cv


def Mosaic(*params):
    cmd = params[0]
    target = params[1]
    if cmd == 'IMG_READ':
        return (lambda x: cv.imread(x))(target)
    elif cmd == 'IMG_READ_PLT':
        return (lambda x: cv.cvtColor(cv.imread(x), cv.COLOR_BGR2RGB))(target)
    elif cmd == 'GRAY_SCALE':
        return (lambda x: x[:, :, 0] * 0.114 + x[:, :, 1] * 0.587 + x[:, :, 2] * 0.229)(target)
    elif cmd == 'FROM_ARRAY':
        return (lambda x: Image.fromarray(x))(target)
