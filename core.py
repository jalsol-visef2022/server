import tensorflow as tf
from PIL import Image
import numpy as np
from collections import Counter

KERNEL_SIZE = 224
RESCALING_FACTOR = 1 / 255
ARCHITECTURE = 'xception'
HEALTHY_THRESHOLD = 90 / 100

label = {
    0: 'HLT',
    1: 'CLM',
    2: 'CLR',
    3: 'PLS',
    4: 'CLS'
}

model = tf.keras.models.load_model(f'./{ARCHITECTURE}_model')

'''
    Resize both dimensions to multiples of the kernel size
    to allow splitting the image into tiles
'''
def resize_image(img):
    height, width = img.size
    height = height // KERNEL_SIZE * KERNEL_SIZE
    width = width // KERNEL_SIZE * KERNEL_SIZE
    return img.resize((height, width), Image.ANTIALIAS)

'''
    Splitting the image into tiles of the kernel size
'''
def split_image(img):
    height, width, channels = img.shape

    tiled_array = img.reshape(height // KERNEL_SIZE, KERNEL_SIZE,
                              width // KERNEL_SIZE, KERNEL_SIZE, channels)

    tiled_array = tiled_array.swapaxes(1, 2)
    return tiled_array

'''
    The most frequent element is the most likely

    NOTE: "Healthy" tile have to be handled carefully.
    No matter how high the count of "healthy" tiles are, if there's any spot that's
    not healthy, the whole leaf might not be healthy.
    The threshold is set to 90% (>= 90% Healthy is considered Healthy).
'''
def most_probable_result(model_prediction):
    prediction = np.argmax(model_prediction, axis=1)
    prediction_size = len(prediction)
    counter = Counter(prediction)

    confidence = 0
    for i in range(prediction_size):
        confidence += model_prediction[i][prediction[i]]
    confidence /= prediction_size
    
    if counter[0] >= prediction_size * HEALTHY_THRESHOLD:
        return label[0], str(round(confidence * 100, 2))
    else:
        del counter[0]

    return label[counter.most_common(1)[0][0]], str(round(confidence * 100, 2))

def get_prediction(file):
    img = Image.open(file)
    img = resize_image(img)
    img = np.asarray(img) * RESCALING_FACTOR
    img = split_image(img)
    img = img.reshape(-1, img.shape[-3], img.shape[-2], img.shape[-1])

    model_prediction = model.predict(img)

    return most_probable_result(model_prediction)

# if __name__ == '__main__':
#     print(get_prediction('image_from_api.jpg'))