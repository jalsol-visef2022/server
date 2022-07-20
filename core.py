import tensorflow as tf
from PIL import Image
import numpy as np
from collections import Counter
import time
import os

KERNEL_SIZE = 224
RESCALING_FACTOR = 1 / 255
ARCHITECTURE = 'xception'
HEALTHY_THRESHOLD = 92 / 100

short_symptom_labels = {
    0: 'HLT',
    1: 'CLM',
    2: 'CLR',
    3: 'PLS',
    4: 'CLS'
}

long_symptom_labels = {
    0: 'healthy',
    1: 'miner',
    2: 'rust',
    3: 'phoma',
    4: 'cercospora'
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
    then reshaping it
'''


def split_and_reshape(img):
    height, width, channels = img.shape

    img = img.reshape(height // KERNEL_SIZE, KERNEL_SIZE,
                      width // KERNEL_SIZE, KERNEL_SIZE, channels)

    img = img.swapaxes(1, 2)

    img = img.reshape(-1, img.shape[-3], img.shape[-2], img.shape[-1])
    return img


'''
    The most frequent element is the most likely

    NOTE: "Healthy" tile have to be handled carefully.
    No matter how high the count of "healthy" tiles are, if there's any spot that's
    not healthy, the whole leaf might not be healthy.
    The threshold is set to 92% (>=92% Healthy is considered actually healthy).
'''


def get_prediction(file):
    img = Image.open(file)
    img = resize_image(img)

    original_img = np.asarray(img)
    rescaled_img = original_img * RESCALING_FACTOR

    original_img = split_and_reshape(original_img)
    rescaled_img = split_and_reshape(rescaled_img)

    model_prediction = model.predict(rescaled_img)

    classification = np.argmax(model_prediction, axis=1)
    classification_size = len(classification)

    for i in range(len(long_symptom_labels)):
        path = f'storage/buffer/{long_symptom_labels[i]}'
        os.makedirs(path, exist_ok=True)

    for i in range(len(original_img)):
        path = f'storage/buffer/{long_symptom_labels[classification[i]]}'

        _ = Image.fromarray(original_img[i], 'RGB')
        _.save(f'{path}/{int(time.time_ns())}.jpg')

    counter = Counter(classification)

    confidence = 0
    for i in range(classification_size):
        confidence += model_prediction[i][classification[i]]
    confidence /= classification_size

    if counter[0] >= classification_size * HEALTHY_THRESHOLD:
        return short_symptom_labels[0], str(round(confidence * 100, 2))
    else:
        del counter[0]

    return short_symptom_labels[counter.most_common(1)[0][0]], str(round(confidence * 100, 2))

# if __name__ == '__main__':
#     print(get_prediction('image_from_api.jpg'))
