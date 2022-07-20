from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import shutil

import core

buffer_path = 'storage/buffer/'
process_path = 'storage/process/'
old_path = 'storage/old/'

BATCH_SIZE = 24
IMG_HEIGHT = 224
IMG_WIDTH = 224


def retrain():
    print('>>> Retraining...')
    transfer_buffer(buffer_path, process_path)

    train_gen = ImageDataGenerator(
        rescale=1/255,
        rotation_range=180,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train_data = train_gen.flow_from_directory(
        process_path,
        color_mode="rgb",
        class_mode="categorical",
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        subset='training'
    )

    val_data = train_gen.flow_from_directory(
        process_path,
        color_mode="rgb",
        class_mode="categorical",
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        subset='validation'
    )

    core.model.fit(
        train_data,
        steps_per_epoch=train_data.samples // BATCH_SIZE,
        validation_data=val_data,
        validation_steps=val_data.samples // BATCH_SIZE,
        epochs=50
    )

    # TODO: back the old model up before saving a new one
    core.model.save('./xception_model')

    print('>>> Retraining completed')

    transfer_buffer(process_path, old_path)


def transfer_buffer(src, dest):
    shutil.copytree(src, dest, copy_function=shutil.move, dirs_exist_ok=True)


# if __name__ == '__main__':
#     clean_up_buffer()
