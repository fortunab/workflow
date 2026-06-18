import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.scorecam import Scorecam
from tf_keras_vis.utils.scores import CategoricalScore


# ==========================================================
# CONFIG
# ==========================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 64
EPOCHS = 50
BEST_MODEL_PATH = "best_model.keras"

def get_dataset_path():
    import kagglehub

    path = kagglehub.dataset_download("yasserhessein/dataset-alzheimer")
    print("Dataset path:", path)

    dataset_path = os.path.join(path, "Alzheimer_s Dataset")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Could not find dataset folder: {dataset_path}\n"
            "Please check the downloaded Kaggle dataset structure."
        )

    return dataset_path

def create_data_generators(dataset_path):
    train_dir = os.path.join(dataset_path, "train")
    test_dir = os.path.join(dataset_path, "test")

    train_gen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.2)
    test_gen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_data = train_gen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
    )

    val_data = train_gen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
    )

    test_data = test_gen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    return train_data, val_data, test_data, test_dir


def build_model():
    inputs = layers.Input(shape=(224, 224, 3))

    x = layers.Conv2D(
        64, 3, activation="relu", kernel_regularizer=regularizers.l2(0.001)
    )(inputs)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(
        128, 3, activation="relu", kernel_regularizer=regularizers.l2(0.001)
    )(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(
        256, 3, activation="relu", kernel_regularizer=regularizers.l2(0.001)
    )(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(
        512, 3, activation="relu", kernel_regularizer=regularizers.l2(0.001)
    )(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(
        512, activation="relu", kernel_regularizer=regularizers.l2(0.001)
    )(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(4, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def load_img(path):
    img = tf.keras.utils.load_img(path, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img) / 255.0
    return np.expand_dims(arr, 0), arr


def replace_to_linear_keras3(model_instance):
    model_instance.layers[-1].activation = tf.keras.activations.linear


def norm(cam):
    cam = np.maximum(cam, 0)
    denom = cam.max()
    return cam / (denom + 1e-8) if denom > 0 else cam


def overlay(cam, img):
    cam = cv2.resize(cam, IMG_SIZE)
    cam = np.uint8(255 * cam)

    heat = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

    return cv2.addWeighted(
        np.uint8(img * 255), 0.6,
        heat, 0.4,
        0
    )

def main():
    dataset_path = get_dataset_path()

    train_data, val_data, test_data, test_dir = create_data_generators(dataset_path)

    class_names = list(train_data.class_indices.keys())
    print("Classes:", class_names)

    model = build_model()
    model.summary()

    callbacks = [
        ModelCheckpoint(
            BEST_MODEL_PATH,
            save_best_only=True,
            monitor="val_accuracy",
            mode="max",
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            mode="max",
            verbose=1,
        ),
    ]

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    model = tf.keras.models.load_model(BEST_MODEL_PATH)

    all_images = glob.glob(os.path.join(test_dir, "*", "*.jpg"))

    if not all_images:
        raise FileNotFoundError("No .jpg images found in the test directory.")

    img_path = random.choice(all_images)
    true_class = os.path.basename(os.path.dirname(img_path))

    print("Image:", img_path)
    print("True:", true_class)

    x, orig = load_img(img_path)

    pred = model.predict(x)
    pred_class = int(np.argmax(pred))

    print("Pred:", class_names[pred_class])
    print("Probabilities:", pred[0])

    # Safe warmup
    _ = model.predict(np.zeros((1, 224, 224, 3)))

    score = CategoricalScore(pred_class)

    gradcam = Gradcam(model, model_modifier=replace_to_linear_keras3, clone=False)
    gradcam_pp = GradcamPlusPlus(model, model_modifier=replace_to_linear_keras3, clone=False)
    scorecam = Scorecam(model, model_modifier=replace_to_linear_keras3, clone=False)

    g = norm(gradcam(score, x)[0])
    gp = norm(gradcam_pp(score, x)[0])
    sc = norm(scorecam(score, x)[0])

    g_img = overlay(g, orig)
    gp_img = overlay(gp, orig)
    sc_img = overlay(sc, orig)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(orig)
    plt.title(f"Original\n(True: {true_class})")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(g_img)
    plt.title(f"GradCAM\n(Pred: {class_names[pred_class]})")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(gp_img)
    plt.title("GradCAM++")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(sc_img)
    plt.title("ScoreCAM")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("cam_results.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("Saved CAM visualization to: cam_results.png")


if __name__ == "__main__":
    main()
