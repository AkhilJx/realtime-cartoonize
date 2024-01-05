import os
import cv2
import numpy as np
import tensorflow as tf
import network
import guided_filter
from tqdm import tqdm


def resize_crop(image):
    h, w, c = np.shape(image)
    if min(h, w) > 720:
        if h > w:
            h, w = int(720 * h / w), 720
        else:
            h, w = 720, int(720 * w / h)
    image = cv2.resize(image, (w, h),
                       interpolation=cv2.INTER_AREA)
    h, w = (h // 8) * 8, (w // 8) * 8
    image = image[:h, :w, :]
    return image


def cartoonize(frame):
    tf.reset_default_graph()  # Add this line to clear the TensorFlow graph

    input_photo = tf.placeholder(tf.float32, [1, None, None, 3])
    network_out = network.unet_generator(input_photo)
    final_out = guided_filter.guided_filter(input_photo, network_out, r=1, eps=5e-3)

    all_vars = tf.trainable_variables()
    gene_vars = [var for var in all_vars if 'generator' in var.name]
    saver = tf.train.Saver(var_list=gene_vars)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    sess.run(tf.global_variables_initializer())
    model_path = ".\\saved_models\\"  # Corrected the path separator
    saver.restore(sess, tf.train.latest_checkpoint(model_path))

    image = frame
    image = resize_crop(image)
    batch_image = image.astype(np.float32) / 127.5 - 1
    batch_image = np.expand_dims(batch_image, axis=0)
    output = sess.run(final_out, feed_dict={input_photo: batch_image})
    output = (np.squeeze(output) + 1) * 127.5
    output = np.clip(output, 0, 255).astype(np.uint8)

    sess.close()  # Close the TensorFlow session to release resources

    return output


# def cartoonize(frame):
#     input_photo = tf.placeholder(tf.float32, [1, None, None, 3])
#     network_out = network.unet_generator(input_photo)
#     final_out = guided_filter.guided_filter(input_photo, network_out, r=1, eps=5e-3)
#
#     all_vars = tf.trainable_variables()
#     gene_vars = [var for var in all_vars if 'generator' in var.name]
#     saver = tf.train.Saver(var_list=gene_vars)
#
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
#     sess = tf.Session(config=config)
#
#     sess.run(tf.global_variables_initializer())
#     model_path = "E:/personal files/programs/White-box-Cartoonization-master/White-box-Cartoonization-master/test_code\saved_models/"
#     saver.restore(sess, tf.train.latest_checkpoint(model_path))
#
#     image = frame
#     image = resize_crop(image)
#     batch_image = image.astype(np.float32) / 127.5 - 1
#     batch_image = np.expand_dims(batch_image, axis=0)
#     output = sess.run(final_out, feed_dict={input_photo: batch_image})
#     output = (np.squeeze(output) + 1) * 127.5
#     output = np.clip(output, 0, 255).astype(np.uint8)
#     return output

# Initialize the webcam
cap = cv2.VideoCapture(0)
ret = True

while ret:
    # Read each frame from the webcam
    ret, frame = cap.read()
    cv2.putText(frame, "Press 'q' to quit", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Input", frame)
    a = cartoonize(frame)
    cv2.putText(a, "Press 'q' to quit", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Output", a)
    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()

