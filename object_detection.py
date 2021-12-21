# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------
import tensorflow.compat.v1 as tf
import inception_resnet_v2 as incep_v2
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # Or from tqdm import tqdm if not jupyter notebook
import selectivesearch
import matplotlib.patches as mpatches

tf.compat.v1.disable_eager_execution()
# ------------------------------------------------------------------------------
# Configurations
# ------------------------------------------------------------------------------
image_file_name = 'dog.jpg'  # 'Cat3.jpg' 'Dogs.png' 'NYC.jpg' 'Ring_road2.jpg'
n_rows = 299
n_cols = 299
classes_file_name = 'imagenet1000.txt'

# ------------------------------------------------------------------------------
# Declarations
# ------------------------------------------------------------------------------
def define_model(model, is_training):
    model.Image = tf.placeholder(tf.float32, shape=[None, n_rows, n_cols, 3])
    with incep_v2.slim.arg_scope(incep_v2.inception_resnet_v2_arg_scope()):
        model.logits, model.end_points = incep_v2.inception_resnet_v2(model.Image, is_training=False)

sess =tf.Session()

class Model_Class:
    def __init__(self, is_training):
        define_model(self, is_training=is_training)


# ------------------------------------------------------------------------------
# Create Model
# ------------------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
with tf.device('/cpu:0'):
    model = Model_Class(False)

# ------------------------------------------------------------------------------
# Read Image
# ------------------------------------------------------------------------------
img = cv2.imread(image_file_name, cv2.IMREAD_COLOR)
img = img[:, :, [2, 1, 0]]  # Make it RGB
img = cv2.resize(img, (n_rows, n_cols))
img_rgb = img.copy()
img = 2 * (img / 255.0) - 1.0
#print(img.min())

# ------------------------------------------------------------------------------
# Load the weights and Run the classifier
# ------------------------------------------------------------------------------
model.saver = tf.train.Saver()
model.saver.restore(sess, os.getcwd() + "\inception_resnet_v2_2016_08_30.ckpt")

classification = sess.run(model.end_points['Predictions'], feed_dict={model.Image: [img]})
classification_ = sess.run(model.logits, feed_dict={model.Image: [img]})

# ------------------------------------------------------------------------------
# Print Classification Result
# ------------------------------------------------------------------------------

#classes_ids = (-classification).argsort()  # Classes sorted from highest probability to the lowest

calss_id = np.argmax(classification) - 1  # -1 because that the network has the first class as "I don't know" and it's not included in the imagenet dict
classes_dict = eval(open(classes_file_name).read())
print('Whole image class: ' + classes_dict[calss_id])

# ------------------------------------------------------------------------------
# Run the object detector RNN
# ------------------------------------------------------------------------------
'''detector_window_min_size = 100
detector_window_max_size = 222
detector_window_size_step = 20
detector_window_slide_step = 100
location_size=[]
for s_h in tqdm(range(detector_window_min_size, detector_window_max_size, detector_window_size_step)):
    for s_w in tqdm(range(detector_window_min_size, detector_window_max_size, detector_window_size_step)):
        for h in range(0, len(img[0]), detector_window_slide_step):
            for w in range(0, len(img[1]), detector_window_slide_step):

                #print([w,h,s_w,s_h])
                #[w,h,s_w,s_h] = [100,100,150,200]

                height_to = np.min([h+s_h, n_rows])
                width_to = np.min([w+s_w, n_cols])
                cropped_img = cv2.resize(img[h:height_to,:][:,w:width_to], (n_rows, n_cols))

                classification = sess.run(model.end_points['Predictions'], feed_dict= {model.Image:[cropped_img]})

                confidences.append(np.max(classification))
                location_size.append([h,w,height_to,width_to])
                winner_classes_idx.append(np.argmax(classification)+1) # +1 because that the network has the first class as "I don't know" and it's not included in the imagenet dict'''

# ------------------------------------------------------------------------------
# Run the object detector Fast RNN = Selective search + Inception(regions)
# ------------------------------------------------------------------------------
confidences = []
winner_classes_idx = []
img_lbl, regions = selectivesearch.selective_search(img_rgb, scale=500, sigma=0.4, min_size=10)
candidates = set()
for r in regions:
    if r['rect'] in candidates:  # excluding same rectangle (with different segments)
        continue

    if r['size'] < 30 * 30:  # excluding regions smaller than X pixels
        continue

    ''''x, y, w, h = r['rect'] # excluding regions with extreme aspect ratios
    if h==0 or w==0 or w / h > 3 or h / w > 3:
        continue'''
    candidates.add(r['rect'])
candidates = list(candidates)

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
ax.imshow(img_rgb)
for x, y, w, h in candidates:
    rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=1)
    ax.add_patch(rect)
plt.title('Selective Search Result has ' + str(len(candidates)) + ' boxes, Image Classification is ' + classes_dict[calss_id])
plt.show()

for x, y, w, h in tqdm(candidates):  # Candidates: bottom, left, width, height
    cropped_img = cv2.resize(img[y:y + h, x:x + w], (n_rows, n_cols))
    classification = sess.run(model.end_points['Predictions'], feed_dict={model.Image: [cropped_img]})
    confidences.append(np.max(classification))
    winner_classes_idx.append(np.argmax(
        classification) - 1)  # -1 because that the network has the first class as "I don't know" and it's not included in the imagenet dict

# ------------------------------------------------------------------------------
# Visualize the object detector results
# ------------------------------------------------------------------------------
winnder_box_id = np.argmax(confidences)
winning_class = classes_dict[winner_classes_idx[winnder_box_id]]
print('Winner object class: ' + winning_class)

fig, ax = plt.subplots(nrows=1, ncols=1)
img_scaled = (img + 1) / 2.
img_scaled = img_scaled.copy()  # Bug in Python OpenCV wrapper
plt.imshow(img_scaled)

x, y, w, h = candidates[winnder_box_id]
rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor='green', linewidth=1)
ax.add_patch(rect)

plt.title('Best Object is ' + winning_class)
plt.show()