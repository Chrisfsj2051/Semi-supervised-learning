import random

import scipy.io as sio
import matplotlib.pyplot as plt

train = sio.loadmat('data/svhn/train_32x32.mat')
test = sio.loadmat('data/svhn/test_32x32.mat')

x_train = train['X']
y_train = train['y']

num_rows, num_cols = 8, 14

figs, axis = plt.subplots(
    num_rows,
    num_cols,
    figsize=(num_cols * 6, num_rows * 6),
    gridspec_kw={"wspace":0.2, "hspace":0.2},
    squeeze=False)

for j in range(num_cols * num_rows):
    axi = axis[j // num_cols, j % num_cols]
    idx = random.randint(0, x_train.shape[-1] - 1)
    axi.imshow(x_train[:, :, :, idx])
    axi.axis('off')

plt.tight_layout()
# plt.show()
plt.savefig(f'exchange/svhn.png')
#
# #replace all 0 images with label 10 with 0s
# y_train[y_train==10] = 0
#
# #show the first 3 examples of the dataset
# for i in range (0,3):
#     print("Image index: " + str(i))
#     plt.imshow(x_train[:,:,:,i])
#     plt.show()
#
#     print("Digit Label: " + str(y_train[i]))
#     print()
#
# #check that the replacemente of all incorrect 10 labels have been replaced with 0s
# image_ind = 108 #this one is a 0 and the label is specified ad 10, all these cases must be replaced with 0
# print("Image index: " + str(image_ind))
# plt.imshow(x_train[:,:,:,image_ind])
# plt.show()
# print("Digit Label: " + str(y_train[image_ind]))
# print()
