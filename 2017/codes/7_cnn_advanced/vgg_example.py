# block 2 -- outputs 56x56x128
net = L.conv(net, name="conv2_1", kh=3, kw=3, n_out=128)
net = L.conv(net, name="conv2_2", kh=3, kw=3, n_out=128)
net = L.pool(net, name="pool2", kh=2, kw=2, dh=2, dw=2)

# # block 3 -- outputs 28x28x256
net = L.conv(net, name="conv3_1", kh=3, kw=3, n_out=256)
net = L.conv(net, name="conv3_2", kh=3, kw=3, n_out=256)
net = L.pool(net, name="pool3", kh=2, kw=2, dh=2, dw=2)

import tensorflow.contrib.slim as slim
net = slim.conv2d(inputs=input_val, num_outputs=128, kernel_size=[3,3], scope='conv1_1')


vgg = tf.contrib.slim.nets.vgg
with slim.arg_scope(vgg.vgg_arg_scope()):
  logits, end_points = vgg.vgg_16(inputs=images, num_classes=120, is_training=True)
