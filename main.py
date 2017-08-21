import os
import sys
import numpy as np
import tensorflow as tf
import cv2
import datagen
import matplotlib.pyplot as plt
import argparse
import IPython

def main(train):
    batch_size = 50
    if not train:
        batch_size = 5
    class_num = len(datagen.char_to_idx)
    char_num = 4
    fc_out = char_num*class_num
    width = datagen.width
    height = datagen.height
    input = tf.placeholder(shape=[batch_size,height,width,3],dtype=tf.float32)
    target = tf.placeholder(dtype=tf.int32,shape=[batch_size,4])
    filter1_weights = tf.Variable(tf.truncated_normal(shape=[5,5,3,32],stddev=0.01))
    filter1_bias = tf.Variable(tf.zeros(shape=[32]))
    filter2_weights = tf.Variable(tf.truncated_normal(shape=[3,3,32,64],stddev=0.01))
    filter2_bias = tf.Variable(tf.zeros(shape=[64]))

    conv1 = tf.nn.conv2d(input,filter1_weights,strides=[1,2,2,1],padding="SAME")
    conv1 = tf.nn.bias_add(conv1,filter1_bias)
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

    conv1 = tf.nn.conv2d(conv1,filter2_weights,strides=[1,2,2,1],padding="SAME")
    conv1 = tf.nn.bias_add(conv1,filter2_bias)
    conv1 = tf.nn.relu(conv1)
    # conv1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

    shape = conv1.get_shape().as_list()

    batch = shape[0]
    size = shape[1]*shape[2]*shape[3]

    flat = tf.reshape(conv1,[batch,size])

    fc1_weights = tf.Variable(tf.truncated_normal(shape=[size,fc_out*2],stddev=0.001))
    fc1_bias = tf.Variable(tf.zeros(dtype=tf.float32,shape=[fc_out*2]))

    fc1 = tf.matmul(flat,fc1_weights) + fc1_bias

    fc2_weights = tf.Variable(tf.truncated_normal(shape=[fc_out*2,fc_out],stddev=0.001))
    fc2_bias = tf.Variable(tf.zeros(dtype=tf.float32,shape=[fc_out]))

    fc2 = tf.matmul(fc1,fc2_weights) + fc2_bias

    out = tf.reshape(fc2,shape=[batch_size,char_num,class_num])
    max_out = tf.nn.softmax(out,dim=2)
    pred_out = tf.arg_max(max_out,dimension=2)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out,labels=target)

    loss = tf.reduce_mean(loss)

    trainer = tf.train.GradientDescentOptimizer(0.001)

    step = trainer.minimize(loss)

    epoch = 200
   
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        if(train):
            saver.restore(sess,'verify.model')
            train_data_g, target_data_g = datagen.generate_data(batch_size*1500)
            for i in range(1,1+epoch):
                for j in range(1500):
                    train_data = train_data_g[j*batch_size:j*batch_size+batch_size]
                    target_data = target_data_g[j*batch_size:j*batch_size+batch_size]
                    [l,s] = sess.run([loss,step],feed_dict={input:train_data,target:target_data})
                    if j%10 == 0:
                        print "loss is " + str(l)
                if i % 50 == 0 :
                    saver.save(sess,'verify.model')
            saver.save(sess,'verify.model')
        else:
            if(os.path.exists('verify.model.meta')):
                saver.restore(sess,'verify.model')
                d,l = datagen.generate_data(batch_size)
                [o] = sess.run([pred_out],feed_dict={input:d})
                plot_pred(d,o)
                IPython.embed()
            else:
                print "please train the model first!"
def plot_pred(images,pred):
    for i in range(len(pred)):
        string = [datagen.idx_to_char[idx] for idx in pred[i]]
        print ''.join(string)
        plt.imshow(images[i])
        plt.show()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train',default=False,action="store_true",help='use to train the net')
    args = parser.parse_args()
    print args.train
    main(args.train)
 
    # IPython.embed()

