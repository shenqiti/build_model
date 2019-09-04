'''
使用单层ANN的插补
By:shenqiti
2019/9/4
'''
import tensorflow as tf
from sklearn.externals import joblib
from impyute.dataset.corrupt import Corruptor
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

def SLP_train(x_train,y_train,learning_rate,epochs,batch_size):

    x = tf.placeholder(dtype=tf.float32,shape=[None,15])
    y = tf.placeholder(dtype=tf.float32,shape=[None,1])
    W1 = tf.Variable(tf.random_normal([15,1]), name='W1')
    b1 = tf.Variable(tf.random_normal([1]), name='b1')
    y_ = tf.nn.sigmoid(tf.add(tf.matmul(x, W1), b1))
    y_clipped = tf.clip_by_value(y_, 1e-10, 1.0)
    cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped) + (1 - y) * tf.log(1 - y_clipped), axis=1))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        # 变量初始化
        sess.run(init_op)
        total_batch = int(x_train.shape[0] / batch_size)
        for epoch in range(epochs):
            avg_cost = 0
            for i in range(total_batch):
                batch_x = x_train[np.arange(i*batch_size,min((i+1)*batch_size,x_train.shape[0]-1))]
                batch_y = y_train.reshape(-1,1)[np.arange(i*batch_size,min((i+1)*batch_size,x_train.shape[0]-1))]
                _, c = sess.run([optimizer, cross_entropy], feed_dict={x: batch_x, y: batch_y})
                avg_cost += c / total_batch
            print("Epoch:", (epoch + 1), "cost = ", "{:.3f}".format(avg_cost))
        w = W1.eval(session=sess)
        b = b1.eval(session=sess)
    return w,b

def SLP_impute(impute_data,label,w,b):
    for i,j in np.argwhere(np.isnan(impute_data)):
        label_y = 0.75 if label[i]==1 else 0.25

def MLP(x_train,y_train,x_impute,n_hidden,learning_rate,epochs,batch_size,model_path):
    n_input = x_train.shape[1]
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden])),
        'out': tf.Variable(tf.random_normal([n_hidden, 1]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden])),
        'out': tf.Variable(tf.random_normal([1]))
    }
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, 1])
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    pred = tf.nn.sigmoid(out_layer)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # Initialize variables
        sess.run(init)
        # Training cycle
        for epoch in range(epochs):
            avg_cost = 0.
            total_batch = int(x_train.shape[0] / batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x = x_train[np.arange(i*batch_size,min((i+1)*batch_size,x_train.shape[0]-1))]
                batch_y = y_train.reshape(-1, 1)[
                    np.arange(i * batch_size, min((i + 1) * batch_size, x_train.shape[0] - 1))]
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                              y: batch_y})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % 10 == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", \
                      "{:.9f}".format(avg_cost))
        print("First Optimization Finished!")
        # 保存模型参数到硬盘上
        # save_path = saver.save(sess, model_path)
        # print("Model saved in file: %s" % save_path)
    return sess.run(pred,{x:x_impute})

def MLP_impute(x,y):
    for i, j in np.argwhere(np.isnan(x)):
        temp = np.delete(x,[j],axis=1)
        temp = np.array(pd.DataFrame(np.hstack([temp,y.reshape(-1,1)]).dropna(axis=0)))
        y_train = temp[:, -1]
        x_train = np.delete(temp,[-1],axis=1)





if __name__ == '__main__':
    os.chdir("..\\datas")
    x_init = joblib.load('imputation_x.joblib')
    y_init = joblib.load('imputation_y.joblib')
    reference_x = joblib.load('reference_x.joblib')
    reference_y = joblib.load('reference_y.joblib')

    corruptor = Corruptor(x_init, 0.1)
    x_miss = getattr(corruptor, "mcar")()
    x_miss = np.vstack((x_init, reference_x))
    y_miss = pd.concat((y_init,reference_y),axis=0)

    data_df = pd.DataFrame(pd.concat((pd.DataFrame(x_miss),y_miss.reset_index(drop=True)),axis=1))
    print(data_df.head())
    x_train,x_test,y_train,y_test = train_test_split(np.array(data_df.dropna(axis=0,how='any').drop(columns='总分')),
                                                     np.array(data_df.dropna(axis=0,how='any')['总分']),test_size=0.2, random_state=7)

    w,b = SLP_train(x_train,y_train,learning_rate=0.001,epochs=100,batch_size=10)
