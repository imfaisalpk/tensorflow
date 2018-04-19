import matplotlib as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import  shuffle
from sklearn.model_selection import train_test_split


# reading dataset

def read_dataset():
    df = pd.read_csv("Sonar.csv")
    X = df[df.columns[0:60]].values
    print X

    y = df[df.columns[60]]
    print "y:"
    print y

    encoder = LabelEncoder()
    encoder.fit(y)

    y = encoder.transform(y)
    Y = one_hot_encode(y)
    return X,Y

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros(n_labels,n_unique_labels)
    one_hot_encode[np.arrange(n_labels),labels] = 1
    return one_hot_encode

# reading dataset
X,Y = read_dataset()

# shuffling dataset
X,Y = shuffle(X,Y,random_state=1)


# spliting the dataset into training and testing
train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size=0.20, random_state=415)

# inspecting shape of dataset

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)


# defining parameters and variables to work with tensors

learning_rate = 0.3
epoch, learning_epochs = 0, 1000
cost_history = np.empty(shape=[1], dtype=float)
n_dim = X.shape[1]
print("n_dim",n_dim)
n_classes = 2
model_path = "model"


# number of hidden layers
n_hidden_1=60
n_hidden_2=60
n_hidden_3=60
n_hidden_4=60

x = tf.placeholder(tf.folder,[None, n_dim])
W = tf.placeholder(tf.zeros[n_dim,n_classes])
b = tf.Variable(tf.zeros[n_classes])
y_ = tf.placeholder(tf.float32,[None, n_classes])


# defining model

def multilayer_percepton(x,weights,biases):

    # hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x,weights['h1']),biases['b1'])
    layer_1 = tf.nn.relu(layer_1)


    # hidden layer with sogmoid activation
    layer_2 = tf.add(tf.matmul(layer_1,weights['h2']),biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)

    # hiddden layer with sigmoid activation
    layer_3 = tf.add(tf.matmul(layer_2,weights['h3']),biases['b3'])
    layer_3 = tf.nn.sigmoid(layer_3)

    # hidden layer with RELU activation
    layer_4 = tf.add(tf.matmul(layer_3,weights['h4']),biases['b4'])
    layer_4 = tf.nn.relu(layer_4)

    out_layer = tf.matmul(layer_4,weights['out']) + biases['out']
    return out_layer


weights = {
    'h1':tf.Variable(tf.truncated_normal([n_dim,n_hidden_1])),
    'h2':tf.Variable(tf.truncated_normal([n_hidden_1,n_hidden_2])),
    'h3':tf.Variable(tf.truncated_normal([n_hidden_2,n_hidden_3])),
    'h4':tf.Variable(tf.truncated_normal([n_hidden_3,n_hidden_4])),
    'out':tf.Variable(tf.truncated_normal(n_hidden_4,n_classes))
}


biases={
    'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
    'b2':tf.Variable(tf.truncated_normal([n_hidden_2])),
    'b3':tf.Variable(tf.truncated_normal([n_hidden_3])),
    'b4':tf.Variable(tf.truncated_normal([n_hidden_4])),
    'out': tf.Variable(tf.truncated_normal([n_classes]))
}

init = tf.global_variables_initializer()

saver = tf.train.Saver()

# multilayer perceptron -- model
y = multilayer_percepton(x,weights,biases)

# define the cost and optimzation
cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=y_))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)


sess = tf.Session()
sess.run(init)


# calculate the cost and accuracy foreach epoch(iteration)

mse_history = []
accuracy_history= []

for epoch in range(learning_epochs):
    sess.run(training_step,feed_dict={x:train_x,y_:train_y})
    cost = sess.run(cost_function,feed_dict={x:train_x,y_:train_y})
    cost_history = np.append(cost_function,cost)
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    pred_y = sess.run(y,feed_dict={x:test_x})
    mse = tf.reduce_mean(tf.square(pred_y-test_y))
    mse_ = sess.run(mse)

    mse_history.append(mse_)
    accuracy = (sess.run(accuracy,feed_dict={x:train_x,y_:train_y}))
    accuracy_history.append(accuracy)

    print ('epoch: ',epoch,'--','cost: ',' - MSE: ',mse_," - Train Accuracy: ",accuracy)

save_path = saver.save(sess,model_path)
print ("Model saved in the file: %s" % save_path)

# plot mse and accuracy

plt.plot(mse_history,'r')
plt.show()
plt.plot(accuracy_history)
plt.show()

# print the final accuracy

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print("Test accuracy: ",(sess.run(accuracy,feed_dict={x:test_x,y_:test_y})))

# print the final mean square

pred_y = sess.run(y,feed_dict={x,test_x})
mse = tf.reduce_mean(tf.square(pred_y-test_y))
print ("MSE: %.4f "%sess.run(mse))





























































