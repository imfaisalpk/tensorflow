import tensorflow as tf

#model parameters

W = tf.Variable([-1.0],tf.float32)
b = tf.Variable([1.0],tf.float32)

# inputs and outputs
x = tf.placeholder(tf.float32)

linear_model = W*x+b

y = tf.placeholder(tf.float32)


# Loss

square_delta = tf.square(linear_model-y)

loss = tf.reduce_sum(square_delta)

learning_rate = 0.01

optimizer = tf.train.GradientDescentOptimizer(learning_rate)

train = optimizer.minimize(loss)



init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)


# file_writer = tf.summary.FileWriter("graph",sess.graph)
epochs = 1000
for i in range(epochs):
    sess.run(train,{x:[1,2,3,4],y:[0,-1,-2,-3]})

# print (sess.run(loss,{x:[1,2,3,4],y:[0,-1,-2,-3]}))

print (sess.run([W,b]))



# tensorflow graph
# node1 = tf.constant(3.0,tf.float32)
# node2 = tf.constant(4.0)
#
# node1 = tf.placeholder(tf.float32)
# node2 = tf.placeholder(tf.float32)
#
# # print (node1,node2)
# # sess = tf.Session()
# # print sess.run([node1,node2])
# # sess.close()
#
# # with tf.Session() as sess:
# #     print sess.run([node1,node2])
#
# nodec = node1 * node2
#
# sess = tf.Session()
#
# # file_writer = tf.summary.FileWriter("graph",sess.graph)
#
#
# output = sess.run(nodec,{node1:[2],node2:[3]})
#
# print output
#
# sess.close()