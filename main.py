import tensorflow as tf

# tensorflow graph
# node1 = tf.constant(3.0,tf.float32)
# node2 = tf.constant(4.0)

node1 = tf.placeholder(tf.float32)
node2 = tf.placeholder(tf.float32)

# print (node1,node2)
# sess = tf.Session()
# print sess.run([node1,node2])
# sess.close()

# with tf.Session() as sess:
#     print sess.run([node1,node2])

nodec = node1 * node2

sess = tf.Session()

# file_writer = tf.summary.FileWriter("graph",sess.graph)


output = sess.run(nodec,{node1:[2],node2:[3]})

print output

sess.close()