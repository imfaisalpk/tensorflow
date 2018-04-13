import tensorflow as tf

nodea = tf.constant(5.0,tf.float32)
nodeb= tf.constant(2.0,tf.float32)
nodec = tf.constant(3.0,tf.float32)

nodee = tf.add(nodeb,nodec)

noded = tf.multiply(nodea,nodeb)

nodef = noded - nodee

print ("nodef:",nodef)

sess = tf.Session()

print (sess.run(nodef))

sess.close()