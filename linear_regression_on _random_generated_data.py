#----------------------------------------imported necesssary library 

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd


#--------------------------------creating random data

x_dataset= np.linspace(0.0,10.0,1000000)
noise = np.random.randn(len(x_dataset))
y_true = (0.5 * x_dataset) + 5 +noise                                        #-----------------y=mx+b

#------------------------Y = MX+B


#--------------------------------- preparing the dataset using pandas library

x_df=pd.DataFrame(data= x_dataset, columns=["x_data"])
y_df = pd.DataFrame(data=y_true,columns=["y_axis"])
my_data = pd.concat([x_df,y_df],axis = 1)
my_data.head()


#-----------------------------to plot data
my_data.sample(n=250).plot(kind = 'scatter', x = 'x_data', y = 'y_axis')


batch_size= 8 
#------------------------creating variable and placeholder

m =tf.Variable(0.5)
b= tf.Variable(1.0)

xph =tf.placeholder(tf.float32,[batch_size])
yph = tf.placeholder(tf.float32,[batch_size])



#----------------------building model

y_model = m*xph + b

#----------cost function

error = tf.reduce_sum(tf.square(yph-y_model))

#gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
train =optimizer.minimize(error)

#------------------------initiating variables

init = tf.global_variables_initializer()

#-----------------------------running sessions
with tf.Session() as sess:
  sess.run(init)

  batch = 1000
  for i in range (batch):
    rand_ind = np.random.randint(len(x_dataset), size =batch_size)
    feed = {xph:x_dataset[rand_ind], yph:y_true[rand_ind]}
    sess.run(train,feed_dict = feed)
  model_m,model_b = sess.run([m,b])
  
  
 #-------------------------visualizing slope and biase
model_b
model_m

# --------------------------plotting slope

y_hat= (x_dataset*model_m) +model_b

my_data.sample(n=250).plot(kind = 'scatter', x = 'x_data', y = 'y_axis')
plt.plot(x_dataset,y_hat,'r')
