import tensorflow as tf


#training data
#train_df = df.sample(frac=0.75, random_state=4) 

#testing data 
# it drops the training data
# from the original dataframe
val_df = df.drop(train_df.index)

# now let's separate the targets and labels
#X_train = train_df.drop('quality',axis=1)
#X_val = val_df.drop('quality',axis=1)
#y_train = train_df['quality']
#y_val = val_df['quality']

# We'll need to pass the shape
# of features/inputs as an argument
# in our model, so let's define a variable 
# to save it.
input_shape = [X_train.shape[1]]

#the input shape returns the number of features passed as input to the first NN layer.

#building model
#keras = how you build models 

model = tf.keras.Sequential([
tf.keras.layers.Dense(units=1,input_shape=input_shape)])
 
# after you create your model it's
# always a good habit to print out it's summary
model.summary()

#compiling other info about the model, have to do after model is defined
#here we are setting the optimizer (adam) and loss function (categorical cross entropy)
model.compile(optimizer='adam',  
               
              loss='CategoricalCrossentropy')  
