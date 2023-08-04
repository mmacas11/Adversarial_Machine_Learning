import tensorflow as tf
from keras import backend as K
import numpy as np


# In[3]:


class Sampling(tf.keras.layers.Layer):
    ''' 
    (1) Sampling Layer is a subclass of tf.keras.layers.Layer
    (2) Reparameterization trick: use z_mean and z_log_var to sample z
    '''
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.math.exp(0.5*z_log_var) * epsilon


# In[4]:


class VAE_TD(tf.keras.Model):
    '''
    Our VAE is a subclass of tf.keras.Model
    '''
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE_TD, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")

    @property
    def metrics(self):
        '''
        Model calls automatically reset_states() on any object listed here, 
        at the beginning of each fit() epoch or at the begining of a call to evaluate().
        In this way, calling result() would return per-epoch average and not an average 
        since the start of training.
        '''
        return [self.total_loss_tracker]

    
    def train_step(self, x_true):
        ''' 
        (1) Override train_step(self, x_true) to customize what fit() does.
        (2) We use GradientTape() in order to record operations for automatic differentiation.
        '''
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x_true)
            x_pred = self.decoder(z)
            # reconstruction loss: mean squared error
            reconstruction_loss = tf.reduce_sum(tf.keras.losses.mean_squared_error(x_true, x_pred), axis=(1, 2))
            reconstruction_loss = K.mean(reconstruction_loss)
            # regularization term: KL divergence
            kl_loss = tf.reduce_sum(-0.5 * (1 + z_log_var - tf.math.square(z_mean) - tf.math.exp(z_log_var)), axis=1)
            total_loss = tf.reduce_mean(reconstruction_loss + kl_loss)       
        # self.trainable_variables and self.optimizer are inherited from tf.keras.Model
        # Get gradients of total loss with respect to the weights.
        grads = tape.gradient(total_loss, self.trainable_weights)
        # Update the weights of the model.
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        return {
            "loss": self.total_loss_tracker.result()
        }

# source: https://keras.io/guides/writing_your_own_callbacks/
class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):
    '''
    Stop training when the loss is at its min, i.e. the loss stops decreasing.

    Arguments:
        patience: Number of epochs to wait after min has been hit. 
                  After this number of no improvement, training stops.
    '''

    def __init__(self, patience=30):
        super(EarlyStoppingAtMinLoss, self).__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("loss")
        if np.less(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("\n Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))


# In[14]:


def refinedata (v_fhat,y_test,type_network,step_size):
    if type_network == 'rnn':
        r_fhat = []
        r_y_test = np.reshape(y_test,(y_test.shape[0]*y_test.shape[1]*y_test.shape[2]))
        for i in range(v_fhat.shape[0]):
            val = v_fhat[i]
            r_fhat.extend([val for x in range(step_size)])
    else:
        r_fhat = v_fhat
        r_y_test = y_test
    return np.array(r_fhat),np.array(r_y_test)

