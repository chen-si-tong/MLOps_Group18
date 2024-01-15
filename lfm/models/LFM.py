import tensorflow as tf


# creat SVDModel
class SVDModel(tf.keras.Model):
    '''
    This class represents a Singular Value Decomposition (SVD) model used in the Latent Factor Model (LFM) algorithm.
    
    Attributes:
    - user_num: Number of users in the dataset.
    - item_num: Number of items (movie categories) in the dataset.
    - dim: Dimensionality of the latent factors for users and items.
    - reg_strength: Regularization strength for preventing overfitting.
    
    Methods:
    - __init__: Initializes the SVD model with user and item embeddings, biases, and regularization.
    - call: Defines the forward pass of the model to compute predicted ratings.
    '''
    def __init__(self,cfg):
        super(SVDModel, self).__init__()
        self.reg_strength = cfg['reg_strength']
        initializer = tf.keras.initializers.TruncatedNormal(stddev=0.02)
        self.user_emb = tf.keras.layers.Embedding(cfg['user_num'], cfg['dim'], embeddings_initializer=initializer,trainable=True)
        self.item_emb = tf.keras.layers.Embedding(cfg['item_num'], cfg['dim'], embeddings_initializer=initializer,trainable=True)              
        self.global_bias = tf.Variable(initial_value=0.0, dtype=tf.float32, name="global_bias")
        self.bias_user = tf.keras.layers.Embedding(cfg['user_num'], 1, embeddings_initializer='zeros', name="bias_user")
        self.bias_item = tf.keras.layers.Embedding(cfg['item_num'], 1, embeddings_initializer='zeros', name="bias_item")
   
    
    def call(self, inputs):
        '''
        Defines the forward pass of the SVD model to compute predicted ratings.

        Args:
        - inputs: A tuple of user and item indices.

        Returns:
        - output_star: Predicted ratings clipped between 1.0 and 5.0.
        '''
        user, item = inputs
        user_emb = self.user_emb(user)
        item_emb = self.item_emb(item)
        dot = tf.reduce_sum(tf.multiply(user_emb, item_emb), axis=1)
        global_bias = self.global_bias
        bias_user = self.bias_user(user)
        bias_item = self.bias_item(item)
        output = dot + global_bias + tf.transpose(bias_user)+tf.transpose(bias_item)
        output_star = tf.clip_by_value(output, 1.0, 5.0) # Clip scores, setting them to 0 if they are less than 0 and to 5 if they are greater than 5.
        reg_loss = self.reg_strength * (tf.reduce_sum(tf.square(user_emb)) + tf.reduce_sum(tf.square(item_emb))) 
        self.add_loss(reg_loss) ## perparing to add the regularization into the loss function 
        return output_star
        
