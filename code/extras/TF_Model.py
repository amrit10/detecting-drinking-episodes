import numpy as np
import tensorflow as tf

class Drunk(tf.keras.Model):

    def __init__(self):
        super(Drunk, self).__init__()

        self.model = tf.keras.Sequential([
            # tf.keras.layers.Flatten()
            tf.keras.layers.Dense(units=32, activation = 'relu'),
            tf.keras.layers.Dense(units=32, activation = 'relu'),
            tf.keras.layers.Dense(units=16, activation = 'relu'),
            tf.keras.layers.Dense(units=1, activation = 'sigmoid'),
        ])

    # @tf.function
    def call(self, X0):
        return self.model(X0)
    
#     def get_config(self):
#         return {"decoder":self.decoder}
    
#     @classmethod
#     def from_config(cls,config):
#         return cls(**config)

    def compile(self, optimizer, loss, metrics):
        '''
        Create a facade to mimic normal keras fit routine
        '''
        self.optimizer = optimizer
        self.loss_function = loss 
        self.accuracy_function = metrics[0]

    def train(self,X0 , Y0, batch_size=30):
        """
        Runs through one epoch - all training examples.

        :param model: the initialized model to use for forward and backward pass
        :param train_captions: train data captions (all data for training) 
        :param train_images: train image features (all data for training) 
        :param padding_index: the padding index, the id of *PAD* token. This integer is used when masking padding labels.
        :return: None
        """

        ## TODO: Implement similar to test below.

        ## NOTE: shuffle the training examples (perhaps using tf.random.shuffle on a
        ##       range of indices spanning # of training entries, then tf.gather) 
        ##       to make training smoother over multiple epochs.

        ## NOTE: make sure you are calculating gradients and optimizing as appropriate
        ##       (similar to batch_step from HW2)
        
        num_batches = int(len(X0) / batch_size)

        # Shuffling
        # indices = tf.range(start=0, limit=tf.shape(X0)[0], dtype=tf.int32)
        # shuffled_indices = tf.random.shuffle(indices)
        # train_captions = tf.gather(X0, shuffled_indices)
        # train_image_features = tf.gather(Y0, shuffled_indices)
        

        loss_list = []
        acc_list = []
        for index, end in enumerate(range(batch_size, len(X0)+1, batch_size)):
            start = end - batch_size
            inputs = X0[start:end, :]
            # decoder_input = train_captions[start:end, :-1]
            # decoder_labels = train_captions[start:end, 1:]
            # mask = decoder_labels != padding_index
            # num_predictions = tf.reduce_sum(tf.cast(mask, tf.float32))
            
            with tf.GradientTape() as tape:
                output = self.model(inputs)
                loss = self.loss_function(output, Y0)

            gradients = tape.gradient(loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(gradients,self.trainable_weights))
            accuracy = self.accuracy_function(output, Y0)

            loss_list.append(loss)
            acc_list.append(accuracy)

            
        # total_loss += loss
        # total_seen += num_predictions
        # total_correct += num_predictions * accuracy

        # avg_loss = float(total_loss / total_seen)
        # avg_acc = float(total_correct / total_seen)
        # avg_prp = np.exp(avg_loss)     
        
        # return avg_loss, avg_acc, avg_prp

        return np.mean(loss_list), np.mean(acc_list)

    def test(self, X1, Y1, batch_size=30):
        """
        DO NOT CHANGE; Use as inspiration

        Runs through one epoch - all testing examples.

        :param model: the initilized model to use for forward and backward pass
        :param test_captions: test caption data (all data for testing) of shape (num captions,20)
        :param test_image_features: test image feature data (all data for testing) of shape (num captions,1000)
        :param padding_index: the padding index, the id of *PAD* token. This integer is used to mask padding labels.
        :returns: perplexity of the test set, per symbol accuracy on test set
        """
        num_batches = int(len(test_captions) / batch_size)

        total_loss = total_seen = total_correct = 0
        for index, end in enumerate(range(batch_size, len(test_captions)+1, batch_size)):

            # NOTE: 
            # - The captions passed to the decoder should have the last token in the window removed:
            #	 [<START> student working on homework <STOP>] --> [<START> student working on homework]
            #
            # - When computing loss, the decoder labels should have the first word removed:
            #	 [<START> student working on homework <STOP>] --> [student working on homework <STOP>]

            ## Get the current batch of data, making sure to try to predict the next word
            start = end - batch_size
            batch_image_features = test_image_features[start:end, :]
            decoder_input = test_captions[start:end, :-1]
            decoder_labels = test_captions[start:end, 1:]

            ## Perform a no-training forward pass. Make sure to factor out irrelevant labels.
            probs = self(batch_image_features, decoder_input)
            mask = decoder_labels != padding_index
            num_predictions = tf.reduce_sum(tf.cast(mask, tf.float32))
            loss = self.loss_function(probs, decoder_labels, mask)
            accuracy = self.accuracy_function(probs, decoder_labels, mask)

            ## Compute and report on aggregated statistics
            total_loss += loss
            total_seen += num_predictions
            total_correct += num_predictions * accuracy

            avg_loss = float(total_loss / total_seen)
            avg_acc = float(total_correct / total_seen)
            avg_prp = np.exp(avg_loss)
            print(f"\r[Valid {index+1}/{num_batches}]\t loss={avg_loss:.3f}\t acc: {avg_acc:.3f}\t perp: {avg_prp:.3f}", end='')

        print()        
        return avg_prp, avg_acc
    
    def get_config(self):
        config = {"decoder":self.decoder}
        return config