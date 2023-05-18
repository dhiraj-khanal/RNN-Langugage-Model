from functools import reduce

import numpy as np
import tensorflow as tf
from preprocess import get_data
from tensorflow.keras import Model
import os

# ensures that we run only on cpu
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class Model(tf.keras.Model):
    def __init__(self, vocab_size):
        """
        The Model class predicts the next words in a sequence.

        :param vocab_size: The number of unique words in the data
        """

        super(Model, self).__init__()

        # TODO: initialize emnbedding_size, batch_size, and any other hyperparameters

        self.vocab_size = vocab_size
        self.embedding_size = 20  # TODO
        self.batch_size = 30  # TODO
        self.vocab_size = vocab_size

        # TODO: initialize embeddings and forward pass weights (weights, biases)
        self.learning_rate = 3e-4
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        def create_variable(dims):
            return tf.Variable(tf.random.normal(dims, stddev=.1, dtype=tf.float32))

        self.embedding_matrix = create_variable([self.vocab_size, self.embedding_size])
        self.dense = tf.keras.layers.Dense(self.vocab_size, activation='softmax')
        #self.w2 = ([self.batch_size, self.vocab_size])
        #self.b2 = self.vocab_size

    def call(self, inputs):
        """
        You must use an embedding layer as the first layer of your network
        (i.e. tf.nn.embedding_lookup)

        :param inputs: word ids of shape (batch_size, 2)
        :return: probabilities: The batch element probabilities as a tensor of shape (batch_size, vocab_size)
        """

        # TODO: Fill in
        '''
        #print(np.shape(inputs[:, 0]))
        first_word = tf.nn.embedding_lookup(self.embedding_matrix, inputs[:, 0])
        second_word = tf.nn.embedding_lookup(self.embedding_matrix, inputs[:, 1])
        
       #embed_total = tf.concat([first_word, second_word], axis=1)
        embed_total = tf.concat([first_word, second_word], axis=1)

        '''
        embed = tf.nn.embedding_lookup(self.embedding_matrix, inputs)
        embed_total = tf.concat([embed[:, 0], embed[:, 1]], axis=1)
        prob = self.dense(embed_total)
        return prob

    def loss_function(self, probabilities, labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction

        :param probabilities: a matrix of shape (batch_size, vocab_size)
        :return: the average loss of the model as a tensor of size 1
        """
        # TODO: Fill in
        # We recommend using tf.keras.losses.sparse_categorical_crossentropy
        return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, probabilities,axis=-1))



def train(model, train_input, train_labels):
    """
    Runs through one epoch - all training examples.
    Remember to shuffle your inputs and labels - ensure that they are shuffled
    in the same order. Also you should batch your input and labels here.

    :param model: the initilized model to use for forward and backward pass
    :param train_input: train inputs (all inputs for training) of shape (num_inputs,2)
    :param train_input: train labels (all labels for training) of shape (num_inputs,)
    :return: None
    """

    # TODO Fill in
    indices = tf.range(start=0, limit=tf.shape(train_labels)[0], dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)

    x = tf.gather(train_input, shuffled_indices)
    y = tf.gather(train_labels, shuffled_indices)
    #print(len(y))
    for i in range(0, len(y), model.batch_size):
        with tf.GradientTape() as tape:
            ypred = model.call(x[i:i+model.batch_size])
            loss = model.loss_function(ypred, y[i:i+model.batch_size])
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))




def test(model, test_input, test_labels):
    """
    Runs through all test examples. Test input should be batched here.

    :param model: the trained model to use for prediction
    :param test_input: train inputs (all inputs for testing) of shape (num_inputs,2)
    :param test_input: train labels (all labels for testing) of shape (num_inputs,)
    :returns: perplexity of the test set
    """

    # TODO: Fill in

    acc = 0
    num_batches = int(len(test_labels) / model.batch_size)
    for i in range(num_batches):
        ypred = model.call(test_input[i*model.batch_size:(i+1)*model.batch_size])
        true_labels = test_labels[i*model.batch_size:(i+1)*model.batch_size]
        acc += model.loss_function(ypred, true_labels)
    return np.exp(acc/num_batches)

    # NOTE: Ensure a correct perplexity formula (different from raw loss)
    '''
    n = int(len(test_labels)/model.batch_size)
    for i in range(0, len(test_labels), model.batch_size):
        ypred = model.call(test_input[i:i+model.batch_size])
        acc += model.loss_function(ypred, test_labels[i:i+model.batch_size])
    return np.exp(acc/n)
    '''

def generate_sentence(word1, word2, length, vocab, model):
    """
    Given initial 2 words, print out predicted sentence of targeted length.

    :param word1: string, first word
    :param word2: string, second word
    :param length: int, desired sentence length
    :param vocab: dictionary, word to id mapping
    :param model: trained trigram model

    """

    # NOTE: This is a deterministic, argmax sentence generation

    reverse_vocab = {idx: word for word, idx in vocab.items()}
    output_string = np.zeros((1, length), dtype=np.int)
    output_string[:, :2] = vocab[word1], vocab[word2]

    for end in range(2, length):
        start = end - 2
        output_string[:, end] = np.argmax(model(output_string[:, start:end]), axis=1)
    text = [reverse_vocab[i] for i in list(output_string[0])]

    print(" ".join(text))


def main():
    # TODO: Pre-process and vectorize the data using get_data from preprocess
    ''''''
    trainList, testList, id = get_data('data/train.txt', 'data/test.txt')

    # TO-DO:  Separate your train and test data into inputs and labels
    train_inputs = []
    train_labels = trainList[2:]

    test_inputs = []
    test_labels = testList[2:]

    for i in range(2, len(trainList)):
        train_inputs += [[trainList[i-2], trainList[i-1]]]

    for i in range(2, len(testList)):
        test_inputs += [[testList[i-2], testList[i-1]]]

    #print(np.shape(train_inputs)) #(1465612, 2)
    #print(np.shape(test_inputs)) #(361910, 2)


    # TODO: initialize model
    print("Model")
    model = Model(len(id))

    # TODO: Set-up the training step
    train(model, train_inputs, np.array(train_labels))
    print('finished_training')
    # TODO: Set up the testing steps
    perplexity = test(model, np.array(test_inputs), np.array(test_labels))
    # Print out perplexity
    print(perplexity)
    # BONUS: Try printing out sentences with different starting words

    pass


if __name__ == "__main__":
    main()
