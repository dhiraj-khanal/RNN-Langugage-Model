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

        # TODO: initialize embedding_size, batch_size, and any other hyperparameters

        self.vocab_size = vocab_size
        self.window_size = 20
        self.embedding_size = 50 # TODO
        self.batch_size = 30  # TODO

        # TODO: initialize embeddings and forward pass weights (weights, biases)
        # Note: You can now use tf.keras.layers!
        # - use tf.keras.layers.Dense for feed forward layers
        # - and use tf.keras.layers.GRU or tf.keras.layers.LSTM for your RNN
        self.learning_rate = 0.01
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        def create_variable(dims):
            return tf.Variable(tf.random.normal(dims, stddev=.1, dtype=tf.float32))

        self.embedding_matrix = create_variable([self.vocab_size, self.embedding_size])
        self.LSTM = tf.keras.layers.LSTM(100, return_sequences=True, return_state=True)
        self.dense1 = tf.keras.layers.Dense(200, activation='relu')
        self.dense2 = tf.keras.layers.Dense(self.vocab_size, activation='softmax')

    def call(self, inputs, initial_state):
        """
        - You must use an embedding layer as the first layer of your network
        (i.e. tf.nn.embedding_lookup)
        - You must use an LSTM or GRU as the next layer.

        :param inputs: word ids of shape (batch_size, window_size)
        :param initial_state: 2-d array of shape (batch_size, rnn_size) as a tensor
        :return: the batch element probabilities as a tensor, a final_state
        (NOTE 1: If you use an LSTM, the final_state will be the last two RNN outputs,
        NOTE 2: We only need to use the initial state during generation)
        using LSTM and only the probabilites as a tensor and a final_state as a tensor when using GRU
        """

        # TODO: Fill in
        embed = tf.nn.embedding_lookup(self.embedding_matrix, inputs)
        logits, output_1, output_2 = self.LSTM(embed, initial_state = initial_state)
        final_state = (output_1, output_2)
        relu = self.dense1(logits)
        prob = self.dense2(relu)
        return prob, final_state

    def loss(self, probabilities, labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction

        :param probabilities: a matrix of shape (batch_size, window_size, vocab_size) as a tensor
        :param labels: matrix of shape (batch_size, window_size) containing the labels
        :return: the average loss of the model as a tensor of size 1
        """

        # TODO: Fill in
        # We recommend using tf.keras.losses.sparse_categorical_crossentropy

        return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, probabilities, axis=-1))


def train(model, train_inputs, train_labels):
    """
    Runs through one epoch - all training examples (remember to batch!)
    Here you will also want to reshape your inputs and labels so that they match
    the inputs and labels shapes passed in the call and loss functions respectively.

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,)
    :param train_labels: train labels (all labels for training) of shape (num_labels,)
    :return: None
    """
    # TODO: Fill in
    indices = tf.range(start=0, limit=tf.shape(train_labels)[0], dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)

    x = tf.gather(train_inputs, shuffled_indices)
    y = tf.gather(train_labels, shuffled_indices)

    l = len(train_inputs)%model.window_size

    trainInputs = x[:-l]
    trainLabels = y[:-l]
    shapes = (-1, model.window_size)
    x = np.reshape(trainInputs, shapes)
    y = np.reshape(trainLabels, shapes)

    for i in range(0, len(y), model.batch_size):
        with tf.GradientTape() as tape:
            ypred, state = model.call(x[i:i+model.batch_size], initial_state=None)
            loss = model.loss(ypred, y[i:i+model.batch_size])
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test(model, test_inputs, test_labels):
    """
    Runs through one epoch - all testing examples (remember to batch!)
    Here you will also want to reshape your inputs and labels so that they match
    the inputs and labels shapes passed in the call and loss functions respectively.

    :param model: the trained model to use for prediction
    :param test_inputs: train inputs (all inputs for testing) of shape (num_inputs,)
    :param test_labels: train labels (all labels for testing) of shape (num_labels,)
    :returns: perplexity of the test set
    """

    # TODO: Fill in
    # NOTE: Ensure a correct perplexity formula (different from raw loss)
    acc = 0
    l = len(test_inputs)%model.window_size
    test_inputs = test_inputs[:-l]
    test_labels = test_labels[:-l]
    shapes = (-1, model.window_size)

    test_inputs = np.reshape(test_inputs, shapes)
    test_labels = np.reshape(test_labels, shapes)
    num_batches = int(len(test_labels) / model.batch_size)

    for i in range(num_batches):
        ypred, state = model.call(test_inputs[i*model.batch_size:(i+1)*model.batch_size], initial_state=None)
        true_labels = test_labels[i*model.batch_size:(i+1)*model.batch_size]
        acc += model.loss(ypred, true_labels)
    return np.exp(acc/num_batches)


def generate_sentence(word1, length, vocab, model, sample_n=10):
    """
    Takes a model, vocab, selects from the most likely next word from the model's distribution

    :param model: trained RNN model
    :param vocab: dictionary, word to id mapping
    :return: None
    """

    # NOTE: Feel free to play around with different sample_n values

    reverse_vocab = {idx: word for word, idx in vocab.items()}
    previous_state = None

    first_string = word1
    first_word_index = vocab[word1]
    next_input = [[first_word_index]]
    text = [first_string]

    for i in range(length):
        logits, previous_state = model.call(next_input, previous_state)
        logits = np.array(logits[0, 0, :])
        top_n = np.argsort(logits)[-sample_n:]
        n_logits = np.exp(logits[top_n]) / np.exp(logits[top_n]).sum()
        out_index = np.random.choice(top_n, p=n_logits)

        text.append(reverse_vocab[out_index])
        next_input = [[out_index]]

    print(" ".join(text))


def main():
    # TODO: Pre-process and vectorize the data
    # HINT: Please note that you are predicting the next word at each timestep,
    # so you want to remove the last element from train_x and test_x.
    # You also need to drop the first element from train_y and test_y.
    # If you don't do this, you will see impossibly small perplexities.
    trainList, testList, id = get_data('data/train.txt', 'data/test.txt')

    train_inputs = trainList[:len(trainList)-1]
    train_labels = trainList[1:len(trainList)]
    test_inputs = testList[:len(testList)-1]
    test_labels = testList[1:len(testList)]

    # TODO: Separate your train and test data into inputs and labels

    # TODO: initialize model and tensorflow variables

    model = Model(len(id))
    print("Rnn Model")
    # TODO: Set-up the training step
    train(model, train_inputs, train_labels)
    print('finished_training')
    # TODO: Set up the testing steps
    perplexity = test(model, np.array(test_inputs), np.array(test_labels))
    # Print out perplexity
    print(perplexity)
    # BONUS: Try printing out various sentences with different start words and sample_n parameters
    pass


if __name__ == "__main__":
    main()
