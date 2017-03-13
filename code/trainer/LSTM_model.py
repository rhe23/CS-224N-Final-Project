import numpy as np
import math, os, collections, csv
import cPickle
import tensorflow as tf
from scipy.sparse import identity

os.chdir("../..")
def get_embeddings(embed_path = './data/large_weights.pkl_iter100'):

    with open(embed_path, 'rb') as f:
        embeddings = cPickle.load(f)
        f.close()
    return embeddings

max_length = 0

def get_data(path):

    with open(path, 'rb') as f:
        input = cPickle.load(f)
        f.close()

    return input


def add_padding(max_length, sentences, num_vocab):
    #num_vocab is the total number of vocab in the sentence, we add 1 to this so we don't index into a word
    #max_length is the max length of the sentence
    #assume we're getting a list of tokenized sentences with index values
    #returns a tuple of padded sentences [2,4,5,..,0,0,0,0] ,[true,true,true,...,false,false]

   return zip([[True]*len(i) + [False]*(max_length-len(i)) for i in sentences], [sentence + (max_length - len(sentence))*[num_vocab+1] for sentence in sentences])


def get_batch(data_size, batch_size, shuffle=True):
#returns a list of indices for each batch of data during optimization
    print data_size
    data_indices = np.arange(data_size)
    np.random.shuffle(data_indices)

    for i in xrange(0, data_size, batch_size):
       yield data_indices[i:i+batch_size]

def get_dev_test_sets(dev_size, test_size, training_indices):
    #dev_size should be a float between 0.0 and 1.0
    #returns a list of indices for both training and dev sets

    total_sizes = dev_size+test_size
    temp_inds = np.random.choice(training_indices, math.floor(total_sizes*len(training_indices)), replace = False)
    training_inds = [i for i in training_indices if i not in temp_inds]
    dev_inds = temp_inds[:len(temp_inds)/2]
    test_inds = temp_inds[len(temp_inds)/2:]

    return (training_inds, dev_inds, test_inds)

def get_masks(sentences, max_length):
    return np.array([len(sentence)*[True] + (max_length-len(sentence))*[False] for sentence in sentences])

class Config:

    def __init__(self, max_length, embed_size, output_size, n_features =1 , n_classes=0, hidden_unit_size = 10, batch_size = 256, n_epochs = 10):
        self.dev_set_size =0.1
        self.test_set_size = 0.1
        self.classify= False #determines if we're running a classification
        self.n_features = n_features #number of features for each word in the data
        self.drop_out = 0.5
        self.n_classes = n_classes
        self.max_length = max_length #longest length of all our sentences
        self.hidden_unit_size = hidden_unit_size
        self.batch_size =batch_size
        self.n_epoches = n_epochs
        self.embed_size =embed_size
        self.output_size = output_size #the size of the vocab
        self.learning_rate = 0.05

def generate_padded_seq(max_length, vocab_length, sentences):
    return np.array([sentence + [vocab_length-1]*(max_length-len(sentence)) for sentence in sentences], dtype=np.int32)

class RNN_LSTM:

    def __init__(self, embeddings, x, y, config, data_size):

        self.data_size = data_size
        self.config = config
        self.pretrain_embeddings = embeddings

        idx = list(range(data_size))

        train_inds, dev_inds, test_inds  = get_dev_test_sets(dev_size = self.config.dev_set_size, test_size = self.config.test_set_size, training_indices = idx)

        self.train_x, self.dev_x, self.test_x = x[train_inds],  x[dev_inds], x[test_inds]

        self.train_y, self.dev_y, self.test_y = y[train_inds],  y[dev_inds], y[test_inds]

        # #build model steps
        self.add_placeholders() #initiate placeholders
        self.pred = self.prediction() #forward prop
        self.loss = self.calc_loss(self.pred) #calculate loss
        self.train_op = self.back_prop(self.loss) #optimization step

    def add_placeholders(self):

        self.input_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size, None)) #n_features should just be 1 here (the index into the embeddings)

        self.dropout_placeholder = tf.placeholder(tf.float32)

        self.mask_placeholder = tf.placeholder(tf.bool, [self.config.batch_size, self.config.max_length])

        self.labels_placeholder = tf.placeholder(tf.int32, shape= (self.config.batch_size, None))

        self.sequence_placeholder = tf.placeholder(tf.int32, [self.config.batch_size])

    def create_feed_dict(self, inputs_batch, labels_batch=None, mask_batch = None, seq_length = None, dropout = 1):

        #dropout is the keep probability
        feed_dict = {self.input_placeholder : inputs_batch}

        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        if self.config.drop_out:
            feed_dict[self.dropout_placeholder] = dropout
        if mask_batch is not None:
            feed_dict[self.mask_placeholder] = mask_batch
        if seq_length is not None:
            feed_dict[self.sequence_placeholder] = seq_length

        return feed_dict

    def test_session(self, session): #diagnostic for testing pieces of the model
        #take random sample of training set:
        # inds = np.random.choice(self.training_set, math.floor(0.01*len(self.training_set)), replace=False)
        inds = self.training_set
        small_s = generate_padded_seq(self.config.max_length, self.config.output_size, inds)
        small_s_y = [i[1:] for i in small_s]
        seq_len = [len(i) for i in inds]
        masks = get_masks(inds, self.config.max_length)

        feed = self.create_feed_dict(inputs_batch= small_s, labels_batch= small_s_y, dropout= self.config.drop_out, mask_batch=masks, seq_length = seq_len)

        loss = session.run([self.loss], feed_dict=feed)
        return loss

    def add_embeddings(self):
        #use this function to convert inputs to a tensor of shape (none, sentence length (self.config.max_length), number of features * embedding size
        # (self.config.n_features * self.config.embed_size)
        embeddings = tf.Variable(self.pretrain_embeddings, dtype = tf.float32)
        embeddings = tf.nn.embedding_lookup(embeddings, ids = self.input_placeholder)
        embeddings = tf.reshape(embeddings, [-1, self.config.max_length, self.config.n_features * self.config.embed_size])
        self.x = embeddings

    def set_cell(self):
        self.cell = tf.nn.rnn_cell.LSTMCell(num_units=self.config.hidden_unit_size, input_size=self.config.batch_size,
                                                initializer=tf.contrib.layers.xavier_initializer())
        self.cell = tf.nn.rnn_cell.DropoutWrapper(cell = self.cell, output_keep_prob=self.dropout_placeholder)
        self.cell = tf.nn.rnn_cell.OutputProjectionWrapper(cell=self.cell, output_size=self.config.output_size)

    def prediction(self): #main training function for the model
        #sets up the construction of the graphs such that when session is called these operations will run
        preds = []
        self.add_embeddings()
        self.set_cell()
        # state = tf.Variable(self.cell.zero_state(self.config.batch_size, dtype = tf.float32), trainable=False) #initial state

        outputs, states = tf.nn.dynamic_rnn(self.cell, inputs = self.x, dtype = tf.float32, sequence_length=self.sequence_placeholder)
        # outputs = tf.transpose(outputs, [1,0,2])
        #
        # W = tf.get_variable("W2", shape = [self.config.hidden_unit_size, self.config.output_size], initializer=tf.contrib.layers.xavier_initializer() )
        # b = tf.get_variable("b2", shape = [self.config.output_size], initializer=tf.constant_initializer(0) )
        #
        # for time_step in range(self.config.max_length):
        #     out = tf.gather(outputs, ( time_step ) )
        #     y_t = tf.matmul(out, W) + b
        #     preds.append(y_t)
        #
        # preds = tf.pack(preds, axis=1)
        preds = outputs
        return preds

    def calc_loss(self, preds):
        #preds is of the form: (batch_size, max_length, output_size)
        #preds[:, time_step,: ] = batch_size x output_size
        #calculate loss across every pair of words
        # one_hot_mats = tf.one_hot(self.vocab_by_inds, depth = self.config.batch_size, axis = 0)
        #
        loss = 0

        for time_step in range(self.config.max_length-1):
            loss += tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(preds[:, time_step, :], self.labels_placeholder[:,time_step]))
            # one_hot = tf.transpose(tf.gather(one_hot_mats, self.labels_placeholder[:,time_step]))
            # loss += -tf.reduce_sum(tf.matmul(tf.log(preds[:, time_step, :]), one_hot))

        # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(preds, self.labels_placeholder)
        # loss = tf.boolean_mask(loss, self.mask_placeholder)
        # loss = tf.reduce_mean(loss)

        # return loss
        return loss

    def back_prop(self, loss):

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config.learning_rate)
        train_op = optimizer.minimize(loss)

        return train_op

    def train_on_batch(self, sess, batch_indices):

        batch_x= generate_padded_seq(self.config.max_length, self.config.output_size, batch_indices)
        batch_y = [i[1:] for i in batch_x]
        seq_len = [len(i) for i in batch_indices]
        masks = get_masks(batch_indices, self.config.max_length)

        feed = self.create_feed_dict(inputs_batch=batch_x, labels_batch= batch_y, dropout= self.config.drop_out, mask_batch=masks, seq_length = seq_len)

        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        # loss = sess.run([self.pred], feed_dict=feed)
        return loss

    def get_error(self, sess):

        incorrect = 0

        for time_step in range(self.config.max_length-1):

            incorrect +=tf.not_equal(tf.argmax(self.pred[:, time_step, :], axis = 0),
                                 self.labels_placeholder[:,time_step])

        return tf.reduce_mean(incorrect)

    def run_epoch(self, sess, get_error = False):

        training_size = len(self.train_x)

        for i, indices in enumerate(get_batch(training_size, self.config.batch_size)):
            trainx = self.train_x[indices]
            # trainy = self.train_y[indices]
            loss = self.train_on_batch(sess, trainx)
            print ("Batch " + str(i) + " Loss: " + str(loss))

        if get_error == True:
            train_x = generate_padded_seq(self.config.max_length, self.config.output_size, self.train_x)
            train_y = [i[1:] for i in train_x]
            seq_len = [len(i) for i in self.train_x]
            masks = get_masks(self.train_x, self.config.max_length)

            train_feed = self.create_feed_dict(inputs_batch=train_x, labels_batch= train_y, dropout= self.config.drop_out, mask_batch=masks, seq_length = seq_len)
            trainerror = sess.run([self.get_error], feed_dict=train_feed)

            print "training error: " + str(trainerror)

            test_x = generate_padded_seq(self.config.max_length, self.config.output_size, self.test_x)
            test_y = [i[1:] for i in test_x]
            seq_len_test = [len(i) for i in self.test_x]
            masks_test = get_masks(self.test_x, self.config.max_length)

            test_feed = self.create_feed_dict(inputs_batch=test_x, labels_batch= test_y, dropout= self.config.drop_out, mask_batch=masks_test, seq_length = seq_len_test)
            testerror = sess.run([self.get_error], feed_dict=test_feed)

            print "test error: " + str(testerror)

def main():
    n_epochs = 20
    embeddings = get_embeddings()
    embeddings = np.vstack([embeddings, np.zeros(embeddings.shape[1])])
    all_dat = collections.defaultdict(list)
    raw_data =  get_data(path = './data/2015_data')
    for r, post in raw_data:
        all_dat[r].append(post)

    vocabs = collections.defaultdict(str)

    with open('./data/large_vocab.csv') as csvfile:
        vocab = csv.reader(csvfile)
        for v in  vocab:
            vocabs[v[1]] = v[0]

    def get_indices(sent):
        return [vocabs[i] for i in sent]

    subsample_x = [get_indices(j) for j in all_dat['personalfinance']][0:10]
    subsample_y = [get_indices(j) for j[1:] in all_dat['personalfinance']][0:10]
    max_length = max(len(i) for i in subsample_x)

    #seq_length, max_length, embed_size, output_size
    c = Config(max_length = max_length, embed_size = embeddings.shape[1], output_size=embeddings.shape[0], batch_size = 1)
    m = RNN_LSTM(embeddings = embeddings, x= np.array(subsample_x), y = np.array(subsample_y), config= c, data_size=len(subsample_x))
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(n_epochs):
            print "Epoch: " + str(i)
            m.run_epoch(sess)

if __name__ == '__main__':
    main()