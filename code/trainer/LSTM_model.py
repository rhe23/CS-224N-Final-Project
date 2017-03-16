import numpy as np
import argparse, os, collections, csv, sys
import cPickle
import tensorflow as tf
from processing_utils import get_embeddings, get_data, get_batch, get_dev_test_sets, get_masks
#general structure of tf workflow adapted from HW


os.chdir("../..")

max_length = 0

class Config:

    def __init__(self, max_length, embed_size, output_size, n_features =1 , n_classes=0, hidden_unit_size = 10, batch_size = 256, n_epochs = 10):
        self.dev_set_size =0.1
        self.test_set_size = 0
        self.classify= False #determines if we're running a classification
        self.n_features = n_features #number of features for each word in the data
        self.drop_out = 1
        self.n_classes = n_classes
        self.max_length = max_length #longest length of all our sentences
        self.hidden_unit_size = hidden_unit_size
        self.batch_size =batch_size
        self.n_epoches = n_epochs
        self.embed_size =embed_size
        self.output_size = output_size #the size of the vocab
        self.learning_rate = 0.1

def generate_padded_seq(max_length, vocab_length, sentences):
    return np.array([sentence + [vocab_length-1]*(max_length-len(sentence)) for sentence in sentences], dtype=np.int32)

class RNN_LSTM:

    def __init__(self, embeddings, config):

        # self.data_size = data_size
        self.config = config
        self.pretrain_embeddings = embeddings

        # idx = list(range(data_size))

        # train_inds, dev_inds, test_inds  = get_dev_test_sets(dev_size = self.config.dev_set_size, test_size = self.config.test_set_size, training_indices = idx)
        #
        # self.train_x, self.dev_x, self.test_x = x[train_inds],  x[dev_inds], x[test_inds]
        #
        # self.train_y, self.dev_y, self.test_y = y[train_inds],  y[dev_inds], y[test_inds]

        # #build model steps
        self.add_placeholders() #initiate placeholders
        self.pred = self.training() #forward prop
        self.loss = self.calc_loss(self.pred) #calculate loss
        self.train_op = self.back_prop(self.loss) #optimization step
        self.error = self.return_perplexity(self.loss)
        self.probs = self.get_probabilities(self.pred)

    def add_placeholders(self):

        self.input_placeholder = tf.placeholder(tf.int32, shape=(None, None)) #n_features should just be 1 here (the index into the embeddings) first None ==batch_size

        self.dropout_placeholder = tf.placeholder(tf.float32)

        self.mask_placeholder = tf.placeholder(tf.bool, [None, self.config.max_length]) #None here is batch_size

        self.labels_placeholder = tf.placeholder(tf.int32, shape= (None, None)) #None here is batch_size

        self.sequence_placeholder = tf.placeholder(tf.int32, [None]) #none here is batch_size

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

    def add_embeddings(self):
        #use this function to convert inputs to a tensor of shape (none, sentence length (self.config.max_length), number of features * embedding size
        # (self.config.n_features * self.config.embed_size)
        embeddings = tf.Variable(self.pretrain_embeddings, dtype = tf.float32, trainable=True)
        embeddings = tf.nn.embedding_lookup(embeddings, ids = self.input_placeholder)
        embeddings = tf.reshape(embeddings, [-1, self.config.max_length, self.config.n_features * self.config.embed_size])
        self.x = embeddings

    def set_cell(self):

        self.cell = tf.nn.rnn_cell.LSTMCell(num_units=self.config.hidden_unit_size,
                                        initializer=tf.contrib.layers.xavier_initializer(), activation=tf.sigmoid)
        self.cell = tf.nn.rnn_cell.DropoutWrapper(cell = self.cell, output_keep_prob=self.dropout_placeholder)
        self.cell = tf.nn.rnn_cell.OutputProjectionWrapper(cell=self.cell, output_size=self.config.output_size)

    def training(self): #main training function for the model
        #sets up the construction of the graphs such that when session is called these operations will run

        self.add_embeddings()
        self.set_cell()
        # state = tf.Variable(self.cell.zero_state(self.config.batch_size, dtype = tf.float32), trainable=False) #initial state

        outputs, states = tf.nn.dynamic_rnn(self.cell, inputs = self.x, dtype = tf.float32, sequence_length=self.sequence_placeholder)
        # outputs = tf.transpose(outputs, [1,0,2])
        # # # # Gather last output slice
        #
        # preds = []
        # W = tf.get_variable("W2", shape = [self.config.hidden_unit_size, self.config.output_size], initializer=tf.contrib.layers.xavier_initializer() )
        # b = tf.get_variable("b2", shape = [self.config.output_size], initializer=tf.constant_initializer(0) )
        # # #
        # for time_step in range(self.config.max_length):
        #     out = tf.gather(outputs, ( time_step ) )
        #     y_t = tf.matmul(out, W) + b
        #     preds.append(y_t)
        # # return (tf.reduce_sum(tf.pack(preds, axis=1)))
        # probs = tf.nn.softmax(tf.pack(preds, axis=1))
        # preds = tf.pack(preds, axis=1)

        # probs = tf.nn.softmax(outputs)

        return outputs[:, 0:outputs.get_shape()[1]-1,:]

    def get_probabilities(self, preds):

        return (tf.argmax(preds, axis=2))

        return (tf.nn.softmax(preds), tf.shape(tf.nn.softmax(preds)))
        # return (tf.shape(preds), tf.shape(self.sequence_placeholder))
        # return (tf.shape(preds), tf.shape(self.mask_placeholder[:, 0:self.mask_placeholder.get_shape()[1]-1]))


    def calc_loss(self, preds):
        #preds is of the form: (batch_size, max_length, output_size)
        #preds[:, time_step,: ] = batch_size x output_size
        #calculate loss across every pair of words
        # one_hot_mats = tf.one_hot(self.vocab_by_inds, depth = self.config.batch_size, axis = 0)
        # loss = 0
     #get the raw predicted not in probability form

        losses =tf.reshape(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.labels_placeholder, logits = preds), [-1])
        mask = tf.reshape(self.mask_placeholder[:, 0:self.mask_placeholder.get_shape()[1]-1], [-1])
        loss = tf.reduce_mean(tf.boolean_mask(losses, mask))
        # mask = tf.reshape(self.mask_placeholder, [-1])
        # loss = tf.boolean_mask()
        # for sample in range(self.config.batch_size):

        #     # losses +=  tf.reduce_sum(tf.boolean_mask(loss[sample, :], self.mask_placeholder[sample,:]))
        # losses = tf.boolean_mask(tf.gather(loss, list(range(self.config.max_length))), tf.gather(tf.transpose(self.mask_placeholder), list(range(self.config.max_length))))

        # print self.mask_placeholder.get_shape()

        return loss

    def back_prop(self, loss):

        optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
        train_op = optimizer.minimize(loss)

        return train_op

    def train_on_batch(self, sess, batch):

        batch_x = generate_padded_seq(self.config.max_length, self.config.output_size, batch)

        batch_y = [i[1:] for i in batch_x]

        seq_len = [len(i) for i in batch]

        masks = get_masks(batch, self.config.max_length)

        feed = self.create_feed_dict(inputs_batch=batch_x, labels_batch= batch_y, dropout= self.config.drop_out, mask_batch=masks, seq_length = seq_len)

        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        # loss = sess.run(self.error, feed_dict=feed)
        # pred = sess.run(self.probs, feed_dict=feed)
        # return pred
        return _, loss

    # def predict_on_batch(self, sess, batch):
    #

    def return_perplexity(self, loss):
        #returns the list of probabilities of the actual word labels for all cases in a given batch
        # t = tf.gather_nd(self.pred[0], tf.stack([tf.tile(tf.expand_dims(tf.range(tf.shape(self.labels_placeholder)[0]), 1), [1, tf.shape(self.labels_placeholder)[1]]), tf.transpose(tf.tile(tf.expand_dims(tf.range(tf.shape(self.labels_placeholder)[1]), 1), [1, tf.shape(self.labels_placeholder)[0]])), self.labels_placeholder], 2))

        return 2**loss

    def run_epoch(self, sess, data):

        training_size = len(data)

        for i, indices in enumerate(get_batch(training_size, self.config.batch_size)):

            # trainy = self.train_y[indices]
            t = self.train_on_batch(sess, data[indices])

            print ("Batch " + str(i) + " Loss: " + str(t[1]))

        # if get_error == True:
        #     trainx = generate_padded_seq(self.config.max_length, self.config.output_size, self.train)
        #     trainy = [i[1:] for i in trainx]
        #     seq_len = [len(i) for i in trainx]
        #     masks = get_masks(trainx, self.config.max_length)
        #
        #     train_feed = self.create_feed_dict(inputs_batch=trainx, labels_batch= trainy, dropout= self.config.drop_out, mask_batch=masks, seq_length = seq_len)
        #     train_error = self.get_error(sess, feed =train_feed)
        #
        #     print "training error: " + str(train_error)
        #
        #     test_x = generate_padded_seq(self.config.max_length, self.config.output_size, self.test)
        #     test_y = [i[1:] for i in test_x]
        #     seq_len_test = [len(i) for i in self.test_x]
        #     masks_test = get_masks(test_x, self.config.max_length)
        #
        #     test_feed = self.create_feed_dict(inputs_batch=test_x, labels_batch= test_y, dropout= self.config.drop_out, mask_batch=masks_test, seq_length = seq_len_test)
        #     test_error = self.get_error(sess, feed =test_feed)
        #
        #     print "test error: " + str(test_error)

    def test_session(self, session, inds): #for debugging purposes only
        #take random sample of training set:

        # inds = np.random.choice(self.training_set, math.floor(0.01*len(self.training_set)), replace=False)
        small_s = generate_padded_seq(self.config.max_length, self.config.output_size, inds)
        small_s_y = [i[1:] for i in small_s]
        seq_len = [len(i) for i in inds]
        masks = get_masks(inds, self.config.max_length)

        feed = self.create_feed_dict(inputs_batch= small_s, labels_batch= small_s_y, dropout= self.config.drop_out, mask_batch=masks, seq_length = seq_len)

        loss = session.run([self.loss], feed_dict=feed)
        return loss


def train(args):
    n_epochs = 20
    embeddings = get_embeddings()
    embeddings = np.vstack([embeddings, np.zeros(embeddings.shape[1])])
    all_dat = collections.defaultdict(list)
    raw_data =  get_data(path = './data/2015_data')
    for r, post in raw_data:
        all_dat[r].append(post)

    vocabs = collections.defaultdict(str)

    with open('./data/large_vocab_new.csv') as csvfile:
        vocab = csv.reader(csvfile)
        for v in vocab:
            vocabs[v[1]] = v[0]

    def get_indices(sent):
        return [vocabs[i] for i in sent]

    vocabs_reversed = {v: k for k, v in vocabs.iteritems()}

    def get_words(sent):
        return [vocabs_reversed[str(i)] for i in sent]

    r = args.subreddit
    sample = np.array([get_indices(j) for j in all_dat[r]])
    # subsample_y = [get_indices(j) for j[1:] in all_dat['personalfinance']][0:100]
    max_length = max(len(i) for i in sample)

    #seq_length, max_length, embed_size, output_size
    c = Config(max_length = max_length, embed_size = embeddings.shape[1], output_size=embeddings.shape[0], batch_size = 2)

    idx = np.arange(len(sample))

    train_inds, dev_inds, test_inds = get_dev_test_sets(dev_size = c.dev_set_size, test_size = c.test_set_size, training_indices = idx)

    train, dev, test = sample[train_inds],  sample[dev_inds], sample[test_inds]

    with tf.Graph().as_default():
        m = RNN_LSTM(embeddings = embeddings, config = c)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)
            # loss = m.test_session(sess, train)
            best_perplexity = np.inf
            for j in range(n_epochs):
                print "Epoch: " + str(j)
                #
                m.run_epoch(sess, np.array(train))

                # #evaluate training perplexity
                test_size = len(dev)
                total_perplexity = 0
                total_batches = 0
                for k, indices in enumerate(get_batch(test_size, 100)):

                    total_batches += 1
                    test_batch = test[indices]
                    masks = get_masks(test_batch, c.max_length)

                    seq_len = [len(i) for i in test_batch]
                    batch_x = generate_padded_seq(c.max_length, c.output_size, test_batch)
                    batch_y = [i[1:] for i in batch_x]
                    feed = m.create_feed_dict(inputs_batch=batch_x, labels_batch= batch_y, dropout= c.drop_out, mask_batch=masks, seq_length = seq_len)

                    perplexities = sess.run(m.error, feed_dict=feed)
                    total_perplexity += perplexities
                    # seq_inds = np.arange(len(seq_len))
                    # print "Average Perplexity Across Entire Set: " + str(sum([np.prod(perplexities[i][0:seq_len[i]])**(-1/seq_len[i]) for i in seq_inds])/len(seq_inds))
                    print "Epoch: " + str(j) + " average test perplexity for batch " + str(k) +  ':' + str(perplexities)

                if (total_perplexity/total_batches) < best_perplexity:
                    best_perplexity = (total_perplexity/total_batches)
                    print "New Best Perplexity: " + str(best_perplexity)
                    saver.save(sess, "code/trainer/" + r + "_epoch_" + str(j) + ".ckpt")

                    # #generate outputted sentence using the best weights:
                    predicted_indices = []
                    actual_sentences = []
                    for k, indices in enumerate(get_batch(test_size, 2)):

                        test_batch = test[indices]
                        # actual_sentences += test_batch
                        masks = get_masks(test_batch, c.max_length)

                        for case in test_batch:
                            actual_sentences.append(get_words(case))

                        seq_len = [len(i) for i in test_batch]
                        batch_x = generate_padded_seq(c.max_length, c.output_size, test_batch)
                        batch_y = [i[1:] for i in batch_x]
                        feed = m.create_feed_dict(inputs_batch=batch_x, labels_batch= batch_y, dropout= c.drop_out, mask_batch=masks, seq_length = seq_len)

                        probabilities_unmasked = sess.run(m.probs, feed_dict= feed)

                        seq_inds = np.arange(len(seq_len))

                        for row in seq_inds:
                            predicted_indices.append(probabilities_unmasked[row][0:seq_len[row]])

                        predicted_words = [get_words(j) for j in predicted_indices]
                    predicted_pairs = zip(predicted_words, actual_sentences)

                    with open('./code/trainer/results/' + r + '_epoch_' + str(j) + '.csv', 'wb') as out:
                        csv_out=csv.writer(out)
                        csv_out.writerow(['Predicted','Actual'])
                        for row in predicted_pairs:
                            csv_out.writerow(row)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LSTM model')
    subparser = parser.add_subparsers()

    parse = subparser.add_parser('train')
    parse.set_defaults(function = train)
    parse.add_argument('-r', '--subreddit', type =str, choices= ['AskReddit', 'LifeProTips', 'nottheonion', 'news', 'science', 'trees', 'tifu', 'personalfinance', 'mildlyinteresting', 'interestingasfuck'])


    ARGS = parser.parse_args()
    if ARGS.function is not None:

        ARGS.function(ARGS)

    else:

        sys.exit(1)