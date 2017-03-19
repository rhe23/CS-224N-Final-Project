#generate variable lengths as samples
import numpy as np
import argparse, os, collections, csv, sys, cPickle
import tensorflow as tf
from processing_utils import get_embeddings, get_data, get_batch, get_dev_test_sets, get_masks
#general structure of tf workflow adapted from HW

os.chdir("../..")

max_length = 0

class Config:

    def __init__(self, max_length, embed_size, output_size, n_features =1 , n_classes=0, hidden_unit_size = 100, batch_size = 256, n_epochs = 10, num_layers =1, learning_rate=0.05, drop_out = 0.5,
                 sequence_length = 10):
        self.sequence_length = sequence_length
        self.dev_set_size =0.1
        self.test_set_size = 0
        self.classify= False #determines if we're running a classification
        self.n_features = n_features #number of features for each word in the data
        self.drop_out = drop_out
        self.n_classes = n_classes
        self.max_length = max_length #longest length of all our sentences
        self.hidden_unit_size = hidden_unit_size
        self.batch_size =batch_size
        self.n_epoches = n_epochs
        self.embed_size =embed_size
        self.output_size = output_size #the size of the vocab
        self.learning_rate = learning_rate
        self.numlayers = num_layers

def generate_padded_seq(max_length, vocab_length, sentences):
    return np.array([sentence + [vocab_length-1]*(max_length-len(sentence)) for sentence in sentences], dtype=np.int32) #vocab_length-1 because that's the index for the NULL vector in our embedding matrix


def get_sequence(max_length, sequence_length):
    sequences = []
    data_indices = np.arange(max_length)

    for i in xrange(0, max_length-1, sequence_length):
        if i+sequence_length <= max_length-1:
            sequences.append(data_indices[i:i+sequence_length])
        else: sequences.append(data_indices[i:max_length-1])
    return np.array(sequences)

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
        # self.probs = self.get_probabilities(self.pred)
        self.last_state = self.get_last_state(self.pred)

    def add_placeholders(self):

        self.input_placeholder = tf.placeholder(tf.int32, shape=(None, None)) #n_features should just be 1 here (the index into the embeddings) first None ==batch_size

        self.dropout_placeholder = tf.placeholder(tf.float32)

        self.mask_placeholder = tf.placeholder(tf.bool, [None, None]) #None here is to account for variable batch_size

        self.labels_placeholder = tf.placeholder(tf.int32, shape= (None, None)) #None here is batch_size

        self.sequence_placeholder = tf.placeholder(tf.int32, [None]) #none here is to account for variable batch_size

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
        embeddings = tf.reshape(embeddings, [-1, 1, self.config.n_features * self.config.embed_size]) #1 here since we're only feeding one word at a time
        self.x = embeddings

    def set_cell(self):

        self.cell = tf.nn.rnn_cell.LSTMCell(num_units=self.config.hidden_unit_size,
                                        initializer=tf.contrib.layers.xavier_initializer(), activation=tf.sigmoid, state_is_tuple=True)
        self.cell = tf.nn.rnn_cell.DropoutWrapper(cell = self.cell, output_keep_prob=self.dropout_placeholder)
        # self.cell = tf.contrib.rnn.MultiRNNCell([self.cell]*self.config.num_layers, state_is_tuple=False)
        self.cell = tf.nn.rnn_cell.OutputProjectionWrapper(cell=self.cell, output_size=self.config.output_size)

        # self.cell = tf.nn.rnn_cell.MultiRNNCell([self.cell] * self.config.numlayers)

    def training(self): #main training function for the model
        #sets up the construction of the graphs such that when session is called these operations will run

        self.add_embeddings()
        self.set_cell()
        # state = tf.Variable(self.cell.zero_state(self.config.batch_size, dtype = tf.float32), trainable=False) #initial state

        outputs, last_state = tf.nn.dynamic_rnn(self.cell, inputs = self.x, dtype = tf.float32)
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
        # return outputs
        # return (last_state, outputs[:, 0:outputs.get_shape()[1]-1,:])
        return outputs

    def get_last_state(self, preds):

        return tf.reshape(tf.nn.softmax(preds), [self.config.output_size])


    def get_probabilities(self, preds):

        # remove_last_state = preds[:, 0:preds.get_shape()[1]-1,:]
        # return (remove_last_state)
        # return (tf.argmax(remove_last_state, axis=2))

        return(tf.argmax(preds, axis =2))

        # return (tf.nn.softmax(preds), tf.shape(tf.nn.softmax(preds)))
        # return (tf.shape(preds), tf.shape(self.sequence_placeholder))
        # return (tf.shape(preds), tf.shape(self.mask_placeholder[:, 0:self.mask_placeholder.get_shape()[1]-1]))

    def calc_loss(self, preds):
        #preds is of the form: (batch_size, max_length, output_size)
        #preds[:, time_step,: ] = batch_size x output_size
        #calculate loss across every pair of words

        #get the raw predicted not in probability form
        # pred = preds[:, 0:preds.get_shape()[1]-1,:]
        pred = preds

        loss = tf.reshape(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.labels_placeholder, logits = pred), [-1])
        loss = tf.reduce_mean(tf.boolean_mask(loss, tf.reshape(self.mask_placeholder, [-1]) ))
        # losses =tf.reshape(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.labels_placeholder, logits = pred), [-1])
        # mask = tf.reshape(self.mask_placeholder[:, 0:self.mask_placeholder.get_shape()[1]-1], [-1])
        # loss = tf.reduce_mean(tf.boolean_mask(losses, mask))
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

        feed = self.create_feed_dict(inputs_batch=batch_x, labels_batch= batch_y, dropout= self.config.drop_out, mask_batch=masks)

        # _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        # loss = sess.run(self.error, feed_dict=feed)
        pred = sess.run(self.pred, feed_dict=feed)
        return pred
        # return _, loss

    def train_on_batch_single(self, sess, batch):
        seq_length = self.config.sequence_length
        total_loss = 0.

        max_len = max(len(case) for case in batch)
        padded = generate_padded_seq(max_len, self.config.output_size, batch)
        masks = np.matrix(get_masks(batch, max_len))
        batch_x = [i[:-1] for i in padded]
        batch_y = [i[1:] for i in padded]

        sequences = get_sequence(max_len, sequence_length=seq_length)
        #make the batches into a matrix so that we can have easier time feeding

        batch_x_mat = np.matrix(batch_x)
        batch_y_mat = np.matrix(batch_y)

        # assert batch_x_mat.shape[1] == batch_y_mat.shape[1], "x and y are not the same length. x: " +str(batch_x_mat.shape[1]) + ". y: " + str(batch_y_mat.shape[1])

        for i in sequences:
            x = batch_x_mat[:,i]
            y = batch_y_mat[:,i]
            m = masks[:,i]


            feed = self.create_feed_dict(inputs_batch=x, labels_batch= y, dropout= self.config.drop_out, mask_batch=m)

            _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
            # pred = sess.run(self.pred, feed_dict=feed)

            # preds = sess.run(self.loss, feed_dict=feed)

            total_loss += loss

        return total_loss/batch_x_mat.shape[1]

    def return_perplexity(self, loss):

        return 2**loss

    def run_epoch(self, sess, data):

        training_size = len(data)

        for i, indices in enumerate(get_batch(training_size, self.config.batch_size)):

            t = self.train_on_batch_single(sess, data[indices])

            print "Batch " + str(i+1) + " Loss: " + str(t)


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
    embeddings = get_embeddings(embed_path='./data/new_embeddings_final_filtered.pkl')
    # embeddings = np.load('./data/final_large_weights.npy')
    # embeddings = np.vstack([embeddings, np.zeros(embeddings.shape[1])])
    all_dat = collections.defaultdict(list)
    raw_data =  get_data(path = './data/2015_data_tokenzed.pkl')
    for r, post in raw_data:
        all_dat[r].append(post)

    # vocabs = collections.defaultdict(str)

    # with open('./data/large_vocab') as csvfile:
    #     vocab = csv.reader(csvfile)
    #     for v in vocab:
    #         vocabs[v[1]] = v[0]

    #get vocab:
    with open('./data/large_vocab_final_filtered.pkl', 'rb') as f:
        vocabs = cPickle.load(f)
        f.close()

    vocabs = collections.defaultdict(str, vocabs)

    def get_indices(sent):
        return [vocabs[i] for i in sent]

    vocabs_reversed = {v: k for k, v in vocabs.iteritems()}

    def get_words(sent):
        return [vocabs_reversed[i] for i in sent]

    r = args.subreddit
    sample = np.array([get_indices(j) for j in all_dat[r]])
    # subsample_y = [get_indices(j) for j[1:] in all_dat['personalfinance']][0:100]
    max_length = max(len(i) for i in sample)

    #seq_length, max_length, embed_size, output_size
    config_file = Config(drop_out=args.dropout, max_length = max_length, embed_size = embeddings.shape[1], output_size=embeddings.shape[0], batch_size = 2,
                         learning_rate = args.learningrate, hidden_unit_size=args.hiddensize, num_layers=args.numlayers, sequence_length=args.seqlength)

    idx = np.arange(len(sample))

    train_inds, dev_inds, test_inds = get_dev_test_sets(dev_size = config_file.dev_set_size, test_size = config_file.test_set_size, training_indices = idx)

    train, dev, test = sample[train_inds],  sample[dev_inds], sample[test_inds]

    with tf.Graph().as_default():
        m = RNN_LSTM(embeddings = embeddings, config = config_file)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)
            # loss = m.test_session(sess, train)
            best_perplexity = np.inf
            for epoch in range(n_epochs):
                print "Epoch: " + str(epoch + 1)

                m.run_epoch(sess, np.array(train))

                # # #evaluate test perplexity
                test_size = len(dev)
                total_perplexity = 0
                total_batches = 0

                for k, indices in enumerate(get_batch(test_size, 100)):

                    total_batches += 1
                    test_batch = dev[indices]

                    max_len = max(len(case) for case in test_batch)
                    padded = generate_padded_seq(max_len, config_file.output_size, test_batch)
                    masks = np.matrix(get_masks(test_batch, max_len))

                    batch_x = [i[:-1] for i in padded]
                    batch_y = [i[1:] for i in padded]

                    batch_x_mat = np.matrix(batch_x)
                    batch_y_mat = np.matrix(batch_y)

                    batch_perplexity = 0

                    sequences = get_sequence(max_len, sequence_length=config_file.seq_length)

                    for bat in sequences:
                        x = batch_x_mat[:,bat]
                        y = batch_y_mat[:,bat]
                        batch_mask = masks[:,bat]
                        feed = m.create_feed_dict(inputs_batch=x, labels_batch= y,
                                                  dropout= config_file.drop_out, mask_batch=batch_mask, seq_length = [1] * len(test_batch))

                        perplexities = sess.run(m.error, feed_dict=feed)
                        # print "Single word-pair perplexity: " + str(perplexities)
                        batch_perplexity += perplexities

                    total_perplexity += batch_perplexity

                    print "Epoch " + str(epoch + 1) + " Total test perplexity for batch " + str(k + 1) +  ' :' + str(batch_perplexity/batch_x_mat.shape[1])

                if total_perplexity < best_perplexity:
                    best_perplexity = total_perplexity
                    print "New Best Perplexity: " + str(best_perplexity)
                saver.save(sess, "./code/trainer/models/" + r.lower() + "/single_epoch_" + str(epoch + 1) + ".ckpt")

            with open('./code/trainer/diag/diagnostics_single_backprop.csv', 'a') as diag_out:
                csv_diag_out = csv.writer(diag_out)
                csv_diag_out.writerow([args.subreddit, str(best_perplexity), str(config_file.drop_out), str(config_file.hidden_unit_size), str(config_file.learning_rate), str(config_file.embed_size)])

def generate(args):

    embeddings = get_embeddings(embed_path='./data/new_embeddings_final_filtered.pkl')

    # vocabs = collections.defaultdict(str)

    with open('./data/large_vocab_final_filtered.pkl', 'rb') as f:
        vocabs = cPickle.load(f)
        f.close()

    vocabs = collections.defaultdict(str, vocabs)
    # with open('./data/large_vocab') as csvfile:
    #     vocab = csv.reader(csvfile)
    #     for v in vocab:
    #         vocabs[v[1]] = v[0]

    vocabs_reversed = {v: k for k, v in vocabs.iteritems()}

    def get_indices(sent):
        return [vocabs[i] for i in sent]

    def get_words(sent):
        return [vocabs_reversed[i] for i in sent]

    model = args.model
    model_path = './code/trainer/models/' + model +'/'

    c = Config(max_length = 1, embed_size = embeddings.shape[1], output_size=embeddings.shape[0], batch_size = 36, num_layers=args.numlayers, drop_out=1, sequence_length=args.seqlength) #max length is 1 becuase we want 1 word generated at a time

    with tf.Graph().as_default():

        m = RNN_LSTM(embeddings = embeddings, config = c)
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()

        with tf.Session() as session:
            session.run(init)
            saver.restore(session, tf.train.latest_checkpoint(model_path))

            all_sentences = []

            for sent in range(args.numsentences):

                current_word = '<start>'
                sentence = [current_word]
                #get index of <start> token:

                while current_word != '<end>':
                    current_ind =  vocabs[current_word]

                    x = [[current_ind]]

                    feed = m.create_feed_dict(inputs_batch=x, seq_length=[1])

                    preds = session.run(m.last_state, feed_dict=feed)

                    largest_10_inds = preds.argsort()[::-1][:args.numwords]
                    largest_10_unscaled_p = preds[largest_10_inds]
                    scaled_p = largest_10_unscaled_p/sum(largest_10_unscaled_p)
                    current_ind = np.random.choice(largest_10_inds, p = scaled_p)

                    current_word = vocabs_reversed[current_ind]
                    sentence.append(current_word)

                all_sentences.append(' '.join(sentence[1:-1]))

            for sentence in all_sentences:
                print sentence


def generator(args):

    choices = ['askreddit', 'lifeprotips', 'nottheonion', 'news', 'science', 'trees', 'tifu', 'personalfinance', 'mildlyinteresting', 'interestingasfuck']

    embeddings = get_embeddings(embed_path='./data/new_embeddings_final_filtered.pkl')

    # vocabs = collections.defaultdict(str)

    with open('./data/large_vocab_final_filtered.pkl', 'rb') as f:
        vocabs = cPickle.load(f)
        f.close()

    vocabs = collections.defaultdict(str, vocabs)
    # with open('./data/large_vocab') as csvfile:
    #     vocab = csv.reader(csvfile)
    #     for v in vocab:
    #         vocabs[v[1]] = v[0]

    vocabs_reversed = {v: k for k, v in vocabs.iteritems()}

    def get_indices(sent):
        return [vocabs[i] for i in sent]

    def get_words(sent):
        return [vocabs_reversed[i] for i in sent]

    c = Config(max_length = 1, embed_size = embeddings.shape[1], output_size=embeddings.shape[0], batch_size = 36, drop_out=1, sequence_length=args.seqlength) #max length is 1 becuase we want 1 word generated at a time

    with tf.Graph().as_default():

        m = RNN_LSTM(embeddings = embeddings, config = c)
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()

        with tf.Session() as session:
            session.run(init)

            print "Hello, please select the sureddit from which you would like to generate a post for: \n 1. AskReddit \n 2. LifeProTips \n 3. nottheonion \n 4. news \n 5. science" \
                  "\n 6. trees \n 7. tifu \n 8. personalfinance \n 9. mildlyinteresting \n 10.interestingasfuck "

            while True:

                try:
                    subr = raw_input("subreddit: ")
                    subr = subr.lower()
                    if subr not in choices:
                        print "Please input a correct subreddit."
                        continue
                    else:

                        model_path = './code/trainer/models/' + subr +'/'
                        saver.restore(session, tf.train.latest_checkpoint(model_path))

                        current_word = '<start>'
                        sentence = [current_word]
                        #get index of <start> token:

                        while current_word != '<end>':
                            current_ind =  vocabs[current_word]

                            x = [[current_ind]]

                            feed = m.create_feed_dict(inputs_batch=x, seq_length=[1])

                            preds = session.run(m.last_state, feed_dict=feed)

                            largest_inds = preds.argsort()[::-1][:100] #top 100
                            largest_unscaled_p = preds[largest_inds]
                            scaled_p = largest_unscaled_p/sum(largest_unscaled_p)
                            current_ind = np.random.choice(largest_inds, p = scaled_p)

                            current_word = vocabs_reversed[current_ind]
                            sentence.append(current_word)

                        print ' '.join(sentence[1:-1])

                        continue

                except EOFError:
                    continue



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LSTM model')
    subparser = parser.add_subparsers()

    parse = subparser.add_parser('train')
    parse.set_defaults(function = train)
    parse.add_argument('-r', '--subreddit', type =str, choices= ['AskReddit', 'LifeProTips', 'nottheonion', 'news', 'science', 'trees', 'tifu', 'personalfinance', 'mildlyinteresting', 'interestingasfuck'])
    parse.add_argument('-lr', '--learningrate', type = float, default=0.0004)
    parse.add_argument('-hs', '--hiddensize', type =int, default = 100)
    parse.add_argument('-do', '--dropout', type = float, default = 0.5)
    parse.add_argument('-l', '--numlayers', type=int, default = 2)
    parse.add_argument('-sq', '--seqlength', type = int, default = 10)

    parse = subparser.add_parser('generate') #generate phrases
    parse.set_defaults(function = generate)
    parse.add_argument('-g', '--model', type = str,choices= ['AskReddit', 'LifeProTips', 'nottheonion', 'news', 'science', 'trees', 'tifu', 'personalfinance', 'mildlyinteresting', 'interestingasfuck'])
    parse.add_argument('-nw', '--numwords', type = int)
    parse.add_argument('-n', '--numsentences', type = int)
    parse.add_argument('-l', '--numlayers', type=int, default = 2)
    parse.add_argument('-sq', '--seqlength', type = int, default = 10)


    parse = subparser.add_parser('generator', help='')
    parse.set_defaults(function = generator)
    parse.add_argument('-sq', '--seqlength', type = int, default = 10)

    ARGS = parser.parse_args()
    if ARGS.function is not None:

        ARGS.function(ARGS)

    else:

        sys.exit(1)