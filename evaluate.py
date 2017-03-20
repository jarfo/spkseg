from __future__ import print_function
import codecs
import numpy as np
import re
import argparse
import cPickle as pickle
from model.LSTMCNN import LSTMCNN
from util.BatchLoaderUnk import ENCODING, process_line, vocab_unpack
from math import exp


class Vocabulary:
    def __init__(self, tokens, vocab_file, max_word_l=65):
        self.tokens = tokens
        self.max_word_l = max_word_l
        self.prog = re.compile('\s+')

        print('loading vocabulary file...')
        vocab_mapping = np.load(vocab_file)
        self.idx2word, self.word2idx, self.idx2char, self.char2idx = vocab_unpack(vocab_mapping)
        self.vocab_size = len(self.idx2word)
        print('Word vocab size: %d, Char vocab size: %d' % (len(self.idx2word), len(self.idx2char)))
        self.word_vocab_size = len(self.idx2word)
        self.char_vocab_size = len(self.idx2char)

    def index(self, word):
        w = self.word2idx[word] if word in self.word2idx else self.word2idx[self.tokens.UNK]
        c = np.zeros(self.max_word_l, dtype='int32')
        chars = [self.char2idx[self.tokens.START]] # start-of-word symbol
        chars += [self.char2idx[char] for char in word if char in self.char2idx]
        chars.append(self.char2idx[self.tokens.END]) # end-of-word symbol
        if len(chars) >= self.max_word_l:
            chars[self.max_word_l-1] = self.char2idx[self.tokens.END]
            c = chars[:self.max_word_l]
        else:
            c[:len(chars)] = chars

        return w, c


    def get_input(self, word, batch_size, p=None):
        if word == '<unk>':
            word = self.tokens.UNK # replace <unk> with a single character
        w, c = self.index(word)
        output_words = [w]
        output_chars = [c]
        words = np.array(output_words, dtype='int32')[np.newaxis, :]
        words = np.tile(words, (batch_size, 1))
        chars = np.array(output_chars, dtype='int32')[np.newaxis, :, :]
        chars = np.tile(chars, (batch_size, 1, 1))
        x = {'word':words, 'chars':chars}

        if p is not None:
            prb = np.array([p], dtype='float32')[np.newaxis, :, :]
            prb = np.tile(prb, (batch_size, 1, 1))
            x['prb'] = prb

        return x


class Evaluator:
    def __init__(self, name, vocabulary, init, k):
        self.opt = pickle.load(open('{}.pkl'.format(name), "rb"))
        self.opt.seq_length = 1
        self.logk = 1
        while (1 << self.logk) < k:
            self.logk += 1
        self.opt.batch_size = 1 << self.logk
        self.reader = Vocabulary(self.opt.tokens, vocabulary, max_word_l=self.opt.max_word_l)
        self.model = LSTMCNN(self.opt)
        self.model.load_weights('{}.h5'.format(name))

        self.hyp_score = np.zeros(k, dtype='float32')
        self.hyp_samples = np.empty((k,0), dtype='int32')
        self.hyp_prob = np.empty((k,0), dtype='float32')
        if init:
            self.state_mean = np.load(init)
        else:
            self.state_mean = None

    def clear(self, reset):
        self.hyp_score.fill(0);
        self.hyp_samples.resize((self.opt.batch_size,0))
        self.hyp_prob.resize((self.opt.batch_size,0))
        if reset:
            self.model.reset_states()
            if self.state_mean != None:
                self.model.set_state_updates_value(self.state_mean)

    @property
    def delay(self):
        return self.opt.delay

    def gen_sample(self, word, prb=None):
        x = self.reader.get_input(word, self.opt.batch_size, prb)

        # previous hyp spk id
        ncol = self.hyp_samples.shape[1]
        if ncol > 0:
            spk = self.hyp_samples[:,-1].astype('float32')
        else:
            spk = np.zeros(self.opt.batch_size, dtype='float32')
            spk[1::2] = 1
        x['spk'] = spk[:,np.newaxis];

        # spk prediction
        y = self.model.predict(x, batch_size = self.opt.batch_size)[:,0,0]

        # sort new scores
        if ncol < self.logk-1:
            spk_indices = np.arange(self.opt.batch_size) % (1 << (ncol+2)) >= (1 << (ncol+1))
            spk_prob = y.copy()
            spk_prob[spk_indices==0] = 1 - y[spk_indices==0]
            # update states
            self.hyp_samples = np.append(self.hyp_samples, spk_indices[:,np.newaxis], axis=1)
            self.hyp_prob = np.append(self.hyp_prob, spk_prob[:,np.newaxis], axis=1)
            self.hyp_score += -np.log(spk_prob)
        else:
            spk_prob = np.concatenate((1-y, y))
            cand_score = np.concatenate((self.hyp_score, self.hyp_score)) - np.log(spk_prob)
            ranks = cand_score.argsort()[:self.opt.batch_size]
            batch_indices = ranks % self.opt.batch_size;
            spk_indices = ranks / self.opt.batch_size;
            # update states
            self.model.reindex_states(batch_indices)
            self.hyp_samples = np.append(self.hyp_samples[batch_indices], spk_indices[:,np.newaxis], axis=1)
            self.hyp_prob = np.append(self.hyp_prob[batch_indices], spk_prob[ranks,np.newaxis], axis=1)
            self.hyp_score = cand_score[ranks]

    def get_prob(self, word, prb):
        assert(self.opt.batch_size == 1)
        x = self.reader.get_input(word, self.opt.batch_size, prb)
        y = self.model.predict(x, batch_size = self.opt.batch_size)[0,0,0]
        self.hyp_samples = np.append(self.hyp_samples, [y>0.5], axis=1)
        self.hyp_prob = np.append(self.hyp_prob, [y], axis=1)
        self.hyp_score = [self.hyp_score - np.log(np.maximum(y, 1-y))]


def save_spkseg(ev, lines, calc, state_sum, nl, file):
    yp = ev.hyp_samples[0, ev.delay+1:].copy()
    pp = ev.hyp_prob[0, ev.delay+1:].copy()
    prev_filename = ''
    prev_spk = ''
    prev_line = ''
    for i, line in enumerate(lines):
        filename, _, spk = process_line(line)
        if (filename != prev_filename) or (spk != prev_spk):
            first_word = 1;
            prev_filename = filename
            prev_spk = spk
        else:
            first_word = 0;
        if i>0 and i <= len(yp):
            ls = prev_line.split()
            l = ' '.join(ls[0:5])
            print(l, first_word, yp[i-1], pp[i-1], file=file)
        prev_line = line

    if calc:
        for ssum, update in zip(state_sum, ev.model.states_value):
            ssum += update[0]
            nl += 1
        ev.clear(False)
    else:
        ev.clear(True)
    return state_sum, nl


def main(name, vocabulary, init, itext, otext, calc, k):

    ev = Evaluator(name, vocabulary, None if calc else init, k)

    fin = codecs.open(itext, 'r', ENCODING)
    fout = codecs.open(otext, 'w', ENCODING)

    state_sum = [np.zeros_like(a[0]) for a in ev.model.states_value] if calc else None
    nl = 0
        
    lines = []
    for line in fin:
        line = line.strip()
        if line == "" or line.startswith(';;'):
            continue
        _, word, _, prb = process_line(line, ev.opt.vector_size)
        if ev.opt.use_spk:
            ev.gen_sample(word, prb)
        else:
            ev.get_prob(word, prb)
        lines.append(line)
        
    state_sum, nl = save_spkseg(ev, lines, calc, state_sum, nl, file=fout)

    if calc:
        state_mean = [a[0]/nl for a in state_sum]
        np.save(init, state_mean)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--vocabulary', type=str)
    parser.add_argument('--init', type=str)
    parser.add_argument('--itext', type=str)
    parser.add_argument('--otext', type=str)
    parser.add_argument('--calc', action='store_true', default=False)
    parser.add_argument('--beam', type=int, default=8)

    args = parser.parse_args()

    main(args.model, args.vocabulary, args.init, args.itext, args.otext, args.calc, args.beam)
