import logging
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.state_ops import variable_op

class TextCNN():
    def __init__(self, filter_sizes, num_filters, sequence_length,
                 need_placeholder=True, vocab_size = None, embedding_size = None, w2v_model = None,
                 top = True, num_classes = 0):
        """
        params:
            need_placeholder: 是否需要占位，如果是作为其他模型子组件就不需要
            top: 是否需要进行分类
        """
        assert need_placeholder is False or (need_placeholder and vocab_size and embedding_size)
        assert top is False or (top and num_classes)

        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.sequence_length = sequence_length
        self.need_placeholder = need_placeholder
        self.top = top
        if need_placeholder:
            self.input_x = tf.placeholder(tf.int32, [None, sequence_length])
            self.input_y = tf.placeholder(tf.int32, [None, num_classes])
        if self.need_placeholder:
            self.vocab_size = vocab_size
            self.embedding_size = embedding_size
            self.w2v_model = w2v_model
            if w2v_model is not None:
                assert w2v_model.shape[0] == vocab_size and w2v_model.shape[1] == embedding_size
        if top:
            self.num_classes = num_classes

        
    
    def forward(self, input_x = None, strides=1, padding='valid'):
        """
        在这里定义计算图
        self.output = xxx
        """
        if not self.need_placeholder:   
            with tf.name_scope('textcnn/embedding'):
                if self.w2v_model is None:
                    self.W = tf.Variable(tf.random_uniform[self.vocab_size, self.embedding_size],
                                        name = 'w2v')
                else:
                    self.W = tf.get_variable('w2v',
                                            initializer=self.w2v_model.astype(tf.int32))
                self.conv1d_input = tf.nn.embedding_lookup(self.W, input_x)
        else:
            self.conv1d_input = self.input_x
        
        with tf.name_scope('textcnn/conv1d'):
            self.pooled = []
            for i, kernel_size in enumerate(self.filter_sizes):
                conv1d_output = tf.layers.Conv1D(self.conv1d_input,
                                        kernel_size,
                                        self.num_filters,
                                        strides=strides,
                                        padding = padding,
                                        name = 'conv1d_{}'.format(i)
                                        ) # [N, L-k+s, num_filters]
                conv1d_activation = tf.nn.relu(conv1d_output)
                pool_output = tf.layers.max_pooling1d(conv1d_activation,
                                                    self.sequence_length - kernel_size + strides,
                                                    padding=padding,
                                                    strides=strides,
                                                    name = 'pool_{}'.format(i)
                                                    ) # [N, 1, num_filters]
                self.pooled.append(pool_output)
            self.pool_result = tf.concat(self.pooled, -1)
            self.pool_output = tf.reshape(self.pooled, [-1, self.num_filters*len(self.filter_sizes)])
        if self.top:
            self.output = tf.layers.dense(self.pool_output, self.num_classes)
        else:
            self.output = self.pool_output



    def cal_metrics(self):
        # todo 添加对预测任务支持
        if self.top is False or self.need_placeholder is False:
            logging.warning('top is False， no need to calculate metrics')
            return
        with tf.name_scope('textcnn/prediction'):
            self.prediction = tf.argmax(self.output, 1, name = 'predictions')
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.output,
                                                                labels=self.input_y)
            self.loss = tf.reduce_mean(losses, name = 'loss')
        with tf.accuracy('textcnn/accuracy'):
            self.accuracy = tf.reduce_mean(
                tf.equal(self.prediction, tf.argmax(self.input_y, 1)),
                name = 'accuracy'
            )
                                                            
            

        

    def build(self, **kwargs):
        self.forward(**kwargs)
        self.cal_metrics()

    def show(self):
        """
        输出输入输出维度
        """
        pass