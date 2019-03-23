import tensorflow as tf
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.rnn import GRUCell


def cbow_forward(config, inputs, scope=None):
    with tf.variable_scope(scope or "forward"):

        JX, JQ = config.max_context_size, config.max_ques_size
        d = config.hidden_size
        x, x_len, q, q_len = [inputs[key] for key in ['x', 'x_len', 'q', 'q_len']]
        x_mask = tf.sequence_mask(x_len, JX) # [N, JX]
        q_mask = tf.sequence_mask(q_len, JQ) # [N, JQ]

        # emb_mat = tf.get_variable('emb_mat', shape=[V, d])
        emb_mat = config.emb_mat_ph if config.serve else config.emb_mat
        emb_mat = tf.slice(emb_mat, [2, 0], [-1, -1])
        emb_mat = tf.concat([tf.get_variable('emb_mat', shape=[2, d]), emb_mat], axis=0)
        xx = tf.nn.embedding_lookup(emb_mat, x, name='xx')  # [N, JX, d]
        qq = tf.nn.embedding_lookup(emb_mat, q, name='qq')  # [N, JQ, d]

        qq_avg = tf.reduce_mean(bool_mask(qq, q_mask, expand=True), axis=1)  # [N, d]
        qq_avg_exp = tf.expand_dims(qq_avg, axis=1)  # [N, 1, d]
        qq_avg_tiled = tf.tile(qq_avg_exp, [1, JX, 1])  # [N, JX, d]

        xq = tf.concat([xx, qq_avg_tiled, xx * qq_avg_tiled], axis=2)  # [N, JX, 3d]
        xq_flat = tf.reshape(xq, [-1, 3*d])  # [N * JX, 3*d]

        # Compute logits
        with tf.variable_scope('start'):
            logits1 = exp_mask(tf.reshape(tf.layers.dense(inputs=xq_flat, units=1), [-1, JX]), x_mask)  # [N, JX]
            yp1 = tf.argmax(logits1, axis=1)  # [N]
        with tf.variable_scope('stop'):
            logits2 = exp_mask(tf.reshape(tf.layers.dense(inputs=xq_flat, units=1), [-1, JX]), x_mask)  # [N, JX]
            yp2 = tf.argmax(logits2, axis=1)  # [N]

        outputs = {'logits1': logits1, 'logits2': logits2, 'yp1': yp1, 'yp2': yp2}
        variables = {'emb_mat': emb_mat}
        return variables, outputs

def rnn_forward(config, inputs, scope=None):
    with tf.variable_scope(scope or "forward"):
        JX, JQ = config.max_context_size, config.max_ques_size
        d = config.hidden_size
        x, x_len, q, q_len = [inputs[key] for key in ['x', 'x_len', 'q', 'q_len']]
        x_mask = tf.sequence_mask(x_len, JX)
        q_mask = tf.sequence_mask(q_len, JQ)

        # ph: place holder
        emb_mat = config.emb_mat_ph if config.serve else config.emb_mat
        emb_mat = tf.slice(emb_mat, [2, 0], [-1, -1])
        emb_mat = tf.concat([tf.get_variable('emb_mat', shape=[2, d]), emb_mat], axis=0)
        xx = tf.nn.embedding_lookup(emb_mat, x, name='xx')  # [N, JX, d]
        qq = tf.nn.embedding_lookup(emb_mat, q, name='qq')  # [N, JQ, d]

        gru_fw_x = tf.nn.rnn_cell.GRUCell(num_units=config.hidden_size, name="gru_fw_x")
        gru_bw_x = tf.nn.rnn_cell.GRUCell(num_units=config.hidden_size, name="gru_bw_x")
        gru_fw_q = tf.nn.rnn_cell.GRUCell(num_units=config.hidden_size, name="gru_fw_q")
        gru_bw_q = tf.nn.rnn_cell.GRUCell(num_units=config.hidden_size, name="gru_bw_q")
        if config.is_train:
            gru_fw_x = tf.nn.rnn_cell.DropoutWrapper(gru_fw_x, output_keep_prob=config.keep_prob)
            gru_bw_x = tf.nn.rnn_cell.DropoutWrapper(gru_bw_x, output_keep_prob=config.keep_prob)
            gru_fw_q = tf.nn.rnn_cell.DropoutWrapper(gru_fw_q, output_keep_prob=config.keep_prob)
            gru_bw_q = tf.nn.rnn_cell.DropoutWrapper(gru_bw_q, output_keep_prob=config.keep_prob)
        xx, xx_state = tf.nn.bidirectional_dynamic_rnn(gru_fw_x, gru_bw_x, xx, dtype=tf.float32)
        xx = tf.concat(xx, 2) # [N, JX, 2 * d]
        qq, qq_state = tf.nn.bidirectional_dynamic_rnn(gru_fw_q, gru_bw_q, qq, dtype=tf.float32)
        qq = tf.concat(qq, 2) # [N, JX, 2 * d]

        # zhen: orignally, it reduce JX dim into one
        qq_avg = tf.reduce_mean(bool_mask(qq, q_mask, expand=True), axis=1)  # [N, 2 * d]
        # expand the value into a (singleton) vector
        qq_avg_exp = tf.expand_dims(qq_avg, axis=1)  # [N, 1, 2 * d]
        # replicate the averages
        qq_avg_tiled = tf.tile(qq_avg_exp, [1, JX, 1])  # [N, JX, 2 * d]

        xq = tf.concat([xx, qq_avg_tiled, xx * qq_avg_tiled], axis=2)  # [N, JX, 6 * d]
        xq_flat = tf.reshape(xq, [-1, 6*d])  # [N * JX, 6*d]

        # Compute logits
        with tf.variable_scope('start'):
            logits1 = exp_mask(tf.reshape(tf.layers.dense(inputs=xq_flat, units=1), [-1, JX]), x_mask)  # [N, JX]
            yp1 = tf.argmax(logits1, axis=1)  # [N]
        with tf.variable_scope('stop'):
            logits2 = exp_mask(tf.reshape(tf.layers.dense(inputs=xq_flat, units=1), [-1, JX]), x_mask)  # [N, JX]
            yp2 = tf.argmax(logits2, axis=1)  # [N]

        outputs = {'logits1': logits1, 'logits2': logits2, 'yp1': yp1, 'yp2': yp2}
        variables = {'emb_mat': emb_mat}
        return variables, outputs

def attention_forward(config, inputs, scope=None):
    with tf.variable_scope(scope or "forward"):
        JX, JQ = config.max_context_size, config.max_ques_size
        d = config.hidden_size
        x, x_len, q, q_len = [inputs[key] for key in ['x', 'x_len', 'q', 'q_len']]
        x_mask = tf.sequence_mask(x_len, JX) # [N, JX]
        q_mask = tf.sequence_mask(q_len, JQ) # [N, JQ]
        xq_mask = tf.tile(tf.expand_dims(x_mask, axis=-1), [1, 1, JQ])
        xq_mask2 = tf.tile(tf.expand_dims(q_mask, axis=1), [1, JX, 1])
        xq_mask3 = tf.cast(xq_mask, tf.float32) * tf.cast(xq_mask2, tf.float32)

        emb_mat = config.emb_mat_ph if config.serve else config.emb_mat
        emb_mat = tf.slice(emb_mat, [2, 0], [-1, -1])
        emb_mat = tf.concat([tf.get_variable('emb_mat', shape=[2, d]), emb_mat], axis=0)
        xx = tf.nn.embedding_lookup(emb_mat, x, name='xx')  # [N, JX, d]
        qq = tf.nn.embedding_lookup(emb_mat, q, name='qq')  # [N, JQ, d]

        gru_fw_x = tf.nn.rnn_cell.GRUCell(num_units=config.hidden_size, name="gru_fw_x")
        gru_bw_x = tf.nn.rnn_cell.GRUCell(num_units=config.hidden_size, name="gru_bw_x")
        gru_fw_q = tf.nn.rnn_cell.GRUCell(num_units=config.hidden_size, name="gru_fw_q")
        gru_bw_q = tf.nn.rnn_cell.GRUCell(num_units=config.hidden_size, name="gru_bw_q")
        if config.is_train:
            gru_fw_x = tf.nn.rnn_cell.DropoutWrapper(gru_fw_x, output_keep_prob=config.keep_prob)
            gru_bw_x = tf.nn.rnn_cell.DropoutWrapper(gru_bw_x, output_keep_prob=config.keep_prob)
            gru_fw_q = tf.nn.rnn_cell.DropoutWrapper(gru_fw_q, output_keep_prob=config.keep_prob)
            gru_bw_q = tf.nn.rnn_cell.DropoutWrapper(gru_bw_q, output_keep_prob=config.keep_prob)
        xx, xx_state = tf.nn.bidirectional_dynamic_rnn(gru_fw_x, gru_bw_x, xx, dtype=tf.float32)
        xx = tf.concat(xx, 2) # [N, JX, 2 * d]
        qq, qq_state = tf.nn.bidirectional_dynamic_rnn(gru_fw_q, gru_bw_q, qq, dtype=tf.float32)
        qq = tf.concat(qq, axis=-1) # [N, JQ, 2 * d]

        xx_2 = tf.expand_dims(xx, 2) # [N, JX, 1, 2d]
        xx_3 = tf.tile(xx_2, [1, 1, JQ, 1])  # [N, JX, JQ, 2d]
        qq_2 = tf.expand_dims(qq, 1) # [N, 1, JQ, 2d]
        qq_3 = tf.tile(qq_2, [1, JX, 1, 1])  # [N, JX, JQ, 2d]

        xq_attn = tf.concat([xx_3, qq_3, xx_3 * qq_3], axis=-1)  # [N, JX, JQ, 6 * d]
        xq_flat_attn = tf.reshape(xq_attn, [-1, 6*d])  # [N * JX * JQ, 6*d]

        attn_logits = exp_mask(tf.reshape(tf.layers.dense(inputs=xq_flat_attn, units=1, use_bias=True), [-1, JX, JQ]), xq_mask3)
        attn_weight = tf.nn.softmax(attn_logits) # [N, JX, JQ]
        attn_weight_T = tf.transpose(tf.nn.softmax(attn_weight, dim=1), (0, 2, 1)) # [N, JQ, JX]
        attn_weight_X = tf.matmul(attn_weight, attn_weight_T) # [N, JX, JX]

        # qq : [N, JQ, 2 * d]
        # xx : [N, JX, 2 * d]
        qq_attn = tf.matmul(attn_weight, qq) # [N, JX, 2 * d]

        xq = tf.concat([xx, qq_attn, xx * qq_attn], axis=2)  # [N, JX, 6 * d]
        xq_flat = tf.reshape(xq, [-1, 6*d])  # [N * JX, 6*d]

        # Compute logits
        with tf.variable_scope('start'):
            logits1 = exp_mask(tf.reshape(tf.layers.dense(inputs=xq_flat, units=1), [-1, JX]), x_mask)  # [N, JX]
            yp1 = tf.argmax(logits1, axis=1)  # [N]
        with tf.variable_scope('stop'):
            logits2 = exp_mask(tf.reshape(tf.layers.dense(inputs=xq_flat, units=1), [-1, JX]), x_mask)  # [N, JX]
            yp2 = tf.argmax(logits2, axis=1)  # [N]

        outputs = {'logits1': logits1, 'logits2': logits2, 'yp1': yp1, 'yp2': yp2}
        variables = {'emb_mat': emb_mat}
        return variables, outputs

def get_loss(config, inputs, outputs, scope=None):
    with tf.name_scope(scope or "loss"):
        y1, y2 = inputs['y1'], inputs['y2']
        logits1, logits2 = outputs['logits1'], outputs['logits2']
        loss1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y1, logits=logits1))
        loss2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y2, logits=logits2))
        loss = loss1 + loss2
        acc1 = tf.reduce_mean(tf.cast(tf.equal(y1, tf.cast(tf.argmax(logits1, 1), 'int32')), 'float'))
        acc2 = tf.reduce_mean(tf.cast(tf.equal(y2, tf.cast(tf.argmax(logits2, 1), 'int32')), 'float'))
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('acc1', acc1)
        tf.summary.scalar('acc2', acc2)
        return loss


def exp_mask(val, mask, expand=False):
    if expand:
        mask = tf.expand_dims(mask, -1)
    return val - (1.0 - tf.cast(mask, 'float')) * 10.0e10


def bool_mask(val, mask, expand=False):
    if expand:
        mask = tf.expand_dims(mask, -1)
    return val * tf.cast(mask, 'float')
