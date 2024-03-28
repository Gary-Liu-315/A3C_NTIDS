'''
    评估器 模型构建
    Date:2024/03
    Author:刘韦豪
'''
import tensorflow as tf

def build_shared_network(X, add_summaries=False):
    """
    构建一个共享网络模型，其中包含三个全连接层。
    
    参数：
        X：输入
        add_summaries：如果为 true，则向 Tensorboard 添加层摘要。
    返回：
        最终层激活。
    """

    # 三个卷积层
    in_layer = tf.contrib.layers.fully_connected(
        inputs=X,
        num_outputs=256,
        activation_fn=tf.nn.relu,
        scope="In_layer")
    hidd_layer1 = tf.contrib.layers.fully_connected(
        inputs=in_layer,
        num_outputs=256,
        activation_fn=tf.nn.relu, 
        scope="hidden1")
    hidd_layer2 = tf.contrib.layers.fully_connected(
        inputs=hidd_layer1,
        num_outputs=256,
        activation_fn=tf.nn.relu, 
        scope="hidden2")
    
    out = tf.contrib.layers.fully_connected(
        inputs=hidd_layer2,
        num_outputs=256,
        scope="out")
    
    if add_summaries:
        tf.contrib.layers.summarize_activation(in_layer)
        tf.contrib.layers.summarize_activation(hidd_layer1)
        tf.contrib.layers.summarize_activation(hidd_layer2)
        tf.contrib.layers.summarize_activation(out)
    
    return out

class PolicyEstimator():
    """
    策略函数近似器。给定观测值，返回概率
    在所有可能的动作上。
    
    参数：
        num_outputs：动作空间的大小。
        reuse：如果为 true，则将重用现有的共享网络。
        trainable：如果为 true，我们将训练操作添加到网络中。
          不更新其本地模型且不需要的执行组件线程
          训练操作会将其设置为 false。
    """

    def __init__(self, num_outputs,observation_space, reuse=False, trainable=True):
        self.num_outputs = num_outputs
        self.observation_space = observation_space
    
        self.states = tf.compat.v1.placeholder(shape=[None, self.observation_space], 
                                     dtype=tf.float32, name="X")
        # TD目标值（基于即时奖励和下一状态的预估价值来替代当前状态在状态序列结束时可能得到的收获）
        self.targets = tf.compat.v1.placeholder(shape=[None], dtype=tf.float32, name="y")
        # 选择操作的整数 ID
        self.actions = tf.compat.v1.placeholder(shape=[None], dtype=tf.int32, name="actions")
    
        # 正常化
        X = tf.to_float(self.states)
        batch_size = tf.shape(self.states)[0]
    
        # 与价值网络共享的图表
        with tf.compat.v1.variable_scope("shared", reuse=reuse):
            out = build_shared_network(X, add_summaries=(not reuse))
    
    
        with tf.compat.v1.variable_scope("policy_net"):
            self.logits = tf.contrib.layers.fully_connected(out, num_outputs, activation_fn=None)
            self.probs = tf.nn.softmax(self.logits) + 1e-8
    
            self.predictions = {
            "logits": self.logits,
            "probs": self.probs
            }
    
            # 在损失中增加熵
            self.entropy = -tf.reduce_sum(self.probs * tf.math.log(self.probs), 1, name="entropy")
            self.entropy_mean = tf.reduce_mean(self.entropy, name="entropy_mean")
    
            # 仅获取所选操作的预测
            gather_indices = tf.range(batch_size) * tf.shape(self.probs)[1] + self.actions
            self.picked_action_probs = tf.gather(tf.reshape(self.probs, [-1]), gather_indices)
            
            # 探索性参数 beta
            beta = 0.01
            
            self.losses = - (tf.math.log(self.picked_action_probs) * self.targets + beta * self.entropy)
            self.loss = tf.reduce_sum(self.losses, name="loss")
    
            tf.compat.v1.summary.scalar(self.loss.op.name, self.loss)
            tf.compat.v1.summary.scalar(self.entropy_mean.op.name, self.entropy_mean)
            tf.compat.v1.summary.histogram(self.entropy.op.name, self.entropy)
    
            if trainable:
                self.optimizer = tf.compat.v1.train.AdamOptimizer(0.0002)
                self.optimizer = tf.compat.v1.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
                self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
                self.grads_and_vars = [[grad, var] for grad, var in self.grads_and_vars if grad is not None]
                self.train_op = self.optimizer.apply_gradients(self.grads_and_vars,
                global_step=tf.compat.v1.train.get_global_step())

        # 合并此网络和共享网络（但不包括价值网络）的摘要
        var_scope_name = tf.compat.v1.get_variable_scope().name
        summary_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.SUMMARIES)
        sumaries = [s for s in summary_ops if "policy_net" in s.name or "shared" in s.name]
        sumaries = [s for s in summary_ops if var_scope_name in s.name]
        self.summaries = tf.compat.v1.summary.merge(sumaries)


class ValueEstimator():
    """
    值函数近似器。返回一批观测值的值估计器。
    
    参数：
        reuse：如果为 true，则将重用现有的共享网络。
        trainable：如果为 true，我们将训练操作添加到网络中。
          不更新其本地模型且不需要的执行组件线程
          训练操作会将其设置为 false。
    """

    def __init__(self,observation_space, reuse=False, trainable=True):
        
        self.observation_space = observation_space

        self.states = tf.compat.v1.placeholder(shape=[None, observation_space], dtype=tf.float32, name="X")
        # TD目标价值
        self.targets = tf.compat.v1.placeholder(shape=[None], dtype=tf.float32, name="y")
    
        X = tf.to_float(self.states)
    
        # 与Value Net共享的图表
        with tf.compat.v1.variable_scope("shared", reuse=reuse):
            out = build_shared_network(X, add_summaries=(not reuse))
    
        with tf.compat.v1.variable_scope("value_net"):
            self.logits = tf.contrib.layers.fully_connected(
                    inputs=out,
                    num_outputs=1,
                    activation_fn=None)
            self.logits = tf.squeeze(self.logits, squeeze_dims=[1], name="logits")
    
            self.losses = tf.math.squared_difference(self.logits, self.targets)
            self.loss = tf.reduce_sum(self.losses, name="loss")
            
            self.predictions = {"logits": self.logits}
    
            # 摘要
            prefix = tf.compat.v1.get_variable_scope().name
            tf.compat.v1.summary.scalar(self.loss.name, self.loss)
            tf.compat.v1.summary.scalar("{}/max_value".format(prefix), tf.reduce_max(self.logits))
            tf.compat.v1.summary.scalar("{}/min_value".format(prefix), tf.reduce_min(self.logits))
            tf.compat.v1.summary.scalar("{}/mean_value".format(prefix), tf.reduce_mean(self.logits))
            tf.compat.v1.summary.scalar("{}/reward_max".format(prefix), tf.reduce_max(self.targets))
            tf.compat.v1.summary.scalar("{}/reward_min".format(prefix), tf.reduce_min(self.targets))
            tf.compat.v1.summary.scalar("{}/reward_mean".format(prefix), tf.reduce_mean(self.targets))
            tf.compat.v1.summary.histogram("{}/reward_targets".format(prefix), self.targets)
            tf.compat.v1.summary.histogram("{}/values".format(prefix), self.logits)
    
            if trainable:
                self.optimizer = tf.compat.v1.train.AdamOptimizer(0.0002)
#                self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
                self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
                self.grads_and_vars = [[grad, var] for grad, var in self.grads_and_vars if grad is not None]
                self.train_op = self.optimizer.apply_gradients(
                        self.grads_and_vars,
                        global_step=tf.compat.v1.train.get_global_step())
    
        var_scope_name = tf.compat.v1.get_variable_scope().name
        summary_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.SUMMARIES)
        sumaries = [s for s in summary_ops if "policy_net" in s.name or "shared" in s.name]
        sumaries = [s for s in summary_ops if var_scope_name in s.name]
        self.summaries = tf.compat.v1.summary.merge(sumaries)
