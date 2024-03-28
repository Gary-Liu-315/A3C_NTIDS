import sys
import os
import itertools
import collections
import numpy as np
import tensorflow as tf

from inspect import getsourcefile
current_path = os.path.dirname(os.path.abspath(getsourcefile(lambda:0)))
import_path = os.path.abspath(os.path.join(current_path, "../.."))

if import_path not in sys.path:
    sys.path.append(import_path)

from estimators import ValueEstimator, PolicyEstimator

Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

def make_copy_params_op(v1_list, v2_list):
    """
    创建一个操作，将v1_list中的变量参数复制到v2_list中的变量上。
    这两个列表中的变量顺序必须完全相同。
    """
    v1_list = list(sorted(v1_list, key=lambda v: v.name))
    v2_list = list(sorted(v2_list, key=lambda v: v.name))

    update_ops = []
    for v1, v2 in zip(v1_list, v2_list):
        op = v2.assign(v1)
        update_ops.append(op)

    return update_ops

def make_train_op(local_estimator, global_estimator):
    """
    创建一个操作，将本地估计器的梯度应用到全局估计器上
    """
    local_grads, _ = zip(*local_estimator.grads_and_vars)
    # 裁剪梯度
    local_grads, _ = tf.compat.v1.clip_by_global_norm(local_grads, 5.0)
    _, global_vars = zip(*global_estimator.grads_and_vars)
    local_global_grads_and_vars = list(zip(local_grads, global_vars))
    return global_estimator.optimizer.apply_gradients(local_global_grads_and_vars,
          global_step=tf.compat.v1.train.get_global_step())

class Worker(object):

    def __init__(self, name, env, policy_net, value_net, global_counter, discount_factor=0.99, summary_writer=None, max_global_steps=None):
        self.name = name
        self.discount_factor = discount_factor
        self.max_global_steps = max_global_steps
        self.global_step = tf.compat.v1.train.get_global_step()
        self.global_policy_net = policy_net
        self.global_value_net = value_net
        self.global_counter = global_counter
        self.local_counter = itertools.count()
        self.summary_writer = summary_writer
        self.env = env

        # 创建不进行异步更新的本地策略/价值网络
        with tf.compat.v1.variable_scope(name):
            self.policy_net = PolicyEstimator(policy_net.num_outputs,policy_net.observation_space)
            self.value_net = ValueEstimator(value_net.observation_space,reuse=True)

        # 从全局策略/价值网络复制参数的操作
        self.copy_params_op = make_copy_params_op(
            tf.contrib.slim.get_variables(scope="global", collection=tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES),
            tf.contrib.slim.get_variables(scope=self.name+'/', collection=tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES))

        self.vnet_train_op = make_train_op(self.value_net, self.global_value_net)
        self.pnet_train_op = make_train_op(self.policy_net, self.global_policy_net)

        self.state = None

    def run(self, sess, coord, t_max):
        with sess.as_default(), sess.graph.as_default():
            # 初始状态
            self.state = self.env.reset()
            try:
                for epoch in range(self.max_global_steps):
                    # 从全局网络复制参数
                    sess.run(self.copy_params_op)

                    # 收集经验
                    transitions, local_t, global_t = self.run_n_steps(t_max, sess)

                    if self.max_global_steps is not None:
                        progress = global_t / self.max_global_steps
                        bar_length = 30
                        filled_length = int(bar_length * progress)
                        bar = '=' * filled_length + '-' * (bar_length - filled_length)
                        #if epoch % 100 == 0:
                         #   print(f'\r[{bar}] {progress * 100:.0f}% Epoch {epoch + 1}/{self.max_global_steps}', end='')

                    if global_t >= self.max_global_steps:
                        tf.logging.info("\nReached global step {}. Stopping.".format(global_t))
                        coord.request_stop()
                        return

                    # 更新全局网络
                    self.update(transitions, sess)

            except tf.compat.v1.errors.CancelledError:
                return

    def _policy_net_predict(self, state, sess):
        feed_dict = { self.policy_net.states: [state]}
        preds = sess.run(self.policy_net.predictions, feed_dict)
        return preds["probs"][0]

    def _value_net_predict(self, state, sess):
        feed_dict = { self.value_net.states: [state] }
        preds = sess.run(self.value_net.predictions, feed_dict)
        return preds["logits"][0]

    def run_n_steps(self, n, sess):
        transitions = []
        for _ in range(n):
            # 进行一步操作
            action_probs = self._policy_net_predict(self.state, sess)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done = self.env.step(action)
            # 存储过渡
            transitions.append(Transition(
                state=self.state, action=action, reward=reward, next_state=next_state, done=done))

            # 增加本地和全局计数器
            local_t = next(self.local_counter)
            global_t = next(self.global_counter)

            if done:
                self.state = self.env.reset()
                break
            else:
                self.state = next_state
        return transitions, local_t, global_t

    def update(self, transitions, sess):
        """
        更新基于收集到的经验的全局策略和值网络

        参数:
          transitions: 一系列经验过渡的列表
          sess: 一个TensorFlow会话
        """

        # 如果该回合数未完成，我们从最后一个状态引导值。
        reward = 0.0
        if not transitions[-1].done:
            reward = self._value_net_predict(transitions[-1].next_state, sess)

        # 累积小批量示例
        states = []
        policy_targets = []
        value_targets = []
        actions = []

        for transition in transitions[::-1]:
            reward = transition.reward + self.discount_factor * reward
            policy_target = (reward - self._value_net_predict(transition.state, sess))
            # 累积更新
            states.append(transition.state)
            actions.append(transition.action)
            policy_targets.append(policy_target)
            value_targets.append(reward)

        feed_dict = {
            self.policy_net.states: np.array(states),
            self.policy_net.targets: policy_targets,
            self.policy_net.actions: actions,
            self.value_net.states: np.array(states),
            self.value_net.targets: value_targets,
        }

        # 使用局部梯度训练全局估计器
        global_step, pnet_loss, vnet_loss, _, _, pnet_summaries, vnet_summaries = sess.run([
            self.global_step,
            self.policy_net.loss,
            self.value_net.loss,
            self.pnet_train_op,
            self.vnet_train_op,
            self.policy_net.summaries,
            self.value_net.summaries
        ], feed_dict)

        # 写入摘要
        if self.summary_writer is not None:
            self.summary_writer.add_summary(pnet_summaries, global_step)
            self.summary_writer.add_summary(vnet_summaries, global_step)
            self.summary_writer.flush()

        return pnet_loss, vnet_loss, pnet_summaries, vnet_summaries
