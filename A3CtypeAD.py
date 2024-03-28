'''
    异常检测文件
    Date:2024/03
    Author:刘韦豪
'''

# 基于Denny Britz提出的A3C算法
import tensorflow as tf
import threading
import multiprocessing
import os
import shutil
import itertools
from my_enviroment import my_env
from estimators import ValueEstimator, PolicyEstimator
from policy_monitor import PolicyMonitor
from worker import Worker

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)

del_all_flags(tf.flags.FLAGS)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

tf.flags.DEFINE_string("model_dir", "tmp/a3c", "用于编写 Tensorboard 摘要和记录的目录。")
tf.flags.DEFINE_integer("t_max", 2, "执行更新前的步骤数")
tf.flags.DEFINE_integer("max_global_steps", 500000, "在环境中执行这些步骤后停止训练。默认则无限期运行。")
tf.flags.DEFINE_integer("eval_every", 180, "每 N 秒评估一次策略")
tf.flags.DEFINE_boolean("reset", True, "如果设置，请删除现有模型目录并从头开始训练。")
tf.flags.DEFINE_integer("parallelism", None, "要运行的线程数。如果未设置，我们将运行 [num_cpu_cores] 线程。")

FLAGS = tf.flags.FLAGS

# 环境的初始化
def make_env():
    kdd_train = '../../datasets/NSL/KDDTrain+.txt'
    kdd_test = '../../datasets/NSL/KDDTest+.txt'
    
    formated_train_path = "../../datasets/formated/formated_train_type.data"
    formated_test_path = "../../datasets/formated/formated_test_type.data"
    batch_size = 1
    fails_episode = 10 # 回合中的失败次数
    
    env = my_env('train',train_path=kdd_train,test_path=kdd_test,
                formated_train_path = formated_train_path,
                formated_test_path = formated_test_path,
                batch_size=batch_size,
                fails_episode=fails_episode)
    return env

env_ = make_env()
VALID_ACTIONS = list(range(env_.action_space))

# 设置工作成员数量
NUM_WORKERS = multiprocessing.cpu_count()
if FLAGS.parallelism:
  NUM_WORKERS = FLAGS.parallelism

MODEL_DIR = FLAGS.model_dir
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")

# 可选为空模型目录
if FLAGS.reset:
  shutil.rmtree(MODEL_DIR, ignore_errors=True)

if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

summary_writer = tf.compat.v1.summary.FileWriter(os.path.join(MODEL_DIR, "train"))

with tf.device("/cpu:0"):
    
    # 跟踪执行的更新次数
    global_step = tf.Variable(0, name="global_step", trainable=False)
    
    # 全局策略和价值网络
    with tf.compat.v1.variable_scope("global") as vs:
        policy_net = PolicyEstimator(num_outputs=len(VALID_ACTIONS),
                                     observation_space=env_.observation_space)
        value_net = ValueEstimator(observation_space=env_.observation_space,
                                   reuse=True)
    # 全局步长迭代器
    global_counter = itertools.count()

    # 创建工作线程图
    workers = []
    for worker_id in range(NUM_WORKERS):
        # 只在其中一个worker上写摘要，因为所有worker都是一样的
        worker_summary_writer = None
        if worker_id == 0:
            worker_summary_writer = summary_writer
            
        worker = Worker(name="worker_{}".format(worker_id),
                      env=make_env(),
                      policy_net=policy_net,
                      value_net=value_net,
                      global_counter=global_counter,
                      discount_factor = 0.005,
                      summary_writer=worker_summary_writer,
                      max_global_steps=FLAGS.max_global_steps)
        workers.append(worker)

    saver = tf.compat.v1.train.Saver(keep_checkpoint_every_n_hours=0.5, max_to_keep=10)

    # 用于间断为策略网络保存记录并将回合奖励写入Tensorboard
    pe = PolicyMonitor(
            env=make_env(),
            policy_net=policy_net,
            summary_writer=summary_writer,
            saver=saver)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    coord = tf.train.Coordinator()

    # 加载上一个检查点（如果存在）
    latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if latest_checkpoint:
        print("加载模型检查点: {}".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)
        
    # 启动工作线程
    worker_threads = []
    for worker in workers:
        worker_fn = lambda worker=worker: worker.run(sess, coord, FLAGS.t_max)
        t = threading.Thread(target=worker_fn)
        t.start()
        worker_threads.append(t)

    # 为策略评估任务启动线程
    monitor_thread = threading.Thread(target=lambda: pe.continuous_eval(FLAGS.eval_every, sess, coord))
    monitor_thread.start()

    # 等待所有工作完成
    coord.join(worker_threads)
