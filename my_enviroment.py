'''
    环境定义
    Date:2024/02
    Author:刘韦豪
'''

import numpy as np
from data_preprocessing import data_cls

class my_env(data_cls):
    '''
    设置环境属性
    '''
    def __init__(self,train_test,**kwargs):
        data_cls.__init__(self,train_test,**kwargs)
        self.data_shape = self.get_shape() # 数据形状
        self.batch_size = kwargs.get('batch_size',1) # 经验回放-> 批处理 = 1
        self.fails_episode = kwargs.get('fails_episode',10) # 失败阈值
        self.action_space = len(self.attack_types) # 动作空间的大小
        self.observation_space = self.data_shape[1]-self.action_space # 观察空间大小
        self.counter = 0 # 计数器

    def _update_state(self):
        self.states,self.labels = self.get_batch(self.batch_size)
        
    def reset(self):
        '''
        重置环境的状态
        '''
        #self.states,self.labels = data_cls.get_sequential_batch(self,self.batch_size)
        self.states,self.labels = data_cls.get_batch(self,self.batch_size)
        self.counter = 0
        
        return self.states
     
    def step(self,actions):
        '''
        接受动作作为参数，并返回新的状态、奖励以及环境是否完成的标志。
        它首先清除之前的奖励，然后根据动作是否与标签匹配来设置新的奖励。
        接着更新状态，并根据计数器是否超过失败阈值来设置完成标志。
        '''
        # 清除以前的奖励      
        self.reward = 0
        
        # 实现新的奖励
        if actions == np.argmax(self.labels.values):
            self.reward = 1

        else: 
            self.counter += 1
            if (np.argmax(self.labels.values)!=0)&(actions!=0): # 一定被攻击
                self.reward = 0.5

        # 获取新的状态
        self._update_state()
        
        if self.counter >= self.fails_episode:
            self.done = True
            
        else:
            self.done = False
            
        return self.states, self.reward, self.done
    
