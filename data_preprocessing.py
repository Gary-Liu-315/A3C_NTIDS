'''
    数据类处理
    Date:2024/02
    Author:刘韦豪
'''
import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

# 用于初始化各种参数和数据路径
class data_cls:
    def __init__(self,train_test,**kwargs):
        col_names = ["duration","protocol_type","service","flag","src_bytes",
            "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
            "logged_in","num_compromised","root_shell","su_attempted","num_root",
            "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
            "is_host_login","is_guest_login","count","srv_count","serror_rate",
            "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
            "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
            "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
            "dst_host_rerror_rate","dst_host_srv_rerror_rate","labels"]
        self.index = 0
        self.loaded = False

        # 数据格式化路径和测试路径
        self.train_test = train_test
        self.train_path = kwargs.get('train_path', '../../datasets/NSL/KDDTrain+.txt')
        self.test_path = kwargs.get('test_path','../../datasets/NSL/KDDTest+.txt')
        
        self.formated_train_path = kwargs.get('formated_train_path', 
                                              "../../datasets/formated/formated_train_type.data")
        self.formated_test_path = kwargs.get('formated_test_path',
                                             "../../datasets/formated/formated_test_type.data")

        # 对异常类型归为五分类——正常、Dos、Probe、R2L、U2R
        self.attack_types = ['normal','DoS','Probe','R2L','U2R']
        self.attack_map =   { 'normal': 'normal',
                        
                        'back': 'DoS','land': 'DoS','neptune': 'DoS','pod': 'DoS',
                        'smurf': 'DoS','teardrop': 'DoS','mailbomb': 'DoS',
                        'apache2': 'DoS','processtable': 'DoS','udpstorm': 'DoS',
                        
                        'ipsweep': 'Probe','nmap': 'Probe','portsweep': 'Probe',
                        'satan': 'Probe','mscan': 'Probe','saint': 'Probe',
                    
                        'ftp_write': 'R2L','guess_passwd': 'R2L','imap': 'R2L',
                        'multihop': 'R2L','phf': 'R2L','spy': 'R2L',
                        'warezclient': 'R2L','warezmaster': 'R2L','sendmail': 
                        'R2L','named': 'R2L','snmpgetattack': 'R2L',
                        'snmpguess': 'R2L','xlock': 'R2L', 'xsnoop': 'R2L','worm': 'R2L',
                        
                        'buffer_overflow': 'U2R','loadmodule': 'U2R','perl': 
                        'U2R','rootkit': 'U2R','httptunnel': 'U2R','ps': 'U2R',    
                        'sqlattack': 'U2R','xterm': 'U2R'
                    }
        
        formated = False     
        
        # 测试格式化数据是否存在
        if os.path.exists(self.formated_train_path) and os.path.exists(self.formated_test_path):
            formated = True
           
        self.formated_dir = "../../datasets/formated/"
        if not os.path.exists(self.formated_dir):
            os.makedirs(self.formated_dir)

        # 如果不存在，需要格式化数据
        if not formated:
            ''' 将数据集格式化为可直接使用的数据 '''
            self.df = pd.read_csv(self.train_path,sep=',',names=col_names,index_col=False)
            if 'dificulty' in self.df.columns:
                self.df.drop('dificulty', axis=1, inplace=True) # 在困难情况下
                
            data2 = pd.read_csv(self.test_path,sep=',',names=col_names,index_col=False)
            if 'dificulty' in data2:
                del(data2['dificulty'])
            train_indx = self.df.shape[0]
            frames = [self.df,data2]
            self.df = pd.concat(frames)
            
            # 数据框架处理 将这几部分的数据类型进行独热编码
            self.df = pd.concat([self.df.drop('protocol_type', axis=1), pd.get_dummies(self.df['protocol_type'])], axis=1)
            self.df = pd.concat([self.df.drop('service', axis=1), pd.get_dummies(self.df['service'])], axis=1)
            self.df = pd.concat([self.df.drop('flag', axis=1), pd.get_dummies(self.df['flag'])], axis=1)
              
            # 如果尝试了 'su root' 命令，则为1；否则为0
            self.df['su_attempted'] = self.df['su_attempted'].replace(2.0, 0.0)
            
            # 针对反应的独热编码（转换为一个长度为n的向量，如可能取值为“正常和异常”，[0,1]表示异常）  
            all_labels = self.df['labels'] # 提取数据框中所有标签
            mapped_labels = np.vectorize(self.attack_map.get)(all_labels) # 攻击的分类和统计
            self.df = self.df.reset_index(drop=True)
            self.df = pd.concat([self.df.drop('labels', axis=1),pd.get_dummies(mapped_labels)], axis=1)
            
            # 对数据进行Min-Max归一化
            self.df = (self.df-self.df.mean())/(self.df.max()-self.df.min())
            for indx,dtype in self.df.dtypes.iteritems():
                if dtype == 'float64' or dtype == 'int64':
                    if self.df[indx].max() == 0 and self.df[indx].min()== 0:
                        self.df[indx] = 0
                    else:
                        self.df[indx] = (self.df[indx]-self.df[indx].min())/(self.df[indx].max()-self.df[indx].min())   
            
             # 保存数据
            test_df = self.df.iloc[train_indx:self.df.shape[0]]
            test_df = shuffle(test_df,random_state=np.random.randint(0,100))
            self.df = self.df[:train_indx]
            self.df = shuffle(self.df,random_state=np.random.randint(0,100))
            test_df.to_csv(self.formated_test_path,sep=',',index=False)
            self.df.to_csv(self.formated_train_path,sep=',',index=False)
            
    ''' 
        从数据集中顺序地获取n行批量数据，并返回数据框df（包含n行数据）
        和标签labels（用于检测的正确标签）。
        这个过程针对最大的数据集进行顺序处理。
    '''
    def get_sequential_batch(self, batch_size=100):
        if self.loaded is False:
            self.df = pd.read_csv(self.formated_path,sep=',', nrows = batch_size)
            self.loaded = True
        else:
            self.df = pd.read_csv(self.formated_path,sep=',', nrows = batch_size,
                         skiprows = self.index)
        
        self.index += batch_size

        labels = self.df[self.attack_types]
        for att in self.attack_types:
            del(self.df[att])
        return self.df,labels
    
    ''' 
        要求从已加载到RAM（随机存取存储器）中的数据中获取n行数据，提高加载速度。
    '''
    def get_batch(self, batch_size=100):
        
        if self.loaded is False:
            self._load_df()
            
        indexes = list(range(self.index,self.index+batch_size))    
        if max(indexes)>self.data_shape[0]-1:
            dif = max(indexes)-self.data_shape[0]
            indexes[len(indexes)-dif-1:len(indexes)] = list(range(dif+1))
            self.index=batch_size-dif
            batch = self.df.iloc[indexes]
        else: 
            batch = self.df.iloc[indexes]
            self.index += batch_size    

        labels = batch[self.attack_types]
        
        for att in self.attack_types:
            del(batch[att])
            
        batch = batch.to_numpy()[0]
        
        return batch,labels
    
    def get_full(self):

        self._load_df()
        
        batch = self.df
        labels = batch[self.attack_types]
        for att in self.attack_types:
            del(batch[att])
        
        return np.array(batch),labels
    
    def get_shape(self):
        if self.loaded is False:
            self._load_df()
        
        self.data_shape = self.df.shape
        # stata + labels
        return self.data_shape
    
    def _load_df(self):
        if self.train_test == 'train':
            self.df = pd.read_csv(self.formated_train_path,sep=',') # Read again the csv
        else:
            self.df = pd.read_csv(self.formated_test_path,sep=',')
        self.index=np.random.randint(0,self.df.shape[0]-1,dtype=np.int32)
        self.loaded = True





