import pandas as pd

# 读取 NSL-KDD 数据集
data = pd.read_csv("../../datasets/NSL/KDDTrain+.txt", header=None)

# 列名
column_names = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "attack_type", "difficulty_level"]

# 设置列名
data.columns = column_names

# 指定的攻击类型列表
specified_attacks = ['ftp_write', 'guess_passwd', 'httptunnel', 'imap', 'multihop', 'named', 'phf','sendmail','Snmpgetattack','spy','snmpguess','warezclient','warezmaster','xlock','xsnoop']

# 打印指定的攻击类型数据
specified_attacks_data = data[data['attack_type'].isin(specified_attacks)]
print(specified_attacks_data)

# 统计指定攻击类型的数据总数
total_attack_count = specified_attacks_data.shape[0]
print("\nTotal Number of R2L Attacks:", total_attack_count)