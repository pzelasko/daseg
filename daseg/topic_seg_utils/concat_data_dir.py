
import sys, os


ip_data_dir = sys.argv[1] # a comma seperated list of data dir to concat
op_data_dir = sys.argv[2]
concat_dev = int(sys.argv[3]) # 1 if you want to concatenate dev and test in all the ip_data_dir, otherwise 0 which copies only first data dir in ip_data_dir


if ',' in ip_data_dir:
    ip_data_dir = ip_data_dir.split(',')
else:
    ip_data_dir = [ip_data_dir]


def concat_tsv(ip_data_dir, split_tsv):
    train_data = ''
    for temp_data_dir in ip_data_dir:
        print(temp_data_dir)
        split_tsv_path = temp_data_dir + '/' + split_tsv
        if os.path.exists(split_tsv_path):
            with open(split_tsv_path) as f:
                train_data += f.read()
            if not train_data.endswith('\n'):
                train_data += '\n'
        else:
            print(f'{split_tsv_path} does not exist so exiting ...')
            sys.exit()
    return train_data

print(ip_data_dir)
train_data = concat_tsv(ip_data_dir, 'train.tsv')

if concat_dev:
    dev_data = concat_tsv(ip_data_dir, 'dev.tsv')
    test_data = concat_tsv(ip_data_dir, 'test.tsv')
    if os.path.exists(ip_data_dir[0]+'/test_challenge.tsv'):
        test_challenge_data = concat_tsv(ip_data_dir, 'test_challenge.tsv')
else:
    dev_data = concat_tsv([ip_data_dir[0]], 'dev.tsv')
    test_data = concat_tsv([ip_data_dir[0]], 'test.tsv')
    if os.path.exists(ip_data_dir[0]+'/test_challenge.tsv'):
        test_challenge_data = concat_tsv([ip_data_dir[0]], 'test_challenge.tsv')

if os.path.exists(ip_data_dir[0]+'feats.scp_h5'):
    feats_scp_h5_data = concat_tsv(ip_data_dir, 'feats.scp_h5')
else:
    feats_scp_h5_data = None

if os.path.exists(ip_data_dir[0]+'utt2csvpath'):
    utt2csvpath_data = concat_tsv(ip_data_dir, 'utt2csvpath')
else:
    utt2csvpath_data = None


os.makedirs(op_data_dir, exist_ok=True)
with open(op_data_dir  + '/train.tsv', 'w') as f:
    f.write(train_data)

with open(op_data_dir  + '/dev.tsv', 'w') as f:
    f.write(dev_data)

with open(op_data_dir  + '/test.tsv', 'w') as f:
    f.write(test_data)

if os.path.exists(ip_data_dir[0]+'/test_challenge.tsv'):
    print('test_challenge.tsv exists in the first ip_data_dir so writing that too')
    with open(op_data_dir  + '/test_challenge.tsv', 'w') as f:
        f.write(test_challenge_data)

if feats_scp_h5_data is not None:
    with open(op_data_dir  + '/feats.scp_h5', 'w') as f:
        f.write(feats_scp_h5_data)

if utt2csvpath_data is not None:
    with open(op_data_dir  + '/utt2csvpath', 'w') as f:
        
        f.write(utt2csvpath_data)


