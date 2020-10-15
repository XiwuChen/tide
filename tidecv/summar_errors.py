import pickle
import re
if __name__ == '__main__':
    home_dir = '/home/xwchen/experiments/tide/vis/'
    # 如果不使用这些错误的样本的统计结果，可以去掉这部分，使用vis_bboxes函数
    with open(home_dir + "dump_error", 'rb') as f:
        error_dict = pickle.load(f)

    c = re.compile('\.(\w+)')
    for k,v in error_dict.items():

        name=c.findall(str(k))[-1]


        print('%s: %d'%(name,len(v)))

    print('')