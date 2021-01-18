# -*-coding: utf-8 -*-
# 对txt文件进行中文分词
import jieba
import os
#from utils import files_processing

# 字词分割，对整个文件内容进行字词分割
def segment_lines(file_list,segment_out_dir,stopwords=[]):
    for i,file in enumerate(file_list):
        segment_out_name=os.path.join(segment_out_dir,'segment_{}.txt'.format(i))
        with open(file, 'rb') as f:
            document = f.read()
            document_cut = jieba.cut(document)
            sentence_segment=[]
            for word in document_cut:
                if word not in stopwords:
                    sentence_segment.append(word)
            result = ' '.join(sentence_segment)
            result = result.encode('utf-8')
            with open(segment_out_name, 'wb') as f2:
                f2.write(result)

def getFilePathList(file_dir):
    '''
    获取file_dir目录下，所有文本路径，包括子目录文件
    :param file_dir:
    :return:
    '''
    filePath_list = []
    for walk in os.walk(file_dir):
        part_filePath_list = [os.path.join(walk[0], file) for file in walk[2]]
        filePath_list.extend(part_filePath_list)
    return filePath_list


def get_files_list(file_dir,postfix='ALL'):
    '''
    获得file_dir目录下，后缀名为postfix所有文件列表，包括子目录
    :param file_dir:
    :param postfix:
    :return:
    '''
    postfix=postfix.split('.')[-1]
    file_list=[]
    filePath_list = getFilePathList(file_dir)
    if postfix=='ALL':
        file_list=filePath_list
    else:
        for file in filePath_list:
            basename=os.path.basename(file)  # 获得路径下的文件名
            postfix_name=basename.split('.')[-1]
            if postfix_name==postfix:
                file_list.append(file)
    file_list.sort()
    return file_list

# 对source中的txt文件进行分词，输出到segment目录中
file_list=get_files_list('./source', postfix='*.txt')
segment_lines(file_list, './segment')
