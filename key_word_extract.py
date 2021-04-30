# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 15:29:49 2020
关键词提取算法：
TF-IDF
Text-Rank
LSI
LDA
@author: 地三仙
"""
import os
import sys
import jieba
import jieba.posseg as psg
from jieba import analyse
import re
import math
from gensim import corpora, models

# 分词方法，调用jieba接口
def seg_to_list(sentence, pos=False):
    sentence = re.sub('[a-zA-Z0-9]', '', sentence.replace('\n', '')) # 过滤
    if not pos:  # 不进行词性标注的分词方法
        seg_list = jieba.cut(sentence)
    else:
        seg_list = psg.cut(sentence)
    return seg_list


# 停用词表加载方法
def get_stopwords_list():
    path = "C:/Work/NLP/stop_words.txt"
    sw_list = [sw.replace('\n', '').strip() for sw in \
               open(path, encoding = 'utf8').readlines()]  # 去掉换行符 还有去空格
    return sw_list

# 去除干扰词
def word_filter(seg_list, pos=False): # 词性过滤暂不考虑
    sw_list = get_stopwords_list()
    filter_list = []
#    print("********以下为过滤的词*********")
    for word in seg_list: 
        w = word.strip()  # 去掉前后空格 最可怕的是前后
        if len(w) == 0: # 纯空格不要
            continue
        if w in sw_list:
#            print(word)
            continue
        filter_list.append(word)
#    print("********以上为过滤的词*********")
    return filter_list


# 训练idf，单词出现的概率
def train_idf(doc_list):
    idf_dic = {}
    # 总文档数
    doc_num = len(doc_list)  # 迭代器没有len（）
    print("文档篇数：%d" % doc_num)
    word_num = 0 # 统计单词个数
    # 每个词出现的频率
    for doc in doc_list:
        for word in set(doc):
            word_num += 1
            idf_dic[word] = idf_dic.get(word, 0) + 1
#    print(sorted(idf_dic.items(),key = lambda x:x[1],reverse = True)[:20])

    # 计算idf值,频数加1平滑
    for k, v in idf_dic.items():  # 一定要转换成元组，否则不能返回两个元素
        idf_dic[k] = math.log(doc_num / (1.0+v))

    # 对于字典里没有的词，默认其仅出现在1篇文档
    default_idf = math.log(doc_num / (1.0))
    return idf_dic, default_idf


# 定义文档处理类
"""
1.遍历文件目录，找到预料文件
2.分词
3.过滤干扰词
4.计算IDF:词语的篇章比例
5.计算TF:词语在文章中的比例
"""
class MyDocuments(object):    # memory efficient data streaming
    def __init__(self, dirname):
        self.dirname = dirname
        if not os.path.isdir(dirname): # 如果不是目录退出程序
            print(dirname, '- not a directory!')
            sys.exit()
#    def __iter__(self): # 实现了迭代器协议的对象 迭代器数据不能重用
#        for dirfile in os.walk(self.dirname):
#            cnt = 0     # 控制：每个主题只读10篇
#            for fname in dirfile[2]:
#                if cnt > 1:
#                    break
#                text = open(os.path.join(dirfile[0], fname), 
#                        'r', encoding='utf-8').read()
#                cnt += 1
#                print(fname)
#                seg_list = seg_to_list(text)
#                filter_list = word_filter(seg_list)
#                yield filter_list
    
    def load_data(self): # 实现了迭代器协议的对象
        # 对文件数据进行处理，每篇章仅保留非干扰词
        doc_list = []
        for dirfile in os.walk(self.dirname):
            cnt = 0     # 控制：每个主题只读n篇   ????到底几篇文章？？？
            for fname in dirfile[2]:
                if cnt > 100 or cnt >= len(dirfile[2]):
                    break
                text = open(os.path.join(dirfile[0], fname), 
                        'r', encoding='utf-8').read()
                cnt += 1
#                print(fname)
                seg_list = seg_to_list(text)
                filter_list = word_filter(seg_list)
                doc_list.append(filter_list)
        return doc_list
    
# TF-IDF类
class TfIdf(object):
    # 四个参数分别为训练好的idf字典，默认idf值，word_list, 处理后的待提取文本，关键词数量
    def __init__(self, idf_dic, default_idf, word_list, keyword_num):
        self.word_list = word_list
        self.idf_dic = idf_dic
        self.default_idf = default_idf
        self.keyword_num = keyword_num
        self.tf_dic = self.get_tf_dic()
    # 统计tf值
    def get_tf_dic(self):
        tf_dic = {}
        # 样本单词个数
        words_num = len(self.word_list)
        # 样本单词频率
        for word in self.word_list:
            tf_dic[word] = tf_dic.get(word, 0) + 1
        #  计算tf值
        for k, v in tf_dic.items():
            tf_dic[k] = v / words_num
            
        return tf_dic
    
    def get_tfidf(self):    
        # 计算tf-idf：tf*idf,词语不在idf词典中用默认值
        tfidf_dic = {}
        for word in self.tf_dic.keys():
            idf = self.idf_dic.get(word, self.default_idf)
            tfidf_dic[word] = self.tf_dic.get(word, 0) * idf
        # 输出tf-idf排名前keyword_num的词作为关键词
        for k, v in sorted(tfidf_dic.items(), key = lambda x:x[1],
                           reverse = True)[ : self.keyword_num]:  # 是否倒序 默认False 顺序
            print(k + "/",  end=' ')
        print()

# 主题模型
class TopicModel(object):
    # 传入四个参数：处理后的文本， 关键词数量，主题模型（LSI、LDA）,主题数量
    def __init__(self, doc_list, keywords_num, model='LSI',num_topics=10):
        # 先构建词空间
        self.dictionary = corpora.Dictionary(doc_list)
        # 使用BOW模型向量化
        corpus = [self.dictionary.doc2bow(doc) for doc in doc_list]
        # 对每个词根据tf-idf进行加权，得到加权后的向量表示
        self.tfidf_model = models.TfidfModel(corpus)
        self.corpus_tfidf = self.tfidf_model[corpus]  # 中括号吗？
        self.keywords_num = keywords_num  
        self.num_topics = num_topics

        # 选择加载的模型
        if model == 'LSI':
            self.model = self.train_lsi()
        else:
            self.model = self.train_lda()  
            
        # 生成主题-词映射
        word_dic = self.word_dictionary(doc_list)
        self.wordtopic_dic = self.get_wordtopic(word_dic)
     
    def train_lsi(self):
        lsi = models.LsiModel(self.corpus_tfidf, id2word=self.dictionary,
                              num_topics=self.num_topics)
        return lsi
              
    def train_lda(self): 
        lda = models.LdaModel(self.corpus_tfidf, id2word=self.dictionary,
                              num_topics=self.num_topics)
        return lda
            
    def word_dictionary(self, doc_list):
        word_dictionary = {}
        for doc in doc_list:
            for word in doc:
                word_dictionary[word] = word_dictionary.get(word, 0) + 1
        return word_dictionary


    def get_wordtopic(self, word_dic):
        wordtopic_dic = {}
        for word in word_dic:
            single_list = [word]
            word_bow = self.dictionary.doc2bow(single_list)
            # 对每个词根据tf-idf进行加权，得到加权后的向量表示
            word_corpus = self.tfidf_model[word_bow]
            wordtopic = self.model[word_corpus]
            wordtopic_dic[word] = wordtopic
        return wordtopic_dic

    # 计算词的分布和文档的分布的相似度，取相似度最高的keyword_num个词作为关键词
    def get_simword(self, word_list):
        sentcorpus = self.tfidf_model[self.dictionary.doc2bow(word_list)]
        senttopic = self.model[sentcorpus]
        # 计算余弦相似度
        def calsim(l1, l2):
            a,b,c = 0.0, 0.0 ,0.0
            for t1, t2 in zip(l1, l2):
                x1 = t1[1]
                x2 = t2[1]
                a += x1 * x2
                b += x1 * x1
                c += x2 * x2
            sim = a / math.sqrt(b * c) if not (b * c)==0.0 else 0.0
            return sim
    
        # 计算输入文本和每个词的主题分布相似度
        sim_dic = {}
        for k, v in self.wordtopic_dic.items():
            if k not in word_list:
                continue
            sim = calsim(v, senttopic)
            sim_dic[k] = sim
        
        # 输出相似度最高的前keyword_num个词
        for k, v in sorted(sim_dic.items(),key=lambda x:x[1],reverse=True
                           )[ :self.keywords_num]:
            print(k + ':', '{:.2f}'.format(v), end='/')
        print()
        
# 模型接口封装：     
        
def tfidf_extract(word_list, pos=False, keywordn_num = 10):
    inputdir = "THUCNews/THUCNews"  # 训练集文章目录
    doc_list = MyDocuments(inputdir).load_data() # 第二层为文章词数组
    print("训练样本 %d 篇！"  % len(doc_list))
    idf_dic, default_idf = train_idf(doc_list)  
    print(sorted(idf_dic.items(),key = lambda x:x[1],reverse = False)[ :10])
    tfidf_model = TfIdf(idf_dic, default_idf,word_list,keywordn_num)
    tfidf_model.get_tfidf()      
 
def textrank_extract(text, keyword_num=10):
    textrank = analyse.textrank
    keywords = textrank(text, keyword_num)
    # 输出抽取的关键词
    for w in keywords:
        print(w + '/', end=' ')
    print()
    
def topic_extract(word_list, model, keyword_num=10):
    inputdir = "THUCNews/THUCNews"  # 训练集文章目录
    doc_list = MyDocuments(inputdir).load_data() # 第二层为文章词数组
    print("训练样本 %d 篇！"  % len(doc_list))
    tm = TopicModel(doc_list, 30, model=model)
    tm.get_simword(word_list)
       
if __name__ == '__main__':
    flag = 1
    # 测试一
    if flag == 1:       
        text = open("THUCNews/THUCNews/财经/836035.txt", encoding='utf8').read()
        seg_list = seg_to_list(text)
        filter_list = word_filter(seg_list)
        print("TF-IDF模型结果：")
        tfidf_extract(filter_list)
#        
        print("TextRank模型结果：")
        textrank_extract(text)
        
        print("LSI模型结果：")
        topic_extract(filter_list, model='LSI')
        
        print("LDA模型结果：")
        topic_extract(filter_list, model='LDA')
        
    else:
        inputdir = "THUCNews/THUCNews/财经/" 
        doc_list = MyDocuments(inputdir).load_data() 
        print(len(doc_list))
        for doc in doc_list:
            print("*" * 20, len(doc))
            print(doc[ :5])
            
        """构建词频矩阵，训练LDA模型"""
        dictionary = corpora.Dictionary(doc_list)
         
        # corpus[0]: [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1),...]
        # corpus是把每条新闻ID化后的结果，每个元素是新闻中的每个词语，在字典中的ID和频率
        corpus = [dictionary.doc2bow(doc) for doc in doc_list]
  
        lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=10)
        topic_list = lda.print_topics(10)
        print("5个主题的单词分布为：\n")
        len(lda.show_topics(0))
        lda.show_topics(4)
        for topic in topic_list:
            print(topic)

        tfidf_model = models.TfidfModel(corpus)
        
            
        corpus_tfidf = tfidf_model[corpus]  # 每个文章对应的词语以及ifidf值
        print(len(corpus_tfidf))
        for c_tf in corpus_tfidf:
            print(c_tf)
        
        # 词到主题的映射
        sigal_list = ['库存']
        wordcorpus = tfidf_model[dictionary.doc2bow(sigal_list)]
        wordtopic = lda[wordcorpus]
        
        wordtopic_dic = {}
        
        tm = TopicModel(doc_list, 10)
        word_dic = tm.word_dictionary(doc_list)
        tm.get_wordtopic(word_dic)
            
            
#        text = open("THUCNews/THUCNews/财经/836035.txt", encoding='utf8').read()
#        seg_list = seg_to_list(text)
#        filter_list = word_filter(seg_list)
#        
#        tfidf_model = models.TfidfModel(corpus)
#        corpus_tfidf = tfidf_model[corpus] 
#        
#        lda = models.LdaModel(corpus=corpus_tfidf, id2word=dictionary, num_topics=10)
#        topic_list = lda.print_topics(5)
#        print("5个主题的单词分布为：\n")
#        len(lda.show_topics(0))
#        lda.show_topics(4)
#        for topic in topic_list:
#            print(topic)
#        
#        sentcorpus = tfidf_model[dictionary.doc2bow(filter_list)]
#        senttopic = lda[sentcorpus]
        
        
        


        



