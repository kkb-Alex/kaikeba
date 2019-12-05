from gensim.models import word2vec
import numpy as np
from sklearn.decomposition import PCA
import re
import jieba


class SummaryModel:
    def __init__(self):
        # 读取训练完的模型
        self.model = word2vec.Word2Vec.load('model_191115_1')  
        # 计算模型一共有多少词，用于计算词频
        self.total = 0
        for k in self.model.wv.vocab.keys():
            self.total += self.model.wv.vocab[k].count

    # 计算词频
    def get_fre(self, w):
        return self.model.wv.vocab[w].count / self.total

    # SIF模型第一步
    def sif_s1(self, s: list, a) -> np.array:
        v = np.zeros(self.model.wv['算法'].shape)
        count = 0
        for w in s:
            if w not in self.model.wv: continue
            we_w = a / (a + self.get_fre(w)) * self.model.wv[w]
            v += we_w
            count += 1
        if count > 0:
            return v / count
        else:
            return v

    # SIF模型第二部
    @staticmethod
    def sif_s2(s: np.array) -> np.array:
        pca = PCA(n_components=1)
        pca.fit(s)
        pc = pca.components_
        return s - s.dot(pc.T) * pc

    # 用余弦相似度计算句向量之间的关系
    @staticmethod
    def relative(sentences, title, article, weight):
        def cosine_v(v1, v2):
            num = np.dot(v1, v2.T)
            demon = np.linalg.norm(v1) * np.linalg.norm(v2)
            return num / demon

        c = []
        for i in range(sentences.shape[0]):
            s = sentences[i, :]
            c_i = weight * cosine_v(s, article.T) + (1 - weight) * cosine_v(s, title.T)
            c.append(c_i)
        return c

    # 句子分割
    @staticmethod
    def cut_content(content, cut_point='。；！？', spliter='@%#'):
        # 插入分割符
        new_content = ''
        i = 0
        while i < len(content):
            new_content += content[i]
            if content[i] in cut_point:
                if i < len(content) - 1 and content[i + 1] == '”':
                    i += 1
                    new_content += content[i]
                new_content += spliter
            i += 1

        # 分割
        sentences = re.split('[{}]'.format(spliter), new_content)

        # 筛选长度大于5的句子。经过测试，新闻库里有很多图片的名字，并不是正文但是对模型结果有很大的影响，需要去除
        new_sentences = []
        for s in sentences:
            s = re.sub('\\\\n|[\n\u3000\r]', '', s)
            s = s.strip()
            if len(s) > 5:
                new_sentences.append(s)

        return new_sentences

    # 输入标题和内容，计算相似度后进行KNN平滑
    def get_c(self, title, sentences, sif_a, weight=0.5):
        punctuation = '！？｡。，＂＃＄％＆＇（）＊＋－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘\'‛“”„‟…‧﹏' + '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~·ʔ•'
        article = []

        def sentence_to_words(s):
            s = re.sub('[{}]'.format(punctuation + '\n'), ' ', s)
            words = jieba.cut(s)
            return [w for w in words if w.strip()]

        title = sentence_to_words(title)
        for i in range(len(sentences)):
            words = sentence_to_words(sentences[i])
            article += words
            sentences[i] = [sentences[i], words]
        title_v = self.sif_s1(title, sif_a)
        article_v = self.sif_s1(article, sif_a)
        sentences_v = np.zeros([len(sentences), 250])
        for i in range(len(sentences)):
            sentences_v[i, :] = self.sif_s1(sentences[i][1], sif_a)
        sentences_v = self.sif_s2(sentences_v)

        c = self.relative(sentences_v, title_v, article_v, weight)
        # KNN平滑
        knn_c = []

        def left_(i):
            if i <= 2:
                return 0
            else:
                return i - 2

        for i in range(len(c)):
            c_i = sum(c[left_(i): i + 3]) / len(c[left_(i): i + 3])
            knn_c.append(c_i)

        for i in range(len(knn_c)):
            sentences[i] = [sentences[i][0], knn_c[i], i]

        return sentences

    # 输出摘要
    def get_summary(self, title, content, n=5, sif_a=1e-5, weight=0.4):
        # 切割句子
        sentences = self.cut_content(content)
        # 获得句子相关度
        sentences_c = self.get_c(title, sentences, sif_a, weight)
        # 输入最相关的5个句子，并按原文顺序排列
        topn = [s[0] for s in sorted([s for s in sorted(sentences_c, key=lambda x: x[1], reverse=True)][:n], key=lambda x:x[2])]
        return ''.join(topn)