import re
import os
import gensim
import random
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

class DataPre:
    def __init__(self):
        self.stop_words = stopwords.words('english')
        self.wnl = WordNetLemmatizer()

    def remove_special_chars(self, _str):
        """
        移除特殊字符
        :arg"""
        # 保留字母和数字
        result = re.sub(r'[^A-Za-z0-9 ]+', '', _str)
        return result

    def word_lemmatizer(self, _str):
        """
        词形还原
        :arg
        """
        # 获取单词的词性
        def get_wordnet_pos(tag):
            if tag.startswith('J'):
                return wordnet.ADJ
            elif tag.startswith('V'):
                return wordnet.VERB
            elif tag.startswith('N'):
                return wordnet.NOUN
            elif tag.startswith('R'):
                return wordnet.ADV
            else:
                return None
        tokens = word_tokenize(_str)  # 分词
        tagged_sent = pos_tag(tokens)  # 获取单词词性
        lemmas_sent = []
        for tag in tagged_sent:
            wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
            lemmas_sent.append(self.wnl.lemmatize(tag[0], pos=wordnet_pos))  # 词形还原
        return lemmas_sent

    def remove_stopwords(self, word_list):
        """
        移除停用词
        :arg
        """
        result = []
        for word in word_list:
            word_lower = word.lower()
            if word_lower not in self.stop_words and len(word_lower) >= 3:
                result.append(word_lower)
        return result

    def pre(self, _str):
        """
        文本预处理主函数
        :arg
        """
        _str_1 = self.remove_special_chars(_str)
        lemmas_sent = self.word_lemmatizer(_str_1)
        result = self.remove_stopwords(lemmas_sent)
        return " ".join(result).strip()

text_pre_op = DataPre()
random.seed(1234)

rating_data = pd.read_csv('./ratings.csv', header=0,
                          names=['userId', 'movieId', 'rating', "timestamp"])
entity_description_data = pd.read_csv('./i2kg_map_description.csv', header=0,
                                      names=['movieId', 'entityId', 'movieTitle', 'entityName', 'description'])
PATTERN1 = re.compile('[^A-Za-z]')
PATTERN2 = re.compile('[ ]{2,}')
WORD_FREQ_THRESHOLD = 1
ENTITY_FREQ_THRESHOLD = 1
MAX_NAME_LENGTH = 10
MAX_DESC_LENGTH = 20
WORD_EMBEDDING_DIM = 50

word2freq = {}
movie2entity = {}
word2index = {}
entity2index = {}
user2index = {}
movie2index = {}
corpus = []
userId_list = []
movieId_list = []
train_set = []
test_set = []


def count_word_and_entity_freq(files):
    """
    Count the frequency of words in entity name and entity description、entity
    :param files: [movie_info_merge]
    :return: None
    """
    for file in files:
        reader = open(file, encoding='utf-8')
        for line in tqdm(reader):
            array = line.strip().split('||')
            movie_id = array[0]
            entity_name = array[3].lower()
            entity_description = array[4].lower()
            entity_id = array[2]
            entity_id = ".".join(entity_id.split('/')[1:])

            if movie_id not in movie2entity:
                movie2entity[movie_id] = entity_id

            # count entity name word frequency
            for s in entity_name.split():
                if s not in word2freq:
                    word2freq[s] = 1
                else:
                    word2freq[s] += 1

            # count entity description frequency
            for s in entity_description.split():
                s = s.lower()
                if s not in word2freq:
                    word2freq[s] = 1
                else:
                    word2freq[s] += 1
            corpus.append(entity_name.split())
            corpus.append(entity_description.split())
        reader.close()


def construct_word2id_and_entity2id():
    """
    Allocate each valid word and entity a unique index (start from 1)
    :return: None
    """
    def read_map(file):
        entity2index_map = {}
        reader = open(file, encoding='utf-8')
        for line in reader:
            array = line.split('\t')
            if len(array) != 2:  # to skip the first line in entity2id.txt
                continue
            entity_id = array[0]
            index = int(array[1])
            entity2index_map[entity_id] = index
        reader.close()
        return entity2index_map
    # writer = open('./word2index.txt', 'w', encoding='utf-8')
    # cnt = 1  # 0 is for dummy word
    # for w, freq in tqdm(word2freq.items()):
    #     if freq >= WORD_FREQ_THRESHOLD:
    #         word2index[w] = cnt
    #         writer.write('%s\t%s\n' % (str(cnt), w))
    #         cnt += 1
    # print('- word size: %d' % len(word2index))
    # writer.close()
    writer = open('./movie2entity.txt', 'w', encoding='utf-8')
    writer2 = open('./entity2index.txt', 'w', encoding='utf-8')
    book2index_dict = read_map("./movie2index.txt")
    user2index_dict = read_map("./user2index.txt")
    cnt = 1
    for book_id, entity_id in tqdm(movie2entity.items()):
        writer.write('%s\t%s\n' % (book_id, entity_id))
        entity_id = '<http://rdf.freebase.com/ns/' + entity_id + '>'
        writer2.write('%s\t%d\t%d\n' % (entity_id, cnt, book2index_dict[book_id]))  # for later use
        cnt += 1
    for user_id in tqdm(user2index_dict.keys()):
        writer2.write('%s\t%d\t%d\n' % (user_id, cnt, user2index_dict[user_id]))
        cnt += 1
    writer.close()
    writer2.close()
    print('- entity size: %d' % len(entity2index))


def construct_user2id_and_item2id():
    """
    Allocate each valid user and item a unique Index (start from 0)
    :param file: rating_data file
    :return: None
    """
    userId_list = list(set(rating_data['userId']))
    userId_list.sort()
    movieId_list = list(set(rating_data['movieId']))
    movieId_list.sort()
    for name in ['user', 'movie']:
        if name == 'user':
            writer = open("./user2index.txt", "w")
            for i, userId in enumerate(userId_list):
                if userId not in user2index:
                    user2index[userId] = i
                writer.write("{}\t{}\n".format(userId, i))
        else:
            writer = open("./movie2index.txt", "w")
            for i, movieId in enumerate(movieId_list):
                if movieId not in movie2index:
                    movie2index[movieId] = i
                writer.write("{}\t{}\n".format(movieId, i))
        writer.close()
    print('- userId size: %d' % len(userId_list))
    print('- movieId size: %d' % len(movieId_list))

def construct_movie2entity_index():
    """
    map movieId to entity and entity index (embedding)
    :return:
    """

    def read_map(file):
        entity2index_map = {}
        reader = open(file, encoding='utf-8')
        for line in reader:
            array = line.split('\t')
            if len(array) != 2:  # to skip the first line in entity2id.txt
                continue
            entity_id = array[0]
            index = int(array[1])
            entity2index_map[entity_id] = index
        reader.close()
        return entity2index_map
    reader = open("./movie2entity.txt", "r")
    writer = open("./movie2entity_index.txt", 'w', encoding='utf-8')
    entity2index_dict = read_map("../kg/MovieLens_entity2id.txt")
    for line in reader:
        movie_id = line.split()[0]
        entity_id = '<http://rdf.freebase.com/ns/' + line.split()[1] + '>'
        if entity_id in entity2index_dict:
            writer.write("{}\t{}\t{}\n".format(movie_id, entity_id, entity2index_dict[entity_id]))
    reader.close()
    writer.close()

def construct_ml2fb():
    """
    construct movieId to freebase entity mapping
    :return:
    """
    writer = open("../linkage/ml2fb.txt", "w")
    for i in tqdm(range(len(entity_description_data))):
        movie_id = entity_description_data.loc[i, 'movieId']
        entity_id = ".".join(entity_description_data.loc[i, 'entityId'].split('/')[1:])
        writer.write("{}\t{}\n".format(movie_id, entity_id))
    writer.close()

def train_and_test_split():
    """
    split train and test data
    :return:
    """
    rating_list = []
    for i in tqdm(range(len(rating_data))):
        user_id = rating_data.loc[i, 'userId']
        movie_id = rating_data.loc[i, 'movieId']
        rating = int(rating_data.loc[i, 'rating'])
        rating_list.append([user_id, movie_id, rating])
    pos_list = rating_list
    # for i in tqdm(range(len(rating_list))):
    #     pos_list.append([rating_list[i][0], rating_list[i][1], rating_list[i][2]])
    random.shuffle(pos_list)
    train_set = pos_list[:int(0.8 * len(pos_list))]
    test_set = pos_list[int(0.8 * len(pos_list)):len(pos_list)]
    print(len(train_set))
    temp_1 = []
    temp_2 = []
    temp_3 = []
    for user_id, book_id, rating in tqdm(train_set):
        temp_1.append(user_id)
        temp_2.append(book_id)
        temp_3.append(rating)
    train_df = pd.DataFrame({
        "userId": temp_1,
        "itemId": temp_2,
        "rating": temp_3
    })
    train_df.to_csv("./train.csv", index=False)
    with open('./train.txt', "w") as f:
        for i in tqdm(range(len(train_set))):
            f.write("{}\t{}\t{}\n".format(
                train_set[i][0],
                train_set[i][1],
                float(train_set[i][2])
            ))
            # if train_set[i][2] >= 3:
            #     f.write("{}\t{}\t1\n".format(
            #         train_set[i][0],
            #         train_set[i][1]))
            # else:
            #     f.write("{}\t{}\t0\n".format(
            #         train_set[i][0],
            #         train_set[i][1]))
    with open('./test.txt', "w") as f:
        for i in tqdm(range(len(test_set))):
            f.write("{}\t{}\t{}\n".format(
                test_set[i][0],
                test_set[i][1],
                float(test_set[i][2])
            ))
            # if test_set[i][2] >=3:
            #     f.write("{}\t{}\t1\n".format(
            #         test_set[i][0],
            #         test_set[i][1]))
            # else:
            #     f.write("{}\t{}\t0\n".format(
            #         test_set[i][0],
            #         test_set[i][1]))



def get_history_data():
    """
    each user preserve the top-k interacted item
    :return:
    """
    def read_map(file):
        entity2index_map = {}
        reader = open(file, encoding='utf-8')
        for line in reader:
            array = line.split('\t')
            if len(array) != 2:  # to skip the first line in entity2id.txt
                continue
            entity_id = int(array[0])
            index = int(array[1])
            entity2index_map[entity_id] = index
        reader.close()
        return entity2index_map
    k = 5
    user_set = set(rating_data['userId'])
    writer = open("./u_read_list.txt", "w")
    movie2index = read_map("./movie2index.txt")
    user2index = read_map("./user2index.txt")
    for user_id in tqdm(user_set):
        df_user = rating_data[rating_data['userId'] == user_id]
        df_user = df_user.sort_values(axis=0, ascending=True, by='rating')
        item_list = df_user['movieId'].tolist()
        item_list = list(map(lambda x: str(movie2index[x]), item_list))
        writer.write("{} {}\n".format(user2index[user_id], ",".join(item_list)))
    print("--user set length: {}".format(len(user_set)))


def merge_data():
    """
    merge movie info and entity, entity description info
    :return:
    """

    # def string_pre(text):
    #     """
    #     string preprocess, keep English, digital and space
    #     :param text:
    #     :return: preprocessed text
    #     """
    #     comp = re.compile('[^A-Z^a-z^0-9]')
    #     return " ".join(comp.sub(' ', text).strip().split())

    movie_set = set(rating_data['movieId'])
    entity_description_data_set_index = entity_description_data.set_index('movieId')
    writer = open('./movie_info_merge.txt', "w")
    for movie_id in tqdm(movie_set):
        entity_id = entity_description_data_set_index.loc[movie_id, 'entityId']
        movie_title = entity_description_data_set_index.loc[movie_id, 'movieTitle']
        entity_name = text_pre_op.pre(entity_description_data_set_index.loc[movie_id, 'entityName'])
        entity_description = text_pre_op.pre(entity_description_data_set_index.loc[movie_id, 'description'])
        writer.write("{}||{}||{}||{}||{}\n".format(movie_id, movie_title,
                                                   entity_id, entity_name, entity_description))
    writer.close()
    print("--movie_set length: {}".format(len(train_set)))


# def get_local_word2entity(entities):
#     """
#     Given the entities information in one line of the dataset, construct a map from word to entity index
#     E.g., given entities = 'id_1:Harry Potter;id_2:England', return a map = {'harry':index_of(id_1),
#     'potter':index_of(id_1), 'england': index_of(id_2)}
#     :param entities: entities information in one line of the dataset
#     :return: a local map from word to entity index
#     """
#     local_map = {}
#
#     for entity_pair in entities.split(';'):
#         entity_id = entity_pair[:entity_pair.index(':')]
#         entity_name = entity_pair[entity_pair.index(':') + 1:]
#
#         # remove non-character word and transform words to lower case
#         entity_name = PATTERN1.sub(' ', entity_name)
#         entity_name = PATTERN2.sub(' ', entity_name).lower()
#
#         # constructing map: word -> entity_index
#         for w in entity_name.split(' '):
#             entity_index = entity2index[entity_id]
#             local_map[w] = entity_index
#
#     return local_map


def encoding_title(entity_name, entity_description):
    """
    Encoding a title according to word2index map and entity2index map
    :param entity_name: movie entity name text
    :param entity_description: movie entity description text
    :return: encodings of the title with respect to word and entity, respectively
    """

    array = entity_name.split()
    desc_array = entity_description.split()
    # name_encoding = ['0'] * MAX_NAME_LENGTH
    name_encoding = []
    # description_encoding = ['0'] * MAX_DESC_LENGTH
    description_encoding = []
    word2index = {}
    with open("word2index.txt", "r") as f:
        for line in f:
            content = line.split()
            word2index[content[1]] = int(content[0])
    point = 0
    for s in array:
        if s in word2index:
            # name_encoding[point] = str(word2index[s])
            name_encoding.append(str(word2index[s]))
            point += 1
        if point == MAX_NAME_LENGTH:
            break
    point = 0
    for s in desc_array:
        if s in word2index:
            # description_encoding[point] = str(word2index[s])
            description_encoding.append(str(word2index[s]))
            point += 1
        if point == MAX_DESC_LENGTH:
            break
    name_encoding = ','.join(name_encoding)
    description_encoding = ','.join(description_encoding)
    return name_encoding, description_encoding


def transform(input_file, output_file):
    reader = open(input_file, encoding='utf-8')
    lines = reader.readlines()
    writer = open(output_file, 'w', encoding='utf-8')
    for line in tqdm(lines):
        array = line.strip().split('||')
        movie_id = array[0]
        entity_name = array[3].lower()
        entity_description = array[4].lower()
        name_encoding, description_encoding = encoding_title(entity_name, entity_description)
        writer.write('%s\t%s\t%s\n' % (movie_id, name_encoding, description_encoding))
    reader.close()
    writer.close()


def get_word2vec_model():
    # word2vec = gensim.models.KeyedVectors.load_word2vec_format(
    #     "/Users/wsc/Downloads/GoogleNews-vectors-negative300.bin.gz", binary=True)
    if not os.path.exists('word_embeddings_' + str(WORD_EMBEDDING_DIM) + '.model'):
        print('- training word2vec model...')
        w2v_model = gensim.models.Word2Vec(corpus, size=WORD_EMBEDDING_DIM, min_count=1, workers=16)
        print('- saving model ...')
        w2v_model.save('word_embeddings_' + str(WORD_EMBEDDING_DIM) + '.model')
    else:
        print('- loading model ...')
        w2v_model = gensim.models.word2vec.Word2Vec.load('word_embeddings_' + str(WORD_EMBEDDING_DIM) + '.model')
    return w2v_model


if __name__ == '__main__':
    # print("construct user2index and movie2index ...")
    # construct_user2id_and_item2id()
    # #
    # print('splitting train and test data ...')
    # train_and_test_split()
    # #
    # print('merging movie info and entity、entity description info ...')
    # merge_data()
    #
    # print("for each user, get his top-k rating movie ...")
    # get_history_data()

    # print('construct movie id to freebase entity id mapping')
    # construct_ml2fb()

    # print('Count the frequency of words in entity name and entity description、entity ...')
    # count_word_and_entity_freq(['movie_info_merge.txt'])
    # #
    # print('constructing word2id map and entity to id map ...')
    # construct_word2id_and_entity2id()
    # #
    print('transforming training and test dataset ...')
    transform('movie_info_merge.txt', 'movie2name_description_map.txt')
    #
    # #
    # print('getting word embeddings ...')
    # word2index = {}
    # with open("word2index.txt", "r") as f:
    #     for line in f:
    #         content = line.split()
    #         word2index[content[1]] = int(content[0])
    # # word2index 索引从1开始且embeddings维度为len(word2index)+1，是为了使padding值 (0) 索引出的嵌入向量全为0
    # embeddings = np.zeros([len(word2index) + 1, WORD_EMBEDDING_DIM])
    # model = get_word2vec_model()
    # have_embedding = 0
    # all_word = 0
    # for word in tqdm(word2index.keys()):
    #     all_word += 1
    #     if word in model.wv.vocab:
    #         have_embedding += 1
    #     embedding = model[word] if word in model.wv.vocab else np.zeros(WORD_EMBEDDING_DIM)
    #     # print(embedding)
    #     index = word2index[word]
    #     embeddings[index] = embedding
    # print('- writing word embeddings ...')
    # print('having embedding: {}, all word num: {}, ratio: {}%'.format(
    #     have_embedding, all_word, (have_embedding / all_word) * 100))
    # np.save(('trained_word_embeddings_' + str(WORD_EMBEDDING_DIM)), embeddings)


    # print("construct_movie2entity_index")
    # construct_movie2entity_index()