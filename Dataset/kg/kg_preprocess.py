import numpy as np
from tqdm import tqdm

KGE_METHOD = 'TransR'
ENTITY_EMBEDDING_DIM = 50


def read_map(file):
    entity2index_map = {}
    reader = open(file, encoding='utf-8')
    for line in reader:
        array = line.split('\t')
        if len(array) == 1:  # to skip the first line in entity2id.txt
            continue
        entity_id = array[0]
        index = int(array[1])
        entity2index_map[entity_id] = index
    reader.close()
    return entity2index_map


def get_neighbors_for_entity(file):
    reader = open(file, encoding='utf-8')
    entity2neighbor_map = {}
    for line in reader:
        array = line.strip().split('\t')
        if len(array) != 3:  # to skip the first line in triple2id.txt
            continue
        head = int(array[0])
        relation = int(array[2])
        # if (relation == 39):
            # print ("yes")
        tail = int(array[1])
        entity2neighbor_map.setdefault(head, {})
        entity2neighbor_map.setdefault(tail, {})
        if relation in entity2neighbor_map[head]:
            entity2neighbor_map[head][relation].append(tail)
        else:
            entity2neighbor_map[head][relation] = [tail]
        if relation in entity2neighbor_map[tail]:
            entity2neighbor_map[tail][relation].append(head)
        else:
            entity2neighbor_map[tail][relation] = [head]
    reader.close()
    return entity2neighbor_map


if __name__ == '__main__':
    # entity2index.txt (generated by news_preprocess.py) contains all entities appear in the dataset
    # entity2id.txt (generated by prepare_data_for_transx.py) contains all entities in the crawled knowledge graph
    entity2index = read_map('../rs/entity2index.txt')
    full_entity2index = read_map('./amazon_book_entity2id.txt')
    full_relation2index = read_map("./amazon_book_relation2id.txt")
    entity2neighbor = get_neighbors_for_entity('amazon_book_triple2id.txt')

    full_embeddings = np.loadtxt(KGE_METHOD + '_entity2vec_' + str(ENTITY_EMBEDDING_DIM) + '.vec')
    relation_embeddings = np.loadtxt(KGE_METHOD + '_relation2vec_' + str(ENTITY_EMBEDDING_DIM) + '.vec')
    entity_embeddings = np.zeros([len(entity2index) + 1, ENTITY_EMBEDDING_DIM])
    context_embeddings = np.zeros([len(entity2index) + 1, len(full_relation2index)+1, ENTITY_EMBEDDING_DIM])
    padding_relation_embedding = np.array([0 for _ in range(relation_embeddings.shape[1])])
    relation_embeddings = np.vstack((padding_relation_embedding, relation_embeddings))
    # print(context_embeddings[0][38])
    temp = entity2neighbor[120640]
    # 存储与实体相连的关系类型信息
    # writer = open("../rs/entity_read_relation.txt", "w")
    #
    # print('writing entity embeddings...')
    # for entity, index in tqdm(entity2index.items()):
    #     if entity in full_entity2index:
    #         full_index = full_entity2index[entity]
    #         entity_embeddings[index] = full_embeddings[full_index]
    #         if full_index in entity2neighbor:
    #             relation2neighbors = entity2neighbor[full_index]
    #             writer.write("{}\t{}\n".format(index, ",".join(map(lambda x: str(x), relation2neighbors.keys()))))
    #             for relation_index, context_full_indices in relation2neighbors.items():
    #                 context_embeddings[index][relation_index] = np.average(full_embeddings[context_full_indices], axis=0)
    # writer.close()
    np.save(KGE_METHOD + '_relation_embeddings_' + str(ENTITY_EMBEDDING_DIM), entity_embeddings)
    # np.save(KGE_METHOD + '_entity_embeddings_' + str(ENTITY_EMBEDDING_DIM), entity_embeddings)
    # np.save(KGE_METHOD + '_aggregated_embeddings_' + str(ENTITY_EMBEDDING_DIM), context_embeddings)