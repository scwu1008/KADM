import pandas as pd
from tqdm import tqdm


def read_rating_triple(path):
    def read_map(file):
        item2entity_map = {}
        reader = open(file, encoding='utf-8')
        for line in reader:
            array = line.split('\t')
            if len(array) != 2:  # to skip the first line in entity2id.txt
                continue
            item_id = array[0]
            entity_id = array[1].strip()
            item2entity_map[item_id] = entity_id
        reader.close()
        return item2entity_map

    result = set()
    rating_data = pd.read_csv(path)
    item2entity_dict = read_map("../rs/book2entity.txt")
    for i in tqdm(range(len(rating_data)), desc='read rating_data'):
        user_id = rating_data.loc[i, 'user']
        item_id = rating_data.loc[i, 'item']
        result.add("{}\t{}\t{}".format(user_id, 'interact_with',
                                          '<http://rdf.freebase.com/ns/' + item2entity_dict[item_id] + '>'))
    return result


def prepare_data(kg_in, rating_in, triple_out, relation_out, entity_out):
    relation2index = {}
    entity2index = {}
    relation_list = []
    entity_list = []

    reader_kg = open(kg_in, encoding='utf-8')
    # user-item 之间interact-with关系 作为新的异构关系加入kg中
    read_rating = read_rating_triple(rating_in)
    writer_triple = open(triple_out, 'w', encoding='utf-8')
    writer_relation = open(relation_out, 'w', encoding='utf-8')
    writer_entity = open(entity_out, 'w', encoding='utf-8')

    entity_cnt = 0
    relation_cnt = 1
    triple_cnt = 0

    print('reading knowledge graph ...')
    kg = reader_kg.read().strip().split('\n')
    kg = kg + list(read_rating)
    print('writing triples to triple2id.txt ...')
    writer_triple.write('%d\n' % len(kg))
    for line in tqdm(kg):
        array = line.split('\t')
        if len(array) < 3:
            print(line)
            print("")
        head = array[0]
        relation = array[1]
        tail = array[2]
        if head in entity2index:
            head_index = entity2index[head]
        else:
            head_index = entity_cnt
            entity2index[head] = entity_cnt
            entity_list.append(head)
            entity_cnt += 1
        if tail in entity2index:
            tail_index = entity2index[tail]
        else:
            tail_index = entity_cnt
            entity2index[tail] = entity_cnt
            entity_list.append(tail)
            entity_cnt += 1
        if relation in relation2index:
            relation_index = relation2index[relation]
        else:
            relation_index = relation_cnt
            relation2index[relation] = relation_cnt
            relation_list.append(relation)
            relation_cnt += 1
        writer_triple.write(
            '%d\t%d\t%d\n' % (head_index, tail_index, relation_index))
        triple_cnt += 1
    print('triple size: %d' % triple_cnt)

    print('writing entities to entity2id.txt ...')
    writer_entity.write('%d\n' % entity_cnt)
    for i, entity in enumerate(entity_list):
        writer_entity.write('%s\t%d\n' % (entity, i))
    print('entity size: %d' % entity_cnt)

    print('writing relations to relation2id.txt ...')
    writer_relation.write('%d\n' % (relation_cnt - 1))
    for i, relation in enumerate(relation_list):
        writer_relation.write('%s\t%d\n' % (relation, i+1))
    print('relation size: %d' % relation_cnt)

    reader_kg.close()


if __name__ == '__main__':
    prepare_data(kg_in='kg.txt', rating_in="../rs/ratings.csv", triple_out='amazon_book_triple2id.txt',
                 relation_out='amazon_book_relation2id.txt', entity_out='amazon_book_entity2id.txt')
