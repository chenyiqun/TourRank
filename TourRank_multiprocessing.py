import json
import os
import random
import copy
import time
import numpy as np
import openai
from openai import OpenAI
from tqdm import tqdm
from multiprocessing import Process, Manager
import logging

# gpt-3.5
client = OpenAI(
    api_key="Your_api_key"
)

def get_response(messages):

    try:
        try:
            completion = client.chat.completions.create(
            model="gpt-3.5-turbo", # gpt-4-turbo gpt-3.5-turbo
            messages=messages
            )
            return completion.choices[0].message.content
        except openai.InternalServerError:
            print("openai.InternalServerError")
            return get_response(messages)
    except openai.RateLimitError:
        print("Rate limit exceeded and try again...")
        return get_response(messages)
    

def get_prefix_role_prompt(query, N, M):

    return [{'role': 'system',
             'content': "You are an intelligent assistant that can compare multiple documents based on their relevancy to the given query."},
            {'role': 'user',
             'content': "I will provide you with the given query and {} documents. \nConsider the content of all the documents comprehensively and select the {} documents that are most relevant to the given query: {}.".format(N, M, query)},
            {'role': 'assistant', 'content': 'Okay, please provide the documents.'}]

def get_post_role_prompt(query, M):

    post_prompt = """The Query is: {}.
Now, you must output the top {} documents that are most relevant to the Query using the following format strictly, and nothing else. Don't output any explanation, just the following format:
Document 3, ..., Document 1""".format(query, M)
    
    return post_prompt

def sort_docs_by_relevance(doc_ids, relevance_scores):

    combined = list(zip(doc_ids, relevance_scores))

    sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)
    
    sorted_doc_ids = [doc_id for doc_id, score in sorted_combined]
    
    return sorted_doc_ids

def dcg_at_k(scores, k):

    scores = np.asfarray(scores)[:k]
    if scores.size:
        return np.sum(scores / np.log2(np.arange(2, scores.size + 2)))
    
    return 0.0

def ndcg_at_k(scores, k):

    # 计算DCG@k
    dcg_max = dcg_at_k(sorted(scores, reverse=True), k)
    if not dcg_max:
        return 0.
    
    return dcg_at_k(scores, k) / dcg_max

def get_top_M(answer, N=10, M=5, groups_docid=[]):

    # temp = answer.split('\n\n')[-1]
    temp = answer.split('\n')
    temp_length = len(temp)
    for i in range(1, temp_length+1):
        # print(i*(-1),temp_length)
        if 'Document' in temp[i*(-1)]:
            temp = temp[i*(-1)]
            break
    temp = temp.split(':')[-1]
    temp = temp.split('.')[0]
    temp = temp.split(',')
    top_M = []
    for doc in temp:
        try:
            try:
                flag = 0
                if '...' in doc:
                    flag = 1
                if flag == 0:
                    doc_num = int(doc.split()[-1]) - 1
                    top_M.append(doc_num)
            except IndexError:
                print('IndexError occured in doc. (get_top_M), just ignore it.')
                print(doc)
                # debug
                s = 'IndexError, ' + str(N) + ', ' + str(M) + ', ' + answer
                debug_path = './debug.txt'
                with open(debug_path, 'a') as f:
                    f.write('New Error: ' + '\n')
                    f.write(str(top_M) + '\n')
                    f.write(doc + '\n')
                    f.write(s + '\n')
                    f.write('end' + '\n' + '\n')

        except ValueError:
            for j in range(1, N+1):
                if j not in top_M:
                    doc_num = j
                    # top_M.append(j)
                    break
            print('ValueError occured in score. (get_top_M)')
            top_M.append(doc_num)
            # debug
            s = 'ValueError, ' + str(N) + ', ' + str(M) + ', ' + answer
            debug_path = './debug.txt'
            with open(debug_path, 'a') as f:
                f.write('New Error: ' + '\n')
                f.write(str(top_M) + '\n')
                f.write(doc + '\n')
                f.write(s + '\n')
                f.write('end' + '\n' + '\n')
    
    top_M_ids = []
    for doc_num in top_M:
        top_M_ids.append(groups_docid[doc_num])

    return top_M_ids

def get_final_top(answer, groups_docid):
    temp = answer.split('>')
    ranked_list = []
    for doc in temp:
        try:
            try:
                doc_num = int(doc.split()[-1]) - 1
                ranked_list.append(doc_num)
            except IndexError:
                print('IndexError occured in doc. (get_final_top), just ignore it.')
        except ValueError:
            for j in range(1, N+1):
                if j not in ranked_list:
                    # ranked_list.append(j)
                    doc_num = j
                    break
            print('ValueError occured in score. (get_final_top)')
            ranked_list.append(doc_num)
    
    ranked_docids_list = []
    for doc_num in ranked_list:
        ranked_docids_list.append(groups_docid[doc_num])

    return ranked_docids_list

def get_groups_chunk(docs_id, N=10):

    doc_num = len(docs_id)
    docs_groups = []
    cur_num = 0
    while cur_num < doc_num:
        docs_groups.append(docs_id[cur_num: cur_num + N])
        cur_num += N
    
    return docs_groups

def get_groups_skip(docs_id, to_n_groups=10, m_docs_per_group=10):

    # doc_num = len(docs_id)
    docs_groups = []
    for i in range(to_n_groups):
        cur_group = []
        for j in range(m_docs_per_group):
            cur_group.append(docs_id[j*to_n_groups + i])
        docs_groups.append(cur_group)
    
    return docs_groups

def group_processing(groups, query, N, M, all_contents, groups_score_dict_list):
    group_score_dict = {}
    random.shuffle(groups)
    messages = get_prefix_role_prompt(query, N, M)
    for j in range(len(groups)):
        doc_id = groups[j]
        content = all_contents[doc_id]
        messages.append({'role': 'user', 'content': 'Document {}: {}'.format(j+1, content)})
        messages.append({'role': 'assistant', 'content': 'Received Document {}.'.format(j+1)})
    messages.append({'role': 'user', 'content': get_post_role_prompt(query, M)})
    answer = get_response(messages)
    top_M_ids = get_top_M(answer, N=N, M=M, groups_docid=groups)
    for doc_id in top_M_ids:
        group_score_dict[doc_id] = 1
    groups_score_dict_list.append(group_score_dict)

def filter_processing(y_it, query, docs_id, all_contents, docs_score_dicts_list):
    # initialize the score of each doc_id (global)
    docs_score_dict = {}
    for doc in docs_id:
        docs_score_dict[doc] = 0

    # 100->50->20->10->5; Y times
    # 100->50, 5 groups
    N=20
    M=10
    # docs_groups = get_groups_skip(stage1_docs_id, to_n_groups=5, m_docs_per_group=N)
    with Manager() as manager:
        # initialize the score of each doc_id (this stage)
        groups_score_dict_list = manager.list()
        N=20
        M=10
        stage1_docs_id = docs_id
        docs_groups = get_groups_skip(stage1_docs_id, to_n_groups=5, m_docs_per_group=N)

        processes = []
        for y in range(len(docs_groups)):
            groups = docs_groups[y]
            p = Process(target=group_processing, args=(groups, query, N, M, all_contents, groups_score_dict_list))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        # combine the results
        for group_score_dict in groups_score_dict_list:
            for doc, score in group_score_dict.items():
                docs_score_dict[doc] += score
    
    # get randed list
    ranked_list = sort_docs_by_relevance(list(docs_score_dict.keys()), list(docs_score_dict.values()))

    # 50->20, 5 groups
    N=10
    M=4
    stage2_docs_id = ranked_list[: 50]
    with Manager() as manager:
        # initialize the score of each doc_id (this stage)
        groups_score_dict_list = manager.list()
        N=10
        M=4
        stage2_docs_id = ranked_list[: 50]
        docs_groups = get_groups_skip(stage2_docs_id, to_n_groups=5, m_docs_per_group=N)

        processes = []
        for y in range(len(docs_groups)):
            groups = docs_groups[y]
            p = Process(target=group_processing, args=(groups, query, N, M, all_contents, groups_score_dict_list))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        for group_score_dict in groups_score_dict_list:
            for doc, score in group_score_dict.items():
                docs_score_dict[doc] += score

    ranked_list = sort_docs_by_relevance(list(docs_score_dict.keys()), list(docs_score_dict.values()))

    # 20->10;
    N=20
    M=10
    stage3_docs_id = ranked_list[: 20]
    docs_groups = get_groups_skip(stage3_docs_id, to_n_groups=1, m_docs_per_group=N)
    for bat in range(len(docs_groups)):
        groups = docs_groups[bat]
        random.shuffle(groups)
        messages = get_prefix_role_prompt(query, N, M)
        for j in range(len(groups)):
            doc_id = groups[j]
            content = all_contents[doc_id]
            messages.append({'role': 'user', 'content': 'Document {}: {}'.format(j+1, content)})
            messages.append({'role': 'assistant', 'content': 'Received Document {}.'.format(j+1)})
        messages.append({'role': 'user', 'content': get_post_role_prompt(query, M)})
        answer = get_response(messages)
        top_M_ids = get_top_M(answer, N=N, M=M, groups_docid=groups)
        for doc_id in top_M_ids:
            docs_score_dict[doc_id] += 1

    ranked_list = sort_docs_by_relevance(list(docs_score_dict.keys()), list(docs_score_dict.values()))

    # 10->5;
    N=10
    M=5
    stage4_docs_id = ranked_list[: 10]
    docs_groups = get_groups_skip(stage4_docs_id, to_n_groups=1, m_docs_per_group=N)
    for bat in range(len(docs_groups)):
        groups = docs_groups[bat]
        random.shuffle(groups)
        messages = get_prefix_role_prompt(query, N, M)
        for j in range(len(groups)):
            doc_id = groups[j]
            content = all_contents[doc_id]
            messages.append({'role': 'user', 'content': 'Document {}: {}'.format(j+1, content)})
            messages.append({'role': 'assistant', 'content': 'Received Document {}.'.format(j+1)})
        messages.append({'role': 'user', 'content': get_post_role_prompt(query, M)})
        answer = get_response(messages)
        top_M_ids = get_top_M(answer, N=N, M=M, groups_docid=groups)
        for doc_id in top_M_ids:
            docs_score_dict[doc_id] += 1

    ranked_list = sort_docs_by_relevance(list(docs_score_dict.keys()), list(docs_score_dict.values()))

    # 5->2;
    N=5
    M=2
    stage4_docs_id = ranked_list[: 5]
    docs_groups = get_groups_skip(stage4_docs_id, to_n_groups=1, m_docs_per_group=N)
    for bat in range(len(docs_groups)):
        groups = docs_groups[bat]
        random.shuffle(groups)
        messages = get_prefix_role_prompt(query, N, M)
        for j in range(len(groups)):
            doc_id = groups[j]
            content = all_contents[doc_id]
            messages.append({'role': 'user', 'content': 'Document {}: {}'.format(j+1, content)})
            messages.append({'role': 'assistant', 'content': 'Received Document {}.'.format(j+1)})
        messages.append({'role': 'user', 'content': get_post_role_prompt(query, M)})
        answer = get_response(messages)
        top_M_ids = get_top_M(answer, N=N, M=M, groups_docid=groups)
        for doc_id in top_M_ids:
            docs_score_dict[doc_id] += 1

    # combine the points of this tournament to global storage
    docs_score_dicts_list.append(docs_score_dict)

    print("Finished {} process.".format(y_it+1))


if __name__ == '__main__':

    pre_path = '.'

    # load json data
    query_docs_path = '{}/data/bm25_dl20_top100.jsonl'.format(pre_path)
    query_docs = []
    with open(query_docs_path, 'r') as file:
        for line in file:
            query_docs.append(json.loads(line))

    # get all queries
    all_queries = [item['query'] for item in query_docs]

    # open the txt file, get the label
    # rating_file_path = '{}/RankGPT/topics-and-qrels/qrels.beir-v1.0.0-signal1m.test.txt'.format(pre_path)
    rating_file_path = '{}/data/id_rating_20.txt'.format(pre_path)
    with open(rating_file_path, 'r') as file:
        ratings = file.read()

    # text processing
    docs_ratings_dic = {}
    items = ratings.split('\n')
    for item in items:
        temp_list = item.split()
        temp_doc_id, temp_rating = temp_list[2], temp_list[3]
        docs_ratings_dic[temp_doc_id] = temp_rating

    # get the ideal ranking permutation
    all_docs, all_ratings, all_contents, ideal_scores = [], [], {}, {}
    for i in range(len(query_docs)):
        cur = query_docs[i]
        docs = cur['hits']
        cur_docs, cur_ratings = [], []
        for doc in docs:
            doc_id = doc['docid']
            content = doc['content']
            if doc_id in docs_ratings_dic:
                rel = int(docs_ratings_dic[doc_id])
                # print(doc_id, rel, '---------------')
            else:
                rel = 0
                # print(doc_id, rel)
            cur_docs.append(doc_id)
            cur_ratings.append(rel)
            all_contents[doc_id] = content
            ideal_scores[doc_id] = rel
        all_docs.append(cur_docs)
        all_ratings.append(cur_ratings)

    
    # logger for record the results
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler('{}/TREC_results/gpt_3.5/bm25_dl20_reverse_detail_points/my_log.log'.format(pre_path))
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger = logging.getLogger('my_logger')


    # GPT generation *********
    for i in range(len(all_queries)):
        if i <= -1:
            continue
        query = all_queries[i]
        print(i, ' ' , query)
        logger.info('{}  '.format(i) + query + ':')
        # logger.info('\t')
        docs_id = copy.deepcopy(all_docs[i])

        # initialize the score of each doc
        docs_score_dict = {}
        for doc in docs_id:
            docs_score_dict[doc] = 0


        # get the initial ranking permutation
        ranked_list = docs_id
        # get the real labels of the initial ranking permutation
        ranked_docs_score = []
        for doc_id in ranked_list:
            ranked_docs_score.append(ideal_scores[doc_id])

        # calculate the current query's ndcg@1 5 10 20
        cur_ndcg_1 = ndcg_at_k(ranked_docs_score, k=1)
        cur_ndcg_5 = ndcg_at_k(ranked_docs_score, k=5)
        cur_ndcg_10 = ndcg_at_k(ranked_docs_score, k=10)
        cur_ndcg_20 = ndcg_at_k(ranked_docs_score, k=20)
        print('The initial metrics, the ndcg@1, 5, 10, 20 of query {} is {}, {}, {}, {}.'.format(i, round(cur_ndcg_1, 4), round(cur_ndcg_5, 4), round(cur_ndcg_10, 4), round(cur_ndcg_20, 4)))
        # print('\t')
        logger.info('The initial metrics, the ndcg@1, 5, 10, 20 of query {} is {}, {}, {}, {}.'.format(i, round(cur_ndcg_1, 4), round(cur_ndcg_5, 4), round(cur_ndcg_10, 4), round(cur_ndcg_20, 4)))
        # logger.info('\t')
        # save th initial results
        for j in range(len(ranked_list)):
            with open('{}/TREC_results/gpt_3.5/bm25_dl20_detail_points/results_filter_accumulate_final_{}.txt'.format(pre_path, 0), 'a') as f:
                f.write(str(query_docs[i]['hits'][0]['qid']) + ' Q0 ' + ranked_list[j] + ' ' + str(j+1) + ' ' + str(100-j) + ' rank' + '\n')

        # # shuffle the results of BM25
        # random.seed(1)
        # random.shuffle(docs_id)
        # random.seed()
                
        # # reverse the results of BM25
        # docs_id.reverse()

        # # test the results of shuffle/reverseBM25
        # ranked_list = docs_id
        # # get the rel scores
        # ranked_docs_score = []
        # for doc_id in ranked_list:
        #     ranked_docs_score.append(ideal_scores[doc_id])

        # # calculate the current query's ndcg@1 5 10 20
        # cur_ndcg_1 = ndcg_at_k(ranked_docs_score, k=1)
        # cur_ndcg_5 = ndcg_at_k(ranked_docs_score, k=5)
        # cur_ndcg_10 = ndcg_at_k(ranked_docs_score, k=10)
        # cur_ndcg_20 = ndcg_at_k(ranked_docs_score, k=20)
        # print('The reversed metrics, the ndcg@1, 5, 10, 20 of query {} is {}, {}, {}, {}.'.format(i, round(cur_ndcg_1, 4), round(cur_ndcg_5, 4), round(cur_ndcg_10, 4), round(cur_ndcg_20, 4)))
        # print('\t')
        # logger.info('The reversed metrics, the ndcg@1, 5, 10, 20 of query {} is {}, {}, {}, {}.'.format(i, round(cur_ndcg_1, 4), round(cur_ndcg_5, 4), round(cur_ndcg_10, 4), round(cur_ndcg_20, 4)))
        # # logger.info('\t')
        # # save the results
        # for j in range(len(ranked_list)):
        #     with open('{}/TREC_results/gpt_3.5/bm25_dl20_detail_points/results_filter_accumulate_final_{}.txt'.format(pre_path, 0), 'a') as f:
        #         f.write(str(query_docs[i]['hits'][0]['qid']) + ' Q0 ' + ranked_list[j] + ' ' + str(j+1) + ' ' + str(100-j) + ' rank' + '\n')


        with Manager() as manager:
            docs_score_dicts_list = manager.list()
            processes = []
            Y = 10
            for y in range(Y):
                p = Process(target=filter_processing, args=(y, query, docs_id, all_contents, docs_score_dicts_list))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            print('\t')

            # initialize the global score of each doc
            global_docs_score_dict = {}
            for doc in docs_id:
                global_docs_score_dict[doc] = 0
            
            # process scores in each tournament
            if Y - len(docs_score_dicts_list) == 1:
                temp = docs_score_dicts_list[0]
                docs_score_dicts_list.append(temp)
            elif Y - len(docs_score_dicts_list) == 2:
                temp = docs_score_dicts_list[0]
                temp2 = docs_score_dicts_list[1]
                docs_score_dicts_list.append(temp)
                docs_score_dicts_list.append(temp2)
            elif Y - len(docs_score_dicts_list) == 3:
                temp = docs_score_dicts_list[0]
                temp2 = docs_score_dicts_list[1]
                temp3 = docs_score_dicts_list[2]
                docs_score_dicts_list.append(temp)
                docs_score_dicts_list.append(temp2)
                docs_score_dicts_list.append(temp3)

            # save the scores of each tournament
            for y in range(len(docs_score_dicts_list)):
                cur_docs_score_dict = docs_score_dicts_list[y]
                save_docs_score_dict = {'i': i, 'tournament': y, 'score_dict': cur_docs_score_dict}
                score_path = '{}/TREC_results/gpt_3.5/bm25_dl20_reverse_detail_points/score.json'.format(pre_path)
                with open(score_path, 'a') as f:
                    json.dump(save_docs_score_dict, f)
                    f.write("\n")

            for y in range(len(docs_score_dicts_list)):
                cur_docs_score_dict = docs_score_dicts_list[y]
                for doc_id, score in cur_docs_score_dict.items():
                    global_docs_score_dict[doc_id] += score
                # get ranked list
                ranked_list = sort_docs_by_relevance(list(global_docs_score_dict.keys()), list(global_docs_score_dict.values()))
                # get rel labels of ranked list
                ranked_docs_score = []
                for doc_id in ranked_list:
                    ranked_docs_score.append(ideal_scores[doc_id])
                # calculate the ndcg@1 5 10 20
                cur_ndcg_1 = ndcg_at_k(ranked_docs_score, k=1)
                cur_ndcg_5 = ndcg_at_k(ranked_docs_score, k=5)
                cur_ndcg_10 = ndcg_at_k(ranked_docs_score, k=10)
                cur_ndcg_20 = ndcg_at_k(ranked_docs_score, k=20)
                print('After {} iteration, the ndcg@1, 5, 10, 20 of query {} is {}, {}, {}, {}.'.format(y+1, i, round(cur_ndcg_1, 4), round(cur_ndcg_5, 4), round(cur_ndcg_10, 4), round(cur_ndcg_20, 4)))
                # print('\t')
                logger.info('After {} iteration, the ndcg@1, 5, 10, 20 of query {} is {}, {}, {}, {}.'.format(y+1, i, round(cur_ndcg_1, 4), round(cur_ndcg_5, 4), round(cur_ndcg_10, 4), round(cur_ndcg_20, 4)))
                # logger.info('\t')
                # save the results of each tournament
                for j in range(len(ranked_list)):
                    with open('{}/TREC_results/gpt_3.5/bm25_dl20_reverse_detail_points/results_filter_accumulate_final_{}.txt'.format(pre_path, y+1), 'a') as f:
                        f.write(str(query_docs[i]['hits'][0]['qid']) + ' Q0 ' + ranked_list[j] + ' ' + str(j+1) + ' ' + str(100-j) + ' rank' + '\n')

        print('\t')
        print('-----------------------------------------------')
        print('\t')

    print("Finished!")


