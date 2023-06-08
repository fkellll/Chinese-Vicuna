"""
batch_selfinstruct_generate.py

run:
python -m generate_instruction generate_instruction_following_data \
  --output_dir ./ \
  --num_instructions_to_generate 10 \
  --model_name="text-davinci-003" \
"""
import time
import json
import os
import random
import re
import string
from functools import partial
from multiprocessing import Pool

import numpy as np
import tqdm
from rouge_score import rouge_scorer
import utils
from gpt3_api import make_requests as make_gpt3_requests

import fire

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader


def post_process_gpt3_response(num_prompt_instructions, response):
    if response is None:
        return []
    raw_instructions = f"{num_prompt_instructions+1}. Instruction:" + response["text"]
    raw_instructions = re.split("###", raw_instructions)
    instructions = []
    for idx, inst in enumerate(raw_instructions):
        # if the decoding stops due to length, the last example is likely truncated so we discard it
        if idx == len(raw_instructions) - 1 and response["finish_reason"] == "length":
            continue
        idx += num_prompt_instructions + 1
        splitted_data = re.split(f"{idx}\.\s+(Instruction|Input|Output):", inst)
        if len(splitted_data) != 7:
            continue
        else:
            inst = splitted_data[2].strip()
            input = splitted_data[4].strip()
            input = "" if input.lower() == "<noinput>" else input
            output = splitted_data[6].strip()
        # filter out too short or too long instructions
        if len(inst.split()) <= 3 or len(inst.split()) > 150:
            continue
        # filter based on keywords that are not suitable for language models.
        blacklist = [
            "image",
            "images",
            "graph",
            "graphs",
            "picture",
            "pictures",
            "file",
            "files",
            "map",
            "maps",
            "draw",
            "plot",
            "go to",
            "video",
            "audio",
            "music",
            "flowchart",
            "diagram",
        ]
        blacklist += []
        if any(find_word_in_string(word, inst) for word in blacklist):
            continue
        # We found that the model tends to add "write a program" to some existing instructions, which lead to a lot of such instructions.
        # And it's a bit comfusing whether the model need to write a program or directly output the result.
        # Here we filter them out.
        # Note this is not a comprehensive filtering for all programming instructions.
        if inst.startswith("Write a program"):
            continue
        # filter those starting with punctuation
        if inst[0] in string.punctuation:
            continue
        instructions.append({"instruction": inst, "input": input, "output": output})
    return instructions

def post_process_gpt3_question_response(response):
    if response is None or response["choices"][0]["finish_reason"] == "length":
        return []
    raw_instructions = re.split(r"\n\d+\s?\. ", response["choices"][0]["text"])
    instructions = []
    for inst in raw_instructions:
        inst = re.sub(r"\s+", " ", inst).strip()
        inst = inst.strip().capitalize()
        if inst == "":
            continue
        # filter out too short or too long instructions
        if len(inst) <= 3 or len(inst) > 150:
            continue
        # filter based on keywords that are not suitable for language models.
        if any(find_word_in_string(word, inst) for word in ["image", "images", "graph", "graphs", "picture", "pictures", "file", "files", "map", "maps", "draw", "plot", "go to"]):
            continue
        # We found that the model tends to add "write a program" to some existing instructions, which lead to a lot of such instructions.
        # And it's a bit comfusing whether the model need to write a program or directly output the result. 
        # Here we filter them out.
        # Note this is not a comprehensive filtering for all programming instructions.
        if inst.startswith("Write a program"):
            continue
        # filter those starting with punctuation
        if inst[0] in string.punctuation:
            continue
        instructions.append(inst)
    return instructions


def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)


# def generate_instruction_following_data(
#     output_dir="./",
#     seed_tasks_path="./seed_tasks.jsonl",
#     num_instructions_to_generate=100,
#     model_name="text-davinci-003",
#     num_prompt_instructions=3,
#     request_batch_size=5,
#     temperature=1.0,
#     top_p=1.0,
#     num_cpus=16,
# ):
#     seed_tasks = [json.loads(l) for l in open(seed_tasks_path, "r")]
#     seed_instruction_data = [
#         {"instruction": t["instruction"], "input": t["instances"][0]["input"], "output": t["instances"][0]["output"]}
#         for t in seed_tasks
#     ]
#     print(f"Loaded {len(seed_instruction_data)} human-written seed instructions")

#     os.makedirs(output_dir, exist_ok=True)
#     request_idx = 0
#     # load the LM-generated instructions
#     machine_instruction_data = []
#     if os.path.exists(os.path.join(output_dir, "regen.json")):
#         machine_instruction_data = utils.jload(os.path.join(output_dir, "regen.json"))
#         print(f"Loaded {len(machine_instruction_data)} machine-generated instructions")

#     # similarities = {}
#     scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

#     # now let's generate new instructions!
#     progress_bar = tqdm.tqdm(total=num_instructions_to_generate)
#     if machine_instruction_data:
#         progress_bar.update(len(machine_instruction_data))

#     # first we tokenize all the seed instructions and generated machine instructions
#     all_instructions = [d["instruction"] for d in seed_instruction_data] + [
#         d["instruction"] for d in machine_instruction_data
#     ]
#     all_instruction_tokens = [scorer._tokenizer.tokenize(inst) for inst in all_instructions]

#     while len(machine_instruction_data) < num_instructions_to_generate:
#         request_idx += 1

#         batch_inputs = []
#         for _ in range(request_batch_size):
#             # only sampling from the seed tasks
#             prompt_instructions = random.sample(seed_instruction_data, num_prompt_instructions)
#             prompt = encode_prompt(prompt_instructions)
#             batch_inputs.append(prompt)
#         decoding_args = utils.OpenAIDecodingArguments(
#             temperature=temperature,
#             n=1,
#             max_tokens=3072,  # hard-code to maximize the length. the requests will be automatically adjusted
#             top_p=top_p,
#             stop=["\n20", "20.", "20."],
#         )
#         request_start = time.time()
#         results = utils.openai_completion(
#             prompts=batch_inputs,
#             model_name=model_name,
#             batch_size=request_batch_size,
#             decoding_args=decoding_args,
#             logit_bias={"50256": -100},  # prevent the <|endoftext|> token from being generated
#         )
#         request_duration = time.time() - request_start

#         process_start = time.time()
#         instruction_data = []
#         for result in results:
#             new_instructions = post_process_gpt3_response(num_prompt_instructions, result)
#             instruction_data += new_instructions

#         total = len(instruction_data)
#         keep = 0
#         for instruction_data_entry in instruction_data:
#             # computing similarity with the pre-tokenzied instructions
#             new_instruction_tokens = scorer._tokenizer.tokenize(instruction_data_entry["instruction"])
#             with Pool(num_cpus) as p:
#                 rouge_scores = p.map(
#                     partial(rouge_scorer._score_lcs, new_instruction_tokens),
#                     all_instruction_tokens,
#                 )
#             rouge_scores = [score.fmeasure for score in rouge_scores]
#             most_similar_instructions = {
#                 all_instructions[i]: rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
#             }
#             if max(rouge_scores) > 0.7:
#                 continue
#             else:
#                 keep += 1
#             instruction_data_entry["most_similar_instructions"] = most_similar_instructions
#             instruction_data_entry["avg_similarity_score"] = float(np.mean(rouge_scores))
#             machine_instruction_data.append(instruction_data_entry)
#             all_instructions.append(instruction_data_entry["instruction"])
#             all_instruction_tokens.append(new_instruction_tokens)
#             progress_bar.update(1)
#         process_duration = time.time() - process_start
#         print(f"Request {request_idx} took {request_duration:.2f}s, processing took {process_duration:.2f}s")
#         print(f"Generated {total} instructions, kept {keep} instructions")
#         utils.jdump(machine_instruction_data, os.path.join(output_dir, "regen.json"))


def loadBook(source_folder_path, doc_split_temp_path):
    loader = TextLoader(source_folder_path)
    # 将数据转成 document 对象，每个文件会作为一个 document
    documents = loader.load()
    # 初始化加载器  MarkdownTextSplitter md的文档可以试一下
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    # 切割加载的 document
    split_docs = text_splitter.split_documents(documents)
    with open(doc_split_temp_path, 'w') as f:
        for index, doc in enumerate(split_docs):
            page_content = doc.page_content.strip()
            print(index)
            print(len(page_content))
            if not page_content:
                continue
            f.write(json.dumps({
                "content": page_content
            }, ensure_ascii=False).encode('utf-8').decode('utf-8') + "\n")
    return doc_split_temp_path
    # # 初始化 openai 的 embeddings 对象
    # embeddings = OpenAIEmbeddings()
    # # 将 document 通过 openai 的 embeddings 对象计算 embedding 向量信息并临时存入 Chroma 向量数据库，用于后续匹配查询
    # vectorstore = FAISS.from_documents(documents = split_docs, embedding = embeddings)

def generateQA(split_content_file_path):
    with open(split_content_file_path, 'r') as f:
        for index, line in enumerate(f):
            if not line.strip():
                continue
            data = json.loads(line.strip())
            preContent = generateQAWithContent(data, preContent)
            
    
def generateQAWithContent(content, preContentJson):
    """
    生成问题和答案
    :param content: 文本内容
    :param preContentJson: 上一次的内容
    :return: 问题和答案
    """
    
def generateWithContent(prompt):
    """
    生成问题和答案
    :param content: 文本内容
    :param preContentJson: 上一次的内容
    :return: 问题和答案
    """
    temperature=1.0,
    top_p=1.0,
    num_cpus=16,
    batch_inputs = []
    batch_inputs.append(prompt)
    decoding_args = utils.OpenAIDecodingArguments(
        temperature=temperature,
        n=1,
        max_tokens=3072,  # hard-code to maximize the length. the requests will be automatically adjusted
        top_p=top_p,
        stop=["\n20", "20.", "20.","\n\n", "\n16", "16.", "16 ."],
    )
    request_start = time.time()
    # results = utils.openai_completion(
    #     prompts=batch_inputs,
    #     model_name=model_name,
    #     batch_size=len(batch_inputs),
    #     decoding_args=decoding_args,
    #     logit_bias={"50256": -100},  # prevent the <|endoftext|> token from being generated
    # )
    results = make_gpt3_requests(
        engine='text-davinci-003',
        prompts=batch_inputs  ,
        max_tokens=1024,
        temperature=0.7,
        top_p=0.5,
        frequency_penalty=0,
        presence_penalty=2,
        stop_sequences=["\n\n", "\n16", "16.", "16 ."],
        logprobs=1,
        n=1,
        best_of=1,
        organization=None,
    )
    request_duration = time.time() - request_start

    process_start = time.time()
    instruction_data = []
    for result in results:
        new_instructions = post_process_gpt3_question_response(result["response"])
        instruction_data += new_instructions
    return instruction_data

# def main(task, **kwargs):
#     globals()[task](**kwargs)

def generateQAWithContent(content, preContentJson):
    """
    生成问题和答案
    :param content: 文本内容
    :param preContentJson: 上一次的内容
    :return: 问题和答案
    """
    
def generateAnswerWithContent(content, preContent, question):
    """
    生成问题和答案
    :param content: 文本内容
    :param preContentJson: 上一次的内容
    :return: 问题和答案
    """
    temperature=1.0,
    top_p=1.0,
    num_cpus=16,
    batch_inputs = []
    prompt = ''
    if preContent and preContent.strip() != "":
        prompt += f"preContent:" + preContent + "\n"
    if content and content.strip() != "":
        prompt += f"content:" + content + "\n"    
    prompt += f"###\n"
    model_name="text-davinci-003",
    prompt += open("datasets/prompt_generate_answer.txt").read() + "\n" + question +"\n"
    batch_inputs.append(prompt)
    decoding_args = utils.OpenAIDecodingArguments(
        temperature=temperature,
        n=1,
        max_tokens=3072,  # hard-code to maximize the length. the requests will be automatically adjusted
        top_p=top_p,
        stop=["\n20", "20.", "20.","\n\n", "\n16", "16.", "16 ."],
    )
    request_start = time.time()
    # results = utils.openai_completion(
    #     prompts=batch_inputs,
    #     model_name=model_name,
    #     batch_size=len(batch_inputs),
    #     decoding_args=decoding_args,
    #     logit_bias={"50256": -100},  # prevent the <|endoftext|> token from being generated
    # )
    results = make_gpt3_requests(
        engine='text-davinci-003',
        prompts=batch_inputs  ,
        max_tokens=1024,
        temperature=0.7,
        top_p=0.5,
        frequency_penalty=0,
        presence_penalty=2,
        stop_sequences=["\n\n", "\n16", "16.", "16 ."],
        logprobs=1,
        n=1,
        best_of=1,
        organization=None,
    )
    request_duration = time.time() - request_start

    process_start = time.time()
    instruction_data = []
    for result in results:
        new_instructions = post_process_gpt3_question_response(result["response"])
        instruction_data += new_instructions
    return instruction_data


def generate_qa(bookPath, qaPath):
    split_id = 0
    ext = 'txt'
    bookName = os.path.splitext(os.path.basename(bookPath))[0]
    bookDir = os.path.dirname(bookPath) + '/'

    doc_split_temp_path = os.path.join(os.path.dirname(bookDir), bookName + '_plit.' + ext)
    loadBook(bookPath, doc_split_temp_path)
    next_merge_path = os.path.join(os.path.dirname(bookDir), bookName + '_merge.' + ext)
    if os.path.getsize(doc_split_temp_path) < 500:
        need_merge = False
    else:
        need_merge = True
    with open(doc_split_temp_path, 'r') as f: # 打开文件
        for index, line in enumerate(f):
            data = json.loads(line.strip()) # 解析每一行
            content = data['content']
            # 生成总结的内容问题
            prompt = content + '\n' + open("datasets/prompt_generate_question.txt").read() + "\n"
            questions = generateWithContent(prompt)
            questions_str = "\n".join(questions)
            QA = {"instruction": questions_str,
                "input": None,
                "output": content,
                "split_id": split_id,
                "content": content}
            split_id += 1
            with open(qaPath, 'a') as f:
                print(index)
                if not QA:
                    continue
                f.write(json.dumps(QA, ensure_ascii=False).encode('utf-8').decode('utf-8') + "\n")
            # 总结这段内容，写入merge.txt中
            if (need_merge):
                prompt = content + '\n' + open("datasets/prompt_merge.txt").read() + "\n"
                merge_contens = generateWithContent(prompt)
                merge_content = "\n".join(merge_contens)
                with open(next_merge_path, 'a') as f:
                    f.write(merge_content + "\n")

    if need_merge: generate_qa(next_merge_path, qaPath)

generate_qa('datasets/book/book.txt', 'datasets/book/book_qa.jsonl')

# if __name__ == "__main__":
#     fire.Fire(main)
# loadBook('datasets/book/book.txt')
# QAList = []
# split_id = 0
# with open('datasets/book/temp_split.txt', 'r') as f: # 打开文件
#     line1 = f.readline() # 读取第一行
#     line2 = f.readline() # 读取第二行
#     data1 = json.loads(line1.strip()) # 解析第一行
#     data2 = json.loads(line2.strip()) # 解析第二行
# # 获取data2['content']里有几段内容，并获取其为第几段
# split = 1
# for i, content in enumerate(data2['content']):
#     current_content = content
#     QA = {"instruction": '第' + str(i+1) + '段内容如下',
#         "input": None,
#         "output": current_content,
#         "split_id": split_id,
#         "preContent": data1['content'],
#         "content": data2['content']}
#     print(QA)
#     QAList.append(QA)

# questions = generateQuestionWithContent(data1['content'], data2['content'])  
# print(questions)
# split = 1
# for qustion in questions:
#     print('qustion: ' + qustion)
#     answer = generateAnswerWithContent(data1['content'], data2['content'], qustion)
#     answer = ''.join(answer)
#     print('answer: ' + answer)
#     if not answer or answer.strip() == "":
#         continue
#     QA = {"instruction": qustion,
#         "input": None,
#         "output": answer,
#         "split_id": split_id + '_' + split,
#         "preContent": data1['content'],
#         "content": data2['content']}
#     QAList.append(QA)

# doc_split_temp_path = os.path.join(os.path.dirname('datasets/book/'), 'qa.jsonl')
# with open(doc_split_temp_path, 'w') as f:
#     for index, qa in enumerate(QAList):
#         print(index)
#         if not qa:
#             continue
#         f.write(json.dumps(qa, ensure_ascii=False).encode('utf-8').decode('utf-8') + "\n")


