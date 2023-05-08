import json
import os

from bs4 import BeautifulSoup

def pubmed_raw_to_partition(param_dict,type):
    raw_path = param_dict["raw_path"]
    partition_path = param_dict["partition_path"]
    if not os.path.exists(partition_path):
        os.mkdir(partition_path)
    if not os.path.exists(os.path.join(partition_path,type)):
        os.mkdir(os.path.join(partition_path,type))
    with open(os.path.join(raw_path,type+".txt"),"r",encoding="utf-8") as raw_data_file:
        li = raw_data_file.readlines()
        li = li[:3]
        i = 0
        for line in li:
            content = json.loads(line)
            article = content['article_text']
            article = ''.join(article)
            soup = BeautifulSoup(article, "html.parser")
            doc = soup.get_text()

            abstract = content['abstract_text']
            abstract = ''.join(abstract)
            soup1 = BeautifulSoup(abstract, "html.parser")
            abs = soup1.get_text()


            with open(os.path.join(partition_path,type,type+"_sample"+ str(i) +".story") , 'w', encoding='utf-8') as f:
                f.write(doc + '\n')
                f.write('@highlight' + '\n')
                f.write(abs)
            i += 1



