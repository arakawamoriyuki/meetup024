# coding: utf-8

from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

import MeCab
import csv

mt = MeCab.Tagger()

reports = []
with open("reports.tsv") as f:
    # reports.tsvには一行に口コミID,口コミがtab区切りで保存されている
    reader = csv.reader(f, delimiter="\t")
    for report_id, report in reader:
        words = []
        node = mt.parseToNode(report)
        while node:
            if len(node.surface) > 0:
                words.append(node.surface)
            node = node.next
        # wordsが口コミの単語のリスト,tagsには口コミIDを指定
        reports.append(TaggedDocument(words=words, tags=[report_id]))

model = Doc2Vec(documents=reports, size=128, window=8, min_count=5, workers=8)
model.save("doc2vec.model")