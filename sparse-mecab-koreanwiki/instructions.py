# 사용할 stopwords 저장
with open("/opt/ml/input/data/stopwords.txt", "r") as f:
    stopwords = f.readlines()
stopwords = [x.split("\n")[0] for x in stopwords]

# Stopwords 를 제거해주는 함수
# 문서를 tokenize 한 후 BM25 점수 계산 전에 적용해주세요
def remove_stop(tok):
    return [x for x in tok if x.strip() not in stopwords]


# 질문에 특정적인 용어를 삭제하는 함수
def remove_q(query):
    stop = "|".join(
        "어느 무엇인가요 무엇 누가 누구인가요 누구인가 누구 어디에서 어디에 어디서 어디인가요 어디를 어디 언제 어떤 어떠한 몇 얼마 얼마나 뭐 어떻게 무슨 \?".split(
            " "
        )
    )
    rm = re.sub(stop, "", query).strip()
    return rm


# Retriever 에서 사용되는 토크나이저
def process_morphs_simple(query):
    rm = remove_q(query)
    pos = mecab.pos(rm)
    return [x[0] for x in pos if x[1][0] != "J" and x[1][0] != "E" and x[1][0] != "X"]

# 주의할 점!
# wiki 문서 자체를 고쳤기 때문에 original_context 와 비교하기 위해서는 document_id 로 변경된 정답 context 를 가져와야 합니다.
'''
# Retriever class init 시 아래와 같이 저장해둡니다
self.id2context = {
            sample["document_id"]: sample["text"] for sample in wiki.values()
        }

# retrieval 함수 내에서 아래와 같이 사용
tmp["original_context"] = self.id2context[example["document_id"]]
'''