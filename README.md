# Boostcamp AI Tech Stage 3 - MRC

## 팀원 소개

### MRC 6조 - GAAMA (Go Ahead and Ask Me Anything)

|                김도윤                |                나윤석                |             스후페엘레나             |                정근영                 |                 한웅희                 |
| :----------------------------------: | :----------------------------------: | :----------------------------------: | :-----------------------------------: | :------------------------------------: |
| [github](https://github.com/ddoyoon) | [github](https://github.com/alinghi) | [github](https://github.com/sheikra) | [github](https://github.com/GY-Jeong) | [github](https://github.com/dndgml913) |

## Report
https://www.notion.so/rmsdud/Wrap-up-98613c6224f840b49755b0bb8f3b0b72

## 소개

Open Domain Question & Answering (ODQA)

<center><img src="https://user-images.githubusercontent.com/28976984/119233396-cff49080-bb63-11eb-9b64-ccdae20df166.png" width="700" height="300"></center>

"한국에서 가장 오래된 나무는 무엇일까?" 이런 궁금한 질문이 있을 때 검색엔진에 가서 물어보신 적이 있을텐데요, 요즘엔 특히나 놀랍도록 정확한 답변을 주기도 합니다. 어떻게 가능한 걸까요?

질의 응답(Question Answering)은 다양한 종류의 질문에 대해 대답하는 인공지능을 만드는 연구 분야입니다. 그 중에서도 Open-Domain Question Answering 은 주어지는 지문이 따로 존재하지 않고 사전에 구축되어있는 knowledge resource 에서 질문에 대답할 수 있는 문서를 찾는 과정이 추가가 되어야하기에 더 어려운 문제입니다.

본 대회는 두 stage로 구성되어 있습니다. 첫 번째 단계는 질문에 관련된 문서를 찾아주는 "retriever"단계, 다음으로는 관련된 문서를 읽고 간결한 답변을 내보내 주는 "reader" 단계입니다. 이 두 단계를 각각 만든 뒤 둘을 이으면, 어려운 질문을 던져도 척척 답변을 해주는 질의응답 시스템을 직접 만들게 됩니다.

## 모델 소개

### Reader

- Korquad (https://huggingface.co/monologg/koelectra-base-v3-finetuned-korquad) model fine-tuning
- XLM-RoBERTa-large (https://huggingface.co/xlm-roberta-large) model fine-tuning

### Retriever

- Sparse passage retrieval (BM-25)
- Elastic search
