from datasets import (
    Dataset,
    load_from_disk,
    concatenate_datasets,
)
org_dataset = load_from_disk("data/test_dataset")
tt=org_dataset["validation"]
f=open("./question.csv","w")
for idx,i in enumerate(tt):
	f.write("\t ".join([str(idx+1),i["id"],i["question"]])+"\n")
f.close()
