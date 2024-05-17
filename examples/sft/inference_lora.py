#first lora
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel,LoraConfig
from json_utils import ExtractUserContent,GetLabel
from tqdm import tqdm
from sklearn import metrics

model_path = '/home/dxd/qwen1.5-7b-chat/'
lora_path = '/home/dxd/Qwen1.5/examples/sft/output_qwen/'
config = LoraConfig(
            r=64,
            lora_alpha=16,
            target_modules=[
                "up_proj",
                "v_proj",
                "gate_proj",
                "q_proj",
                "k_proj",
                "o_proj",
                "down_proj"
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,device_map="auto",torch_dtype=torch.bfloat16)

# 加载lora权重
model = PeftModel.from_pretrained(model, model_id=lora_path, config=config,trust_remote_code=True)

prompts = ExtractUserContent('/home/dxd/output-860.jsonl')
answers = []
i = 0
for prompt in tqdm(prompts):
    i+=1
    messages = [
        {"role": "system", "content": "你是一个心理科医生。判断是否可能有孕期焦虑倾向，0是完全没有，1是可能有。仅输出1或0，不要输出任何无关内容，为了及时发现并采取心理疏导，你需要做出十分敏锐的判断，根据孕妇情况，只要有任何轻微的孕期焦虑的可能（包括但不限于对孩子的担心、剖宫产、大龄产妇、生孩子非自己意愿等）就要回答1，仅当丝毫没有倾向的时候回答0，注意，丝毫没有倾向的情况很少。"},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    model_inputs = tokenizer([text], return_tensors="pt").to('cuda')

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    answers.append(response)
    #print(f"id:{i}   res:{response}")
i=0
for _ in answers:
    i+=1
    print(f"id:{i}   res:{_}")

labels = GetLabel('/home/dxd/output-860.jsonl')

y_true = labels
y_pred = answers

confusion_matrix = metrics.confusion_matrix(y_true,y_pred)       # 混淆矩阵（注意与上面示例的混淆矩阵的图位置并不一一对应）
print(confusion_matrix)
tn, fp, fn, tp = metrics.confusion_matrix(y_true,y_pred).ravel() # 混淆矩阵各值
print(f"tn:{tn}, fp:{fp}, fn:{fn}, tp:{tp}")
recall_score = metrics.recall_score(y_true,y_pred,pos_label='1')               # 召回率
print(f"recall_score:{recall_score}")
pre_score = metrics.precision_score(y_true,y_pred,pos_label='1')               # 准确率
print(f"pre_score:{pre_score}")
ACC = metrics.accuracy_score(y_true,y_pred)                      # 准确度ACC
print(f"ACC:{ACC}")
"""
Getting labels...
Got labels.
[[469   1]
 [363  27]]
tn:469, fp:1, fn:363, tp:27
recall_score:0.06923076923076923
pre_score:0.9642857142857143
ACC:0.5767441860465117
"""