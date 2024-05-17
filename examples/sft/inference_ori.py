#first lora
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel,LoraConfig
from json_utils import ExtractUserContent,GetLabel
from tqdm import tqdm
from sklearn import metrics

model_path = '/home/dxd/qwen1.5-7b-chat/'
# lora_path = '/home/dxd/Qwen1.5/examples/sft/output_qwen/'
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
# model = PeftModel.from_pretrained(model, model_id=lora_path, config=config,trust_remote_code=True)

prompts = ExtractUserContent('/home/dxd/output-860.jsonl')
answers = []
i = 0
for prompt in tqdm(prompts):
    i+=1
    messages = [
        {"role": "system", "content": "通过给出的孕妇情况判断是否可能有孕期焦虑倾向，0是完全没有，1是可能有。仅输出1或0，不要输出任何无关内容，注意，为了及时发现并采取心理疏导，你需要做出十分敏锐的判断，根据孕妇情况，只要有任何轻微的孕期焦虑的可能（包括但不限于对孩子的担心、剖宫产、大龄产妇、生孩子非自己意愿等）就要回答1，仅当丝毫没有倾向的时候回答0，例如，孕妇情况如下：年龄34岁, 家庭月收入2000-5000 元, 剖宫产, 未患过心理疾患, 家族无遗传病, 想生孩子是父母的愿望, 认为孕前需做好生育知识准备, 已和爱人做好做父母的心理准备, 认为孩子不会影响和爱人之间的关系, 很在意未来宝宝性别, 担心能否生一个健康宝宝, 认为夫妇任一方精神高度紧张不会影响受孕, 认为过度紧张、焦虑、恐惧不会引起流产, 不认为不良的精神刺激和恐惧、焦虑抑郁情绪会影响胎儿发育, 不认为紧张会导致宫外孕, 目前感觉没有压力, 与爱人关系和睦, 与亲友的关系和睦, 与同事关系和睦，这个孕妇属于大龄孕妇且很在意宝宝性别，虽然很多内容都正常,但很担心宝宝健康，所以也要判断为有孕期焦虑倾向，所以你回答1.注意，丝毫没有倾向的情况很少。"},
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
[[228 242]
 [148 242]]
tn:228, fp:242, fn:148, tp:242
recall_score:0.6205128205128205
pre_score:0.5
ACC:0.5465116279069767
"""