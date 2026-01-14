import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from openai import OpenAI

#from fastapi import FastAPI



dataset = pd.read_csv("dataset.csv", sep="\t", names=["text", "label"], nrows=10000)

input_sententce = dataset["text"].apply(lambda x: " ".join(jieba.lcut(x)))  # sklearn对中文处理

vector = CountVectorizer()  # 对文本进行提取特征 默认是使用标点符号分词
vector.fit(input_sententce.values)
input_feature = vector.transform(input_sententce.values)

model = KNeighborsClassifier()
model.fit(input_feature, dataset["label"].values)

client = OpenAI(
    # 
    # https://bailian.console.aliyun.com/?tab=model#/api-key
    api_key="sk-20b5cf7cc6e04b2da387fdc81d7ead76",  # 账号绑定的

    # 大模型厂商的地址
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

app = FastAPI()

# @app.get("/")
async def root():
    return {"message": "欢迎使用FastAPI", "status": "running"}
# @app.get("/classify/ml")
def class_fy_using_ml(text : str) -> str:
    """
    文本分类 使用机器学习
    :param text: 待分类文本
    :return: 分类结果
    """
    # TODO: 实现文本分类

    test_sentence = " ".join(jieba.lcut(text))
    test_feature = vector.transform([test_sentence])
    return model.predict(test_feature)[0]

# @app.get("/classify/llm")
def class_fy_using_llm(text : str) -> str:
    """
    文本分类 使用大模型
    :param text: 待分类文本
    :return: 分类结果
    """
    # TODO: 实现文本分类

    completion = client.chat.completions.create(
        # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model="qwen-plus",  # 模型的代号

        messages=[
            {"role": "user", "content": f"{text}：请帮我对该文本进行分类，仅回答分类内容即可，从以下分类进行回答FilmTele-Play、Video-Play Music-Play、Radio-Listen、Alarm-Update、Travel-Query、HomeAppliance-Control、Weather-Query、Calendar-Query、TVProgram-Play、Audio-Play、Other"},  # 用户的提问
        ]
    )
    return completion.choices[0].message.content


if __name__ == '__main__':
    print(class_fy_using_llm("今天天气怎么样？"))
    print(class_fy_using_ml("我想去北京看看长城"))
