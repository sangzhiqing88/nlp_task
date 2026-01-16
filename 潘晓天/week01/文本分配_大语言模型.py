from openai import OpenAI

client = OpenAI(

    api_key="sk-c034bcc778xxxxx9940ed402", # 账号绑定，用来计费的

    # 大模型厂商的地址，阿里云
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def text_calssify_using_llm(text: str) -> str:
    """
    文本分类（大语言模型），输入文本完成类别划分
    """
    completion = client.chat.completions.create(
        model="qwen-flash",  # 模型的代号

        messages=[
            {"role": "user", "content": f"""帮我进行文本分类：{text}

输出的类别只能从如下中进行选择， 除了类别之外下列的类别，请给出最合适的类别。
FilmTele-Play            
Video-Play               
Music-Play              
Radio-Listen           
Alarm-Update        
Travel-Query        
HomeAppliance-Control  
Weather-Query          
Calendar-Query      
TVProgram-Play      
Audio-Play       
Other             
"""},  # 用户的提问
        ]
    )
    return completion.choices[0].message.content

if __name__ == "__main__":
    PURP = input("请输入意图：")
    # pandas 用来进行表格的加载和分析
    # numpy 从矩阵的角度进行加载和计算
    print("大语言模型: ", text_calssify_using_llm(PURP))

