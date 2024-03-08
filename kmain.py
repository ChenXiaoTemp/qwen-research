import openai

openai.api_key = "e766f8624bd94a3c9d1537657a43f246"
openai.api_base = "http://modelhub.4pd.io/learnware/models/openai/4pd/api/v1"

res = openai.ChatCompletion.create(
    model="public/qwen1-5-72b-chat-int4@main",
    messages=[{ "role": "user", "content": "你好,请介绍一下自己!" }],
    temperature=1,
    max_tokens=128,
    top_p=1,
    stop=None,
)

print(res)