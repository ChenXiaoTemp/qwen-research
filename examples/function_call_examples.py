# Reference: https://openai.com/blog/function-calling-and-other-api-updates
import json

import openai

# To start an OpenAI-like Qwen server, use the following commands:
#   git clone https://github.com/QwenLM/Qwen-7B;
#   cd Qwen-7B;
#   pip install fastapi uvicorn openai pydantic sse_starlette;
#   python openai_api.py;
#
# Then configure the api_base and api_key in your client:
openai.api_base = "http://172.26.1.44:3335/v1"
openai.api_key = "none"


def call_qwen(messages, functions=None):
    print(messages)
    if functions:
        response = openai.ChatCompletion.create(
            model="Qwen", messages=messages, functions=functions
        )
    else:
        response = openai.ChatCompletion.create(model="Qwen", messages=messages)
    print(response)
    print(response.choices[0].message.content)
    return response


def test_1():
    messages = [{"role": "user", "content": "你好"}]
    call_qwen(messages)
    messages.append({"role": "assistant", "content": "你好！很高兴为你提供帮助。"})

    messages.append({"role": "user", "content": "给我讲一个年轻人奋斗创业最终取得成功的故事。故事只能有一句话。"})
    call_qwen(messages)
    messages.append(
        {
            "role": "assistant",
            "content": "故事的主人公叫李明，他来自一个普通的家庭，父母都是普通的工人。李明想要成为一名成功的企业家。……",
        }
    )

    messages.append({"role": "user", "content": "给这个故事起一个标题"})
    call_qwen(messages)


def test_2():
    functions = [
        {
            "name_for_human": "谷歌搜索",
            "name_for_model": "google_search",
            "description_for_model": "谷歌搜索是一个通用搜索引擎，可用于访问互联网、查询百科知识、了解时事新闻等。"
            + " Format the arguments as a JSON object.",
            "parameters": [
                {
                    "name": "search_query",
                    "description": "搜索关键词或短语",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
        },
        {
            "name_for_human": "文生图",
            "name_for_model": "image_gen",
            "description_for_model": "文生图是一个AI绘画（图像生成）服务，输入文本描述，返回根据文本作画得到的图片的URL。"
            + " Format the arguments as a JSON object.",
            "parameters": [
                {
                    "name": "prompt",
                    "description": "英文关键词，描述了希望图像具有什么内容",
                    "required": True,
                    "schema": {"type": "string"},
                }
            ],
        },
    ]

    messages = [{"role": "user", "content": "你好"}]
    call_qwen(messages, functions)
    messages.append(
        {"role": "assistant", "content": "你好！很高兴见到你。有什么我可以帮忙的吗？"},
    )

    messages.append({"role": "user", "content": "谁是周杰伦"})
    call_qwen(messages, functions)
    messages.append(
        {
            "role": "assistant",
            "content": "Thought: 我应该使用Google搜索查找相关信息。",
            "function_call": {
                "name": "google_search",
                "arguments": '{"search_query": "周杰伦"}',
            },
        }
    )

    messages.append(
        {
            "role": "function",
            "name": "google_search",
            "content": "Jay Chou is a Taiwanese singer.",
        }
    )
    call_qwen(messages, functions)
    messages.append(
        {
            "role": "assistant",
            "content": "周杰伦（Jay Chou）是一位来自台湾的歌手。",
        },
    )

    messages.append({"role": "user", "content": "他老婆是谁"})
    call_qwen(messages, functions)
    messages.append(
        {
            "role": "assistant",
            "content": "Thought: 我应该使用Google搜索查找相关信息。",
            "function_call": {
                "name": "google_search",
                "arguments": '{"search_query": "周杰伦 老婆"}',
            },
        }
    )

    messages.append(
        {"role": "function", "name": "google_search", "content": "Hannah Quinlivan"}
    )
    call_qwen(messages, functions)
    messages.append(
        {
            "role": "assistant",
            "content": "周杰伦的老婆是Hannah Quinlivan。",
        },
    )

    messages.append({"role": "user", "content": "给我画个可爱的小猫吧，最好是黑猫"})
    call_qwen(messages, functions)
    messages.append(
        {
            "role": "assistant",
            "content": "Thought: 我应该使用文生图API来生成一张可爱的小猫图片。",
            "function_call": {
                "name": "image_gen",
                "arguments": '{"prompt": "cute black cat"}',
            },
        }
    )

    messages.append(
        {
            "role": "function",
            "name": "image_gen",
            "content": '{"image_url": "https://image.pollinations.ai/prompt/cute%20black%20cat"}',
        }
    )
    call_qwen(messages, functions)


def test_3():
    functions = [
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        }
    ]

    messages = [
        {
            "role": "user",
            # Note: The current version of Qwen-7B-Chat (as of 2023.08) performs okay with Chinese tool-use prompts,
            # but performs terribly when it comes to English tool-use prompts, due to a mistake in data collecting.
            "content": "波士顿天气如何？",
        }
    ]
    call_qwen(messages, functions)
    messages.append(
        {
            "role": "assistant",
            "content": None,
            "function_call": {
                "name": "get_current_weather",
                "arguments": '{"location": "Boston, MA"}',
            },
        },
    )

    messages.append(
        {
            "role": "function",
            "name": "get_current_weather",
            "content": '{"temperature": "22", "unit": "celsius", "description": "Sunny"}',
        }
    )
    call_qwen(messages, functions)


def sets_different(set1, set2):
    missed_items = ','.join(set1 - set2)
    extra_items = ','.join(set2 - set1)
    if len(missed_items) > 0 and len(extra_items) > 0:
        return "缺少参数:" + missed_items + ",多出参数:" + extra_items
    elif len(missed_items) > 0:
        return "缺少参数:" + missed_items
    elif len(extra_items) > 0:
        return "多出参数:" + extra_items
    else:
        return None


def check_arguments(function_name, response_arguments, expected_arguments):
    try:
        print("Function,",function_name,response_arguments)
        res = json.loads(response_arguments)
        keys1 = set(res.keys())
        keys2 = set(expected_arguments.keys())
        diff = sets_different(keys1, keys2)
        if diff is not None:
            return "期待的参数，不匹配," + diff
        return "参数正确"
    except:
        return "返回了错误的参数格式"


def test_4(query_content, expected_function_call_name, expected_arguments=None):
    functions = [
        {
            "name": "real_estate_recommendation",
            "description": "房源推荐",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用户输入的问题"
                    }
                },
                "required": ["query"],
            },
        },
        {
            "name": "loan_calculator",
            "description": "贷款计算",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用户输入的问题"
                    }
                },
                "required": ["query"],
            },
        },
        {
            "name": "policy_analyzer",
            "description": "政策分析",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用户输入的问题"
                    }
                },
                "required": ["query"],
            }
        },
        {
            "name": "real_estate_detail",
            "description": "房源详情",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用户输入的问题"
                    }
                },
                "required": ["query"],
            }
        },
        {
            "name": "real_estates_comparison",
            "description": "房源对比",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "房源名称"
                    }
                },
                "required": ["items"],
            }
        }
    ]

    messages = [
        {
            "role": "user",
            # Note: The current version of Qwen-7B-Chat (as of 2023.08) performs okay with Chinese tool-use prompts,
            # but performs terribly when it comes to English tool-use prompts, due to a mistake in data collecting.
            "content": query_content,
        }
    ]
    response = call_qwen(messages, functions)
    if response.choices[0].message.function_call is not None:
        if expected_function_call_name == response.choices[0].message.function_call.name:
            argument_matched = check_arguments(response.choices[0].message.function_call.name, response.choices[0].message.function_call.arguments, expected_arguments)
            return "函数正确", argument_matched
        else:
            return "函数错误", ""


if __name__ == "__main__":
    print("### 房产聊天助手 ###")
    question_and_answers = [
        {
            "question": "我预算100万，想要一套100平米以上，周边有医院和商场的房子，你能帮我推荐一下吗?",
            "function": {
                "name": "real_estate_recommendation",
                "arguments": {
                    "query": ["预算100万", "100平米以上", "医院", "商场"]
                }
            }
        },
        {
            "question": "请帮我介绍下云山诗意的详细信息",
            "function": {
                "name": "real_estate_detail",
                "arguments": {
                    "query": ["云山诗意"]
                }
            }
        },
        {
            "question": "岸头佳园65万的这套房子和云山诗意91万的这套房子对比，有哪些劣势?",
            "function": {
                "name": "real_estates_comparison",
                "arguments": {
                    "items": ["岸头佳园", "云山诗意"]
                }
            }
        },
        {
            "question": "买房子贷款多少年划算？",
            "function": {
                "name": "policy_analyzer",
                "arguments": {
                    "query": ["卖掉", "最新政策"]
                }
            }
        },
        {
            "question": "我想买一套总价150万的房子，你能帮我算下贷款吗?",
            "function": {
                "name": "loan_calculator",
                "arguments": {
                    "query": ["150万", "贷款"]
                }
            }
        }
    ]

    res = []

    for q_a in question_and_answers[3:4]:
        res.append(test_4(q_a["question"], q_a["function"]["name"], q_a["function"]["arguments"]))

    for q_a, r in zip(question_and_answers, res):
        print("Question:", q_a["question"])
        print("Answer:", r)
