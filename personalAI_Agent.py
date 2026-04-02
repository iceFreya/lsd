"""项目目标：构建一个“个人网络研究员 Agent”

功能要求：

输入： 用户的一个模糊研究课题（例如“2024年大模型在医疗领域的应用进展”）
规划： Agent 自动将课题拆解为 3-5 个具体的搜索关键词
行动：
调用搜索工具
爬取网页内容
记忆与总结： 针对每个网页内容进行总结，剔除无关信息
输出： 汇总所有信息，写出一份带引用来源的 Markdown 简报，并保存到本地文件"""

import json
import time
from openai import OpenAI
from typing import List,Dict
from datetime import datetime
from ddgs import DDGS
from trafilatura import fetch_url,extract

DEEPSEEK_API_KEY = "sk-"
MODEL_NAME = "deepseek-chat"

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/v1"
)

MAX_RETRY = 3
RETURN_RESULT = 3
TIMEOUT = 120

def llm(prompt,temperature:float = 0.3,response_fm:str = "text"):
    response_format = None
    if response_fm == "json_object":
        response_format = {"type":"json_object"}

    for retry in range(MAX_RETRY):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role":"user","content":prompt}],
                temperature=temperature,
                response_format=response_format,
                timeout=TIMEOUT
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM调用失败，错误信息:{str(e)}重试{retry+1}/{MAX_RETRY}")
            time.sleep(1)
    raise Exception("LLM调用多次失败，流程终止")

def split(topic):
    prompt = f"""
    你是一位专业的学术研究助手，根据用户提供的一个模糊的研究课题，将其拆解为3-5个精准且适合搜索的关键词，确保覆盖课题的核心内容。
    要求：
    1. 仅输出JSON格式，结构为 {{"keywords": ["关键词1", "关键词2", "关键词3", "关键词4", "关键词5"]}}
    2. 关键词数量严格控制在3-5个
    3. 关键词必须贴合课题，适合搜索引擎检索，能获取到高相关的权威内容，不能过于宽泛或狭窄
    课题：{topic}
    """
    try:
        result = llm(prompt,temperature=0.3,response_fm="json_object")
        keywords = json.loads(result)["keywords"]

        if 3 <= len(keywords) <= 5:
            print(f"课题已拆解，关键词为:{keywords}")
            return keywords
        raise Exception("关键词数量不符合要求")
    except Exception as e:
        print(f"关键词拆解失败，错误信息:{str(e)}")
        return [topic]

def search(keywords):
    all_results = []
    url_sets = set()
    with DDGS() as ddgs:
        for keyword in keywords:
            try:
                print(f"正在搜索关键词:{keyword}")
                results = ddgs.text(
                    query=keyword,
                    max_results=RETURN_RESULT
                )
                for result in results:
                    url = result.get("href")
                    if url not in url_sets and url:
                        url_sets.add(url)
                        all_results.append({
                            "title": result.get("title", "无标题"),
                            "url": url,
                            "keywords": keyword
                        })
                time.sleep(1)
            except Exception as e:
                print(f"关键词{keyword}搜索失败，错误信息{str(e)}")
    print(f"\n搜索完成，共搜集到{len(all_results)}条链接")
    return all_results

def extract_information(url):
    try:
        html = fetch_url(url)
        if not html:
            return ""
        content = extract(html, include_links=False, include_images=False, include_tables=False)
        return content if content else ""
    except Exception as e:
        print(f"网页提取失败，错误信息:{str(e)}")
        return ""

def summarize(topic,content,url):
    if len(content) < 50:
        return {"is_valid":False,"summary":"","url":url,"title":""}
    prompt = f"""
    你是一位专业的研究内容提炼助手，需要对网页内容进行总结，只保留与研究课题【{topic}】高度相关的核心信息，剔除广告、无关新闻、冗余描述等无效内容。
    要求：
    1. 总结内容必须客观、准确，完全基于原文，不得编造信息
    2. 保留关键数据、核心观点、时间节点、权威结论
    3. 总结简洁精炼，重点突出，字数控制在100-300字
    4. 若原文内容与课题完全无关，仅输出"无关内容"
    网页原文内容：{content[:8000]}
"""
    try:
        summary = llm(prompt,temperature=0.3,response_fm="text")
        if "无关内容" in summary:
            return {"is_valid":False,"summary":"","url":url,"title":""}
        return {
            "is_valid":True,
            "summary":summary,
            "url":url,
            "title":""
        }
    except Exception as e:
        print(f"总结内容失败,错误信息：{str(e)},url:{url}")
        return {"is_valid":False,"summary":"","url":url,"title":""}

def generate(topic,summaries:List[Dict]):
    valid_summaries = [summary for summary in summaries if summary["is_valid"]]
    for idx,item in enumerate(valid_summaries,1):
        item["ref_id"] = idx
    summaries_text = "\n\n".join([
        f"[引用{idx}]标题:{item['title']}\n网页链接:{item['url']}\n内容:{item['summary']}"
        for idx, item in enumerate(valid_summaries,1)
    ])
    prompt = f"""
    你是一位专业的研究报告撰写助手，需要基于以下提炼的研究内容，生成一份结构化的Markdown格式研究简报，主题为【{topic}】。
    要求：
    1. 简报结构清晰，必须包含：课题概述、核心研究发现、分模块详细内容、引用来源四个固定部分
    2. 内容客观严谨，逻辑连贯，不能编造原文没有的信息
    3.语言专业精炼，符合研究简报的规范，避免口语化表达
    4. 所有核心信息必须标注引用编号，格式为[^序号]，与文末的引用来源一一对应
    5. 引用来源需按编号列出，包含序号、网页标题、原文链接
    已提炼的研究内容（含对应来源）：{summaries_text}
"""
    markdown_content = llm(prompt,temperature=0.3)
    return markdown_content

def save(topic,content):
    file_time = datetime.now().strftime("%Y%m%d%H%M%S")
    file_del = "".join([c for c in topic if c.isalnum() or c in (" ","-","_")]).replace(" ","_")
    file_name = f"{file_del}{file_time}.md"

    with open(file_name,"w",encoding="utf-8") as f:
        f.write(content)

    print(f"简报已保存到本地文件，文件名:{file_name}")
    return file_name

def research_agent(topic):
    print(f"启动个人网络研究员Agent,研究课题:{topic}")

    keywords = split(topic)
    results = search(keywords)
    if not results:
        print("未搜索到任何相关内容")
        return

    summaries = []
    for result in results:
        url = result["url"]
        print(f"正在检索网站:{result['title']}")
        content = extract_information(url)
        if not content:
            print("该网页无相关内容，跳过")
            continue
        summary_result = summarize(topic,content,url)
        if summary_result["is_valid"]:
            summary_result["title"] = result["title"]
            summaries.append(summary_result)
        time.sleep(1)
    if not summaries:
        print("未获取任何有效内容")
        return

    print(f"内容总结完毕，共获取到{len(summaries)}条内容")

    markdown_content = generate(topic,summaries)
    save(topic,markdown_content)
    print("任务完成!")

if __name__ == "__main__":
    user_topic = input("请输入你的研究课题")
    research_agent(user_topic)