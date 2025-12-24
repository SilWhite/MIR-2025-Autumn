from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

hyde_prompt = """Please write a passage to answer the question.
Question: {query}
Passage: """


def load_queries(queries_path="../datas/webq-test.txt"):
    queries = []
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(line)
    print(f"Loaded {len(queries)} queries from {queries_path}")
    return queries


class LMAPI:
    def __init__(self):
        self.model_name = "deepseek-chat"
        self.key = "sk-612f9f0c0f6e427c9ed0bd28ffa51e82"
        self.url = "https://api.deepseek.com/v1"

    def __get_message(self, user_prompt: str):
        msg = []
        msg.append({"role": "system", "content": "You are a helpful assistant."})
        msg.append({"role": "user", "content": user_prompt})
        return msg

    def get_llm_rst(self, user_prompt: str, **kwargs):
        client = OpenAI(api_key=self.key, base_url=self.url)
        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=self.__get_message(user_prompt),
                **kwargs,
            )
        except Exception as e:
            raise e
        return response.choices[0].message.content.strip()


def query2hyde_passage(query: list[str], lm_api: LMAPI, concat=False) -> list[str]:
    passages = [""] * len(query)
    with ThreadPoolExecutor() as executor:
        future_to_query = {
            executor.submit(
                lm_api.get_llm_rst,
                hyde_prompt.format(query=q),
                max_tokens=512,
                temperature=0.7,
            ): idx
            for idx, q in enumerate(query)
        }
        for future in tqdm(
            as_completed(future_to_query),
            total=len(query),
            desc="Generating HyDE passages",
            leave=False,
        ):
            idx = future_to_query[future]
            try:
                rst = future.result()
                passages[idx] = rst if not concat else query[idx] + "\n" + rst
            except Exception as e:
                print(f"Error generating passage for query '{query[idx]}': {e}")
                passages[idx] = ""
    return passages
