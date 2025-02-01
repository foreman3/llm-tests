from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

class BatchEvaluator:
    def __init__(self, eval):
        self.system_prompt_start = """You are evaluating records. The user will submit a record to be evalated, and you should response will only the evalution. 
        Respond only with the evalution, and other terms or formatting.
        
        The requested evalution is:"""
        self.eval = eval
        self.client =  OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),  # This is the default and can be omitted
        )
        self.system_prompt = f"{self.system_prompt_start}\n{self.eval}"


    def process_batch(self, records):
        results = []
        for record in records:
            chat_completion = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"{record}"}
                ]
            )
            results.append(chat_completion.choices[0].message.content.strip())
        return results