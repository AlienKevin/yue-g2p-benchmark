from typing import List
import csv
from collections import defaultdict
import ahocorasick
from typing import Optional
from models.G2PModel import G2PModel
import json

class LLMG2PModel(G2PModel):
    def __init__(self):
        self.char_prs = defaultdict(list)
        with open('data/rime/char.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row[2] != '棄用':
                    self.char_prs[row[0]].append({'pr': row[1], 'freq': ','.join(column for column in row[2:5] if column)})
        
        self.phrase_prs = defaultdict(set)
        def load_phrase_prs(filename):
            with open(f'data/rime/{filename}.csv', 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    self.phrase_prs[row[0]].add(row[1])
        load_phrase_prs('word')
        load_phrase_prs('phrase_fragment')
        load_phrase_prs('fixed_expressions')
        with open('data/abccanto/abccanto.json', 'r') as f:
            dict = json.load(f)
            # Add alternative pronunciations from abccanto's headwords and variants
            for entry in dict:
                headword, headword_pr = zip(*entry['headword'])
                variants = [(headword, headword_pr)]
                for variant in entry['variants']:
                    variant_headword, variant_headword_pr = zip(*variant)
                    variants.append((variant_headword, variant_headword_pr))
                for variant, variant_pr in variants:
                    variant = ''.join(variant)
                    if '□' not in variant and '…' not in variant and len(variant) > 1:
                        self.phrase_prs[variant].add(' '.join(variant_pr))

            self.phrase_automaton = ahocorasick.Automaton()
            for phrase in self.phrase_prs:
                # print(phrase)
                self.phrase_automaton.add_word(phrase, phrase)
            self.phrase_automaton.make_automaton()

            # Add alternative pronunciations of phrase fragments as they appear in abccanto's example sentences
            for entry in dict:
                for pos in entry['poses']:
                    for sense in pos['senses']:
                        for example in sense['examples']:
                            text, prs = zip(*example)
                            # TODO: Handle English chunks that are longer than 1 character
                            if len(text) == len(''.join(text)):
                                text = ''.join(text)
                                for end_index, phrase in self.phrase_automaton.iter(text):
                                    start_index = end_index - len(phrase) + 1
                                    self.phrase_prs[phrase].add(' '.join(prs[start_index:end_index+1]))
        self.phrase_prs = {k: list(v) for k, v in self.phrase_prs.items()}

        self.phrase_automaton = ahocorasick.Automaton()
        for phrase in self.phrase_prs:
            self.phrase_automaton.add_word(phrase, phrase)
        self.phrase_automaton.make_automaton()

    def get_name(self) -> str:
        return "LLMG2P"

    def _predict(self, texts: List[str]) -> List[str]:
        def process_text(text):
            if text in self.phrase_prs:
                return ' '.join(self.phrase_prs[text])
            elif all(char in self.char_prs and len(self.char_prs[char]) == 1 for char in text):
                return ' '.join(self.char_prs[char][0]['pr'] for char in text)
            else:
                few_shots = [("一名出名嘅名字學家", "jat1 ming4 ceot1 meng2 ge3 ming4 zi6 hok6 gaa1"),
                             ("劉老師都會去都會大學喎", "lau4 lou5 si1 dou1 wui5 heoi3 dou1 wui6 daai6 hok6 wo5"),
                             ("行人行入行山銀行行房", "hang4 jan4 haang4 jap6 haang4 saan1 ngan4 hong4 hang4 fong4")]
                prompt = '\n\n'.join(self.build_prompt(text, pr) for (text, pr) in few_shots)
                prompt += '\n\n' + self.build_prompt(text)
                # print(prompt)
                return self.call_llm(prompt)

        from concurrent.futures import ThreadPoolExecutor
        from tqdm import tqdm
        with ThreadPoolExecutor() as executor:
            return list(tqdm(executor.map(process_text, texts), total=len(texts), desc="Processing texts"))

    def build_prompt(self, text: str, pr: Optional[str] = None) -> str:
        phrases = []
        for end_index, phrase in self.phrase_automaton.iter(text):
            start_index = end_index - len(phrase) + 1
            phrases.append((start_index, end_index+1, phrase))
        for char_index, char in enumerate(text):
            if char in self.char_prs:
                phrases.append((char_index, char_index+1, char))
        phrases.sort()

        phrases = list(dict.fromkeys(phrase for (start_index, end_index, phrase) in phrases))
        phrase_pr_pairs = []
        for phrase in phrases:
            phrase_prs = self.phrase_prs[phrase] if phrase in self.phrase_prs else [f'{pr["pr"]} ({pr["freq"]})' for pr in self.char_prs[phrase]]
            for phrase_pr in phrase_prs:
                phrase_pr_pairs.append(f"{phrase} -> {phrase_pr}")
        prompt = f"{'\n'.join(phrase_pr_pairs)}\nInput text:{text}\nOutput Jyutping:"
        if pr:
            prompt += pr
        return prompt
    
    def call_llm(self, text: str, base_url="https://api.openai.com", model="gpt-4o", max_attempts=10) -> str:
        from openai import OpenAI
        client = OpenAI(base_url=base_url)
        
        for attempt in range(max_attempts):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": text},
                    ],
                    stream=False
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt == max_attempts-1:  # Last attempt
                    raise e
                else:
                    print(e)
                continue


if __name__ == "__main__":
    model = LLMG2PModel()
    print(model._predict(["新加坡熱過香港", "可汗大點兵，軍書十二卷", "唔好唔記得喎!"]))
