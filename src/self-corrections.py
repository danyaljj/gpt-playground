import json
import os
import openai
import json
import copy
import random
import requests
import os


engines = [
    # "text-curie-001",
    # "text-davinci-002", # GPT3 instruct
    # "text-davinci-001", # GPT3 instruct
    # "text-curie-001",
    # "text-babbage-001",
    # "text-ada-001",
    # "text-curie-001",
    "davinci", # original GPT3
    # "curie",
    # "babbage",
    # "ada",
    # "curie",
]
engine = "davinci"

def query(prompt, engine):
    if not openai.api_key:
        openai.api_key = os.environ['OPENAI']

    print(prompt)
    response = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    # print(response)
    output = response['choices'][0]['text'].replace("\n", " ").replace("\t", " ")
    return output.strip()

def tkinstruct(input, definition):
    url = "http://pinot.cs.washington.edu:8000/api/predict"

    # reset first
    data = {
        'input': input,
        'def': definition,
        'pos-examples': json.dumps([]),
        'neg-examples': json.dumps([]),
    }
    x = requests.post(url, data=data)
    print(f" * definition: {definition}")
    print(f" * input: {input}")

    return json.loads(x.text)['response']


pconts = {
    "task001_quoref_question_generation": "In this task, you're given passages and a question. Modify the question such that it evaluate one's understanding of multiple mentions that refer to the same entity (e.g., people, places, or things). Good questions are expected to link pronouns (she, her, him, his, their, etc.) or other mentions to people, places, or things to which they may refer. Do not ask questions that can be answered correctly without understanding the paragraph or having multiple answers. Avoid questions that do not link phrases referring to the same entity. For each of your questions, the answer should be one or more phrases in the paragraph, and it should be unambiguous.",
}

pconnector = {
    "task001_quoref_question_generation":  "Question",
}


instructions_dir = "/Users/danielk/ideaProjects/natural-instructions-expansion2/natural-instructions/tasks/"
for task in pconts.keys():
    # read the file
    file = instructions_dir + task + ".json"
    with open(file) as f:
        print(" = = = = = ")
        jsoncontent = json.load(f)
        p0 = jsoncontent['Definition'][0]
        for ins in jsoncontent['Instances']:
            input = ins['input']
            output = query(p0 + "\n" + input, engine)
            # output = tkinstruct(input, p0)
            print(f"O0: {output}")
            for i in range(1, 5):
                print(" - - - - -")
                pcond = pconts[task]
                connector = pconnector[task]
                output = query(pcond + "\n" + f"{input} \n{connector}: {output[0]}", engine)
                # output = tkinstruct(f"{input} \n{connector}: {output[0]}", pcond)
                print(f"O{i}: {output}")

            break

    


