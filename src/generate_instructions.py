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
    "davinci",  # original GPT3
    # "curie",
    # "babbage",
    # "ada",
    # "curie",
]
engine = "text-davinci-002"


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


instructions_dir = "/Users/danielk/ideaProjects/natural-instructions-expansion2/natural-instructions/tasks/"

# list all files
files = os.listdir(instructions_dir)
tasks = []
for file in files:
    if ".json" in file:
        tasks.append(file)


def load_task(file):
    with open(instructions_dir + file) as f:
        return json.load(f)

examples_count = 3
task_count = 3

def encode_example(input, output):
    return f"""* Input: {input} \n* Output: {output}\n\n"""

def encode_task(task):
    examples = ""
    for example in task['Instances'][:examples_count]:
        examples += encode_example(example["input"], example["output"][0])

    return f"""Consider these examples: \n{examples} \nHere is the definition of this task: {task['Definition'][0]}"""

import csv
csvfile = open('tasks.csv', 'w')
# add the headers
fieldnames = ['t1', 't2', 't3', 'ex', 'gpt-output']
writer = csv.DictWriter(csvfile, fieldnames=fieldnames)


for _ in range(0, 20):
    data = {}
    print(" ============= ")
    encoded_prompt = ""
    # randomly select a subset of the tasks
    for i, task in enumerate(random.sample(tasks, task_count)):
        encoded_task = encode_task(load_task(task))
        # print(encoded_task)
        encoded_prompt += encoded_task + "\n\n"
        data[f't{i+1}'] = encoded_task

    # random examples
    random_examples = []
    encoded_prompt += f"""Consider these examples: \n"""
    encoded_examples = ""
    for task in random.sample(tasks, examples_count):
        task = load_task(task)
        instances = task['Instances']
        # select one random instance
        instance = random.choice(instances)
        # encoded_prompt += encode_example(instance["input"], instance["output"][0])
        encoded_examples += encode_example(instance["input"], instance["output"][0])

    encoded_prompt += f"{encoded_examples}\nHere is the definition of this task:"
    print(encoded_prompt)
    data[f'ex'] = encoded_examples

    try:
        output = query(encoded_prompt, engine)

        print("GPT output: " + output)
        if "Consider these examples" in output:
            # drop the text after 'Consider these examples'
            output = output.split("Consider these examples")[0]

        # f.write(encoded_prompt + "\t" + output + "\n")
        # data[f'prompt'] = encoded_prompt
        data['gpt-output'] = output
        writer.writerow(data)
    except Exception as e:
        print(e)
        continue


