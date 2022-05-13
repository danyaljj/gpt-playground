import copy
import subprocess
import yaml
import time

with open("yaml_files/default_experiment.yaml", 'r') as f:
    default_yaml = f.read()

cluster = "ai2/mosaic-cirrascale"
# cluster = "ai2/aristo-cirrascale"
# cluster = "ai2/s2-cirrascale"
# cluster = "ai2/karenq-a100-cluster_90"
# cluster = "ai2/general-cirrascale"
# cluster = "ai2/allennlp-cirrascale"
# cluster = "ai2/danielk-a100-cluster-30-preemtible"
# cluster = "ai2/danielk-a100-cluster-50"
# cluster = "ai2/danielk-a100-cluster-10"

multiberts = [ f'google/multiberts-seed_{idx}' for idx in range(0, 25) ]

models = [
    # "bert-base-cased",
    # "bert-large-cased",
    # "bert-base-uncased",
    "bert-large-uncased",
    # 'gpt2',
    # 'gpt2-medium',
    # 'gpt2-large',
    # 'gpt2-xl',
    # 'EleutherAI/gpt-neo-1.3B',
    # 'EleutherAI/gpt-neo-2.7B',
    # 'EleutherAI/gpt-j-6B',
    # 'danyaljj/gpt-j-6B-step-318500',
    # 'danyaljj/gpt-j-6B-step-328500',
    # 'danyaljj/gpt-j-6B-step-338500',
    # 'danyaljj/gpt-j-6B-step-348500',
    # 'danyaljj/gpt-j-6B-step-358500',
    # 'danyaljj/gpt-j-6B-step-378500',
    # 'danyaljj/gpt-j-6B-step-384500',
    # 'danyaljj/gpt-j-6B-step-384000',
    # 'danyaljj/gpt-j-6B-step-383500',
    # 'danyaljj/gpt-j-6B-step-383000',
    # 'danyaljj/gpt-j-6B-step-382500',
    # 'stanford-crfm/eowyn-gpt2-medium-x777',
    # 'stanford-crfm/durin-gpt2-medium-x343',
    # 'stanford-crfm/beren-gpt2-medium-x49',
    # 'stanford-crfm/celebrimbor-gpt2-medium-x81',
    # 'stanford-crfm/arwen-gpt2-medium-x21',
    # 'roberta-base',
    # 'roberta-large',
]


train_sizes = [
    -1
]

epochs = [
    9, 11, 13
]

learning_rates = [
    2e-5 , 1e-5, 4e-5 # 5e-5, 3e-5,
]

num_models = [
     # 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    1, 2, 4, 8, 16, 25
    # 4, 8
]

non_linearity = [
    True, False
    # False
]


datasets = [
    # "copa",
    # "boolq",
    "mrpc",
    # "hellaswag",
    # "swag",
    # 'arc_easy',
    # 'arc_hard',
]


d1 = yaml.load(default_yaml)


for dataset in datasets:
    for model in models:
        for num in num_models:
            for train_size in train_sizes:
                for epoch in epochs:
                    for learning_rate in learning_rates:
                        for nl in non_linearity:

                            d = copy.deepcopy(d1)

                            batch_size = 2
                            # if num <= 3:
                            #     batch_size = 16
                            # elif num <= 6:
                            #     batch_size = 8
                            # elif num <= 10:
                            #     batch_size = 4
                            # elif num <= 15:
                            #     batch_size = 2
                            # else:
                            #     batch_size = 1

                            assert d['tasks'][0]['context']['cluster'] == "ai2/mosaic-cirrascale"
                            d['tasks'][0]['context']['cluster'] = cluster

                            name = f"experiment_train_size={train_size}-model={model.replace('/', '_')}-lr={learning_rate}-epoch={epoch}-num_models={num}-dataset={dataset}-non_linearity={nl}-batch_size={batch_size}"
                            d['description'] = name

                            task_idx = 3
                            assert d['tasks'][0]['arguments'][task_idx] == 'bert-base-cased'
                            d['tasks'][0]['arguments'][task_idx] = model

                            task_idx = 5
                            assert d['tasks'][0]['arguments'][task_idx] == 1000, d['tasks'][0]['arguments'][task_idx]
                            d['tasks'][0]['arguments'][task_idx] = train_size

                            task_idx = 7
                            assert d['tasks'][0]['arguments'][task_idx] == 5, d['tasks'][0]['arguments'][task_idx]
                            d['tasks'][0]['arguments'][task_idx] = epoch

                            task_idx = 9
                            assert d['tasks'][0]['arguments'][task_idx] == 0.001, d['tasks'][0]['arguments'][task_idx]
                            d['tasks'][0]['arguments'][task_idx] = learning_rate

                            task_idx = 11
                            assert d['tasks'][0]['arguments'][task_idx] == 1, d['tasks'][0]['arguments'][task_idx]
                            d['tasks'][0]['arguments'][task_idx] = num

                            task_idx = 13
                            assert d['tasks'][0]['arguments'][task_idx] == True, d['tasks'][0]['arguments'][task_idx]
                            d['tasks'][0]['arguments'][task_idx] = nl

                            task_idx = 15
                            assert d['tasks'][0]['arguments'][task_idx] == 'arc_easy', d['tasks'][0]['arguments'][task_idx]
                            d['tasks'][0]['arguments'][task_idx] = dataset

                            task_idx = 17
                            assert d['tasks'][0]['arguments'][task_idx] == 1, d['tasks'][0]['arguments'][task_idx]
                            d['tasks'][0]['arguments'][task_idx] = batch_size

                            fn = "yaml_files/{}.yaml".format(name)
                            file = open(fn, "w")
                            yaml.dump(d, file, default_flow_style=True)
                            file.close()

                            cmd = "beaker experiment create {} --workspace ai2/ensembles".format(fn)
                            subprocess.Popen(cmd, shell=True)
                            time.sleep(0.5)