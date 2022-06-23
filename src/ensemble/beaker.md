# Steps to running Beaker

### Need to be done once only:
- Step 1: install beaker on your machine: https://beaker.org/user
- Step 1.1: create a workspace: https://beaker.org/workspaces


### Few times, only when you change the data:
- Step 2: build beaker dataset for your data: `beaker dataset create --name NAMEEE data --workspace ai2/ensembles`

- Step 3: build docker image for the repo: `docker build -t ensembles .`
- Step 3.1: build beaker image for our docker image: `beaker image create -n ensembles35 ensembles --workspace ai2/ensembles`
    - If you already have an image with this name, delete the old one: `beaker image delete -n ensembles`


### the main items which is typically quick
- Step 4: set the parameters in the .yaml file (dataset name(s), image names (s), cluster name, parameters of your experiment, etc)
- Step 4.1: run the experiment on beaker: `beaker experiment create yaml_files/default_experiment.yaml --workspace ai2/ensembles`

### to collect the results as a group ...
 - create a group: 
   - `beaker  group  create goood_prompts --workspace ai2/ensembles`
   - `beaker  group  create natural_sentence_prompts --workspace ai2/ensembles`