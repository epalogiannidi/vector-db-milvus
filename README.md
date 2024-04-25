## Set up of a milvus db and creation of collections 

### Create environment:
```
    pipenv --python 3.12
    pipenv shell
    pipenv install --dev
```


### Set up milvus db through docker
View the commands in command.sh and execute in a terminal in the current working directory

### Adjust your data and configuration
- Update the config.yaml to match your needs
- Add your data within the data folder and/or update the configuration file

### Validate code formatting, styling and typehint issues:
```
    make lint
    make isort
    make black
    make mypy
```

### Development
- milvus_handler.py: The main class the handles the milvus db and its operations
- utils.py: Auxiliary functionality
- __main__.py: a wrapper to demonstrate how to employ the milvus handler. 


