from ollama import Client, Model
import os, sys
import time
import subprocess
from rich import print as rprint

client = Client(host='http://localhost:11434')
models = client.list()
rprint(f'{len(models)} models')

try:
    # Run ollama command and capture output
    output, error = subprocess.run(['ollama'], capture_output=True)
except Exception as e:
    rprint(f'{e}')

for model_data in models['models']:
    try:
        model = Model(name=model_data['name'])  # Create a Model object for easier access
        rprint(f'-> {model.name}')
        res = model.delete()
        rprint(f'delete: {res}')
        rprint(f'downloading ', end='')
        subprocess.run(['ollama', 'pull', model.name])  # Use subprocess to pull the model_name
        for gen in model.pull(stream=True):
            rprint(".", end='')
            time.sleep(1)
    except Exception as e:
        rprint(f'exception: {e}')
rprint('done')