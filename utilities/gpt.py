import logging
import os
import ast
import importlib
from . import config
import uuid
import json
import re
import random
from openai import AzureOpenAI
import requests
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored
from jinja2 import Template

logging.basicConfig(encoding='utf-8', level=logging.WARNING, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

class GPT():
    def __init__(self, endpoint, key, model, api_version):
        
        ''' Authenticate using Azure AD '''
        # token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
        # self.client = AzureOpenAI(
        #         api_version = api_version,
        #         azure_endpoint = endpoint,
        #         azure_ad_token_provider = token_provider,
        #     )

        self.azure_endpoint = endpoint
        self.api_key = key
        self.model = model
        self.api_version = api_version
        self.tools = []
        
        self.client = AzureOpenAI(
            azure_endpoint = endpoint, 
            api_key = key,  
            api_version = api_version
            )
        
        # Load agents, if provided
        if os.path.exists('agents'):
            for tool_path in os.listdir('agents'):
                if tool_path.endswith('.json'):
                    self.tools.append(json.load(open(os.path.join('agents', tool_path),'r')))
                    print(f'[INFO] Loaded agent: {tool_path}')


    def get_info(self):
        print('Endpoint:\t', self.azure_endpoint)
        print('API Key:\t', self.api_key[:5]+'*'*(len(self.api_key)-10)+self.api_key[-5:])
        print('Model:\t\t', self.model)
        print('API Version:\t', self.api_version)


    def http_request(self, method, base_url, query_params=None, headers=None, json_body=None):
        try:
            if method == 'POST':
                response = requests.post(base_url, params=query_params, headers=headers, json=json_body)

            elif method == 'PUT':
                response = requests.put(base_url, params=query_params, headers=headers, json=json_body)

            elif method == 'GET':
                response = requests.get(base_url, headers=headers, params=query_params)
            
            status_code = response.status_code
            response_body = response.json() if 'application/json' in response.headers.get('content-type') else response.text

        except Exception as e:
            status_code = 504
            response_body = str(e.args)
        
        return status_code, response_body


    def color_print(self, message, color='cyan'):
        print(colored(message, color))


    def preprocess_args(self, args):
        # get rid of newlines, nulls, tabs, etc.
        matches = ['\\n', 'null', '\\n', '\\t', '\\r', '\\f', '\\v', 'nan']
        pattern = '|'.join(map(re.escape, matches))
        args = re.sub(pattern, '', args)

        return ast.literal_eval(args)


    def execute_tools(self, tool_calls, tags):
        results = []
        
        for tool in tool_calls:

            # parse arguments
            func_name = tool['function']['name']
            args = self.preprocess_args(tool['function']['arguments'])

            # add custom tags into argument list
            if isinstance(tags, dict):
                args.update(tags)

            try:
                # dynamically load functions
                module = importlib.import_module(f'utilities.{func_name}')
                
                if hasattr(module, func_name):
                    self.color_print(f'[INFO] Agent activated -> {func_name} ({args})', 'green')
                    module_function = getattr(module, func_name)
                    outputs = module_function(**args)
                else:
                    raise Exception(f'{func_name} cannot be loaded or is not callable')
                    
            except Exception as e:
                self.color_print(f"Agent '{func_name}' not callable. ({str(e.args)})", 'red')
                outputs = {'value': [], 'status': str(e.args)}

            # schema: tool_outputs
            results.append({
                    'guid': str(uuid.uuid4()), 
                    'seed': random.randint(1,5),
                    'status': outputs['status'],
                    'function_name': func_name, 
                    'counts': len(outputs['value']) if 'value' in outputs.keys() else 0, 
                    'outputs': outputs
                    })

        return results


    def load_template(self, template_name):
        try:
            # load prompt variation from json
            with open(os.path.join(config.PROMPT_FOLDER, template_name+'.json')) as f:
                prompt_variation = json.load(f)
            
            # get prompt whose has key 'active'
            for prompt in prompt_variation:
                if prompt.get('active'):
                    prompt_version = template_name+'_'+str(prompt['version'])+'.jinja2'
                    self.color_print(f'[INFO] Activated prompt template: {prompt_version}', 'yellow')
                    break

            with open(os.path.join(config.PROMPT_FOLDER, prompt_version), 'r') as f:
                template = Template(f.read())
            
            return template
        
        except  Exception as e:
            logging.error(f'Error loading prompt template {template_name}: {e.args}')
            return None


    # @retry(wait=wait_random_exponential(multiplier=1, max=30), stop=stop_after_attempt(2))
    def chat(self, query, system_prompts, chat_history=[]):
        try:
            messages = []
            messages.append({'role':'system', 'content': system_prompts})
            
            if chat_history:
                messages + chat_history[-config.HISTORY_LIMIT:]
            
            if query:
                # messages.append({'role':'user', 'content': [{'type': 'text', 'text':query[:config.CHAR_LIMIT]}]})
                messages.append({'role':'user', 'content': query[:config.CHAR_LIMIT]})
            
            response = self.client.chat.completions.with_raw_response.create(
                model = self.model,
                # tools = self.tools,
                # tool_choice = "auto",
                messages = messages,
                max_tokens = 100,
                stream = False
            )

            # parse raw response
            status_code = response.status_code
            response_header = dict(response.headers.items())
            response_body = dict(response.parse())
            response_json = response.parse().model_dump_json()
            
            # pop 'choices' and convert 'usage' to dict for logging purpose
            response_body.pop('choices')
            response_body['usage'] = dict(response_body['usage'])
        
        except Exception as e:
            status_code = e.status_code
            response_header = dict(e.response.headers) if hasattr(e, 'response') else {}
            response_body = {'error': e.message}
        

        return status_code, response_header, response_body

        

if __name__ == '__main__':
    
    import random
    
    gpt = GPT()
    
    query_bank = [
        '今晩中華食べたいけど、何かおススメある？ 出来ればおいしい中華料理店も教えてほしいな。',
        '明日お会いする方とのディナーに何を作ればいいかな？',
        '今日は何を作ろうかな？渋谷付近でワインがおいしいレストラン教えて',
        '天気良さそうだから買い出しに行きたいけど、何かおススメある？',
        'Microsoftのストック今いくらだろう？',
    ]
    query = random.choice(query_bank)
    system_prompts = gpt.load_template('agent').render(NAME='Maya',ROLE='expert chef',LANGUAGE='Japanese')
    print(system_prompts)
    response = gpt.chat(model='gpt-35-turbo', 
                        system_prompts=system_prompts, 
                        query=query, 
                        chat_history=[])
    
    # inputs
    print('[User inputs]\n', query)
    
    # raw response
    print('[Raw outputs]\n', response)

    # pretty print message 
    message = response['message']
    print('[Dialogue Message]')
    
    # execute tools, if any
    if message.get('tool_calls') and message['tool_calls']:
        outputs = gpt.execute_tools(message['tool_calls'])
        print('[Tool outputs]\n', outputs)