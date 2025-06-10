from config import benchmark_settings
import os
import json
from utils import read_random_lines_from_large_log
from settings import Settings
from drain import LogParser
from pathlib import Path
import pandas as pd


# Define a custom prompt template
CATEGORY_PROMPT_TEMPLATE = (
    """Categorize the provided log lines into one of the categories:{categories}, assuming those logs share a single category.
Return a JSON object with a single key "category" without any preamble, special characters, or explanation.\n
Log:\n{log}\n"""
)

def log_categories_query(log):
    """Identify the category of the log that user provided."""
    # patterns = "\n".join([f"{category}:\nLog Format: {benchmark_settings[category]['log_format']} \nRegex: {benchmark_settings[category]['regex']}" for category in LOG_CATEGORIES])

    prompt = CATEGORY_PROMPT_TEMPLATE.format(log=log, categories=list(benchmark_settings.keys()))

    print(f"Prompt: {prompt}")
    #response = groq_llama3_70b_llm.complete(prompt)
    response = Settings.llm.complete(prompt)

    print(str(response))
    return json.loads(str(response))['category'] # Output example: {'category': 'Mac'}


# log paring with Drain
def log_paring(categorie, log_path, output_dir):
    log_format = benchmark_settings[categorie]['log_format']
    #print(f"log path: {log_path}")
    input_dir = Path(log_path).parent
    print(f"input_dir: {input_dir}")
    print(f"output_dir: {output_dir}")
    log_file_name = Path(log_path).name
    print("log_file_name: ", log_file_name)
    regex = benchmark_settings[categorie]['regex']
    st = benchmark_settings[categorie]['st']
    depth = benchmark_settings[categorie]['depth']

    parser = LogParser(log_format, indir=input_dir, outdir=output_dir,  depth=depth, st=st, rex=regex)
    return parser.parse(log_file_name)


def retry_log_parsing(log_path, output_dir, max_retries=3):
    retries = 0
    while retries < max_retries:
        try :
            # Step 1: Identify log category
            lines = read_random_lines_from_large_log(log_path, 10)
            print(f"Lines: \n{''.join(lines)}")
            identified_category = log_categories_query(''.join(lines))
            print(f"Identified Category: {identified_category}")
            structured_log_path, templates_path = log_paring(identified_category, log_path, output_dir)
            print(f"structured_log_path: {structured_log_path}, templates_path: {templates_path}")
            # Check if templates_path exists and is not empty
            if os.path.exists(templates_path) and os.path.getsize(templates_path) > 0:

                log_length = len(open(log_path, "r").readlines())
                structured_log_length = pd.read_csv(structured_log_path, header=0).shape[0]
                print(f"log_length: {log_length}, structured_log_length: {structured_log_length}")

                if structured_log_length != log_length:
                    print("Warning: Structured log file has fewer rows than the original log file.")
                    retries += 1
                else:
                    print("Log parsing successful.")
                return True
            else:
                print("Templates file does not exist or is empty, retrying log parsing...")
                retries += 1
        except Exception as e:
            print(f"Error: {e}")
            retries += 1
    print("Log parsing failed after {} retries.".format(max_retries))
    return False