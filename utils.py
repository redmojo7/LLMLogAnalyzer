from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from settings import Settings
from llama_index.core import SummaryIndex, VectorStoreIndex, DocumentSummaryIndex
from llama_index.core import PromptTemplate
from prompts import rag_system_prompt
from llama_index.core.schema import TextNode
from typing import List
from pathlib import Path
import pandas as pd
import os
import shutil
import random

import os
from pathlib import Path

def get_output_file_path(log_path: str, suffix: str) -> str:
    """
    Returns the path to an output file based on the input log path.

    Args:
        log_path (str): The path to the log file.
        suffix (str): The suffix to append to the log file name.

    Returns:
        str: The path to the output file.
    """
    log_file_name = Path(log_path).name
    output_dir = os.path.join(os.getcwd(), 'result')
    return os.path.join(output_dir, f"{log_file_name}_{suffix}.csv")

def get_template_path(log_path: str) -> str:
    """
    Returns the template path based on the input log path.

    Args:
        log_path (str): The path to the log file.

    Returns:
        str: The path to the template file.
    """
    return get_output_file_path(log_path, 'templates')

def get_structured_log_path(log_path: str) -> str:
    """
    Returns the structured log path based on the input log path.

    Args:
        log_path (str): The path to the log file.

    Returns:
        str: The path to the structured log file.
    """
    return get_output_file_path(log_path, 'structured')

def get_query_engines(file_path: str):
    """Get router query engine."""
    # load documents
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()

    splitter = SentenceSplitter(
        chunk_size=1024,
        chunk_overlap=0,
        paragraph_separator="\n"
    )
    nodes = splitter.get_nodes_from_documents(documents)

    print("[SentenceSplitter] Total number of nodes:", len(nodes))

    summary_index = SummaryIndex(nodes)
    vector_index = VectorStoreIndex(nodes)

    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
        streaming=True,
        llm=Settings.llm,
    )

    vector_query_engine = vector_index.as_query_engine(
        streaming=True,
    )

    new_index_tmpl_str = (
        rag_system_prompt + "\n"
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information, answer the query."
        "Query: {query_str}\n"
        "Answer: "
    )

    new_index_tmpl = PromptTemplate(new_index_tmpl_str)

    vector_query_engine.update_prompts(
        {"response_synthesizer:text_qa_template": new_index_tmpl}
    )

    # Save a copy on local disk
    name = os.path.basename(file_path)
    current_path = os.getcwd()
    local_file_path = os.path.join(current_path, name)
    shutil.copyfile(file_path, local_file_path)
    print(f"File saved to: {local_file_path}")

    # Clear old 'result' folder
    result_folder = os.path.join(current_path,'result')
    if os.path.exists(result_folder):
      shutil.rmtree(result_folder)
      print("Old 'result' folder cleared.")

    return {
        "nodes": nodes,
        "summary": summary_query_engine,
        "retrieve": vector_query_engine,
        "log_path": local_file_path,
    }
    


# Add Search tool
def search(keywords: List[str], nodes: List[TextNode]) -> List[str]:
    """Searches the query in the index and returns all matched results."""
    keyword_set = set(keywords)
    matched_lines = []

    for node in nodes:
        for line in node.text.split('\n'):
            if any(keyword.lower() in line.lower() for keyword in keyword_set):
                matched_lines.append(line.strip())

    return matched_lines


def search_events(events: List[str], log_path: str, structured_log_path: str) -> List[str]:
    """
    Searches for events in the log file and returns the corresponding log lines.

    Args:
        events (List[str]): A list of events to search for.
        log_path (str): The path to the log file.
        structured_log_path (str): The path to the structured log file.

    Returns:
        List[str]: A list of log lines that contain the searched events.
    """
    event_set = set(events)
    df = pd.read_csv(structured_log_path, header=0)

    # Assuming the structured log file has a column named 'EventId' that contains the log lines
    # and a column named 'event' that contains the event names
    filtered_df = df[df['EventId'].isin(event_set)]

    # Read the original log file
    with open(log_path, 'r') as f:
        log_lines = f.readlines()

    # Get the corresponding log lines using the filtered DataFrame's index
    filtered_log_lines = [log_lines[i].strip() for i in filtered_df.index]

    return filtered_log_lines


def read_random_lines_from_large_log(file_path, num_lines=10):
    with open(file_path, 'r') as file:
        reservoir = []
        for i, line in enumerate(file):
            if i < num_lines:
                reservoir.append(line)
            else:
                # Randomly replace elements in the reservoir with a decreasing probability
                r = random.randint(0, i)
                if r < num_lines:
                    reservoir[r] = line
    return reservoir

from llama_index.core.base.response.schema import StreamingResponse

def process_response(response):
    """Stream the response from the LLM."""
    output = ""
    if isinstance(response, StreamingResponse):
        for text in response.response_gen:
            output += text
            yield output
    else:
        for text in response:
            output += str(text.delta)
            yield output


import os
import pandas as pd

def get_event_logs(query_engines, template_ids):
    log_path = query_engines['log_path']
    log_file_name = Path(log_path).name
    current_path = os.getcwd()

    result_dir = os.path.join(current_path,'result')
    templates_path = os.path.join(result_dir, log_file_name+"_structured.csv")
    
    # check if the templates_path already exist?
    if os.path.exists(templates_path) and os.path.getsize(templates_path) > 0:
        print(f"templates_path exist: {templates_path}")
        df = pd.read_csv(templates_path)
        event_df = df[df['EventId'].isin(template_ids)]
        event_index_dict = event_df.groupby('EventId').apply(lambda x: x.index.tolist()).to_dict()

        with open(log_path, 'r') as f:
            lines = f.readlines()

        event_log_dict = {}
        for event_id, indexes in event_index_dict.items():
            event_log_dict[event_id] = [lines[index].strip() for index in indexes]

        return event_log_dict
    else:
        print("Templates file does not exist or is empty, skipping event logs extraction.")
        return None