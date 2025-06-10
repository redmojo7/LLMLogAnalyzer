from pathlib import Path
from prompts import rag_system_prompt
from settings import Settings
from llama_index.core.base.response.schema import StreamingResponse
from router import router_query, partial_router_query
from utils import search, search_events, get_template_path, get_structured_log_path
import time
import pandas as pd
import os

def create_prompt(query_str):
  prompt_template = ("""
      <|begin_of_text|><|start_header_id|>system<|end_header_id|>

      {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

      {query_str}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
      """)
  return prompt_template.format(system_prompt=rag_system_prompt, query_str=query_str)


def generate_questions(question, engines):

    log_path = engines['log_path']
    log_file_name = Path(log_path).name

    # Read log file
    with open(log_path, 'r') as f:
        log_lines = f.readlines()

    templates_path = get_template_path(log_path)

    templates_file = open(templates_path, "r")
    templates = templates_file.readlines()

    new_summary_tmpl_str = (
        rag_system_prompt + "\n"
        "Context information is below.\n"
        "---------------------\n"
        f"Log file name {log_file_name}\n"
        f"All the events in this log file, which is a csv file, contain three columns: EventId,EventTemplate,Occurrences\n"
        f"{templates[1:]}\n"
        f"The first line of the log file: {log_lines[0].strip()}\n"
        f"The Last line of the log file: {log_lines[-1].strip()}\n"
        "---------------------\n"
        f"The log file contains {len(log_lines)} lines\n"
        f"There are {len(templates)} log events in the log file.\n"
        "The log period can be indicated in the first and last lines of the log file.\n"
        "Given the context information, answer the query.\n"
        f"Query: {question}\n"
        "Answer: "
    )

    print(f"new_summary_tmpl_str: {new_summary_tmpl_str}")

    #response = groq_llama3_70b_llm.stream_complete(new_summary_tmpl_str)
    response = Settings.llm.stream_complete(new_summary_tmpl_str)

    return response


def summary(question, engines):
    """Handle the summary and reanswer the question with the log events."""
    #

    log_path = engines['log_path']
    log_file_name = Path(log_path).name

    # Read log file
    with open(log_path, 'r') as f:
        log_lines = f.readlines()

    templates_path = get_template_path(log_path)

    # Query log templates with user's question
    templates_file = open(templates_path, "r")
    templates = templates_file.readlines()
    print(f"Length of templates {len(templates)}")
    print(f"templates: {templates}")

    new_summary_tmpl_str = (
        rag_system_prompt + "\n"
        "Context information is below.\n"
        "---------------------\n"
        f"Log file name {log_file_name}\n"
        f"All the events in this log file, which is a csv file, contain three columns: EventId,EventTemplate,Occurrences\n"
        f"{templates[1:]}\n"
        f"The first line of the log file: {log_lines[0].strip()}\n"
        f"The Last line of the log file: {log_lines[-1].strip()}\n"
        "---------------------\n"
        f"The log file contains {len(log_lines)} lines\n"
        f"There are {len(templates)} log events in the log file.\n"
        "The log period can be indicated in the first and last lines of the log file.\n"
        "Don't print template IDs as sentences. For every sentence you write, cite the template id as <template_id> using asterisks to bold template id in Markdown.\n"
        "Consider the log template and corresponding 'Occurrences' in context when analysing the context.\n"
        #"At the end of your commentary, write 'anomaly' log templates that you mentioned."
        "Given the context information, answer the query.\n"
        f"Query: {question}\n"
        "Answer: "
    )

    print(f"new_summary_tmpl_str: {new_summary_tmpl_str}")

    #response = groq_llama3_70b_llm.stream_complete(new_summary_tmpl_str)
    response = Settings.llm.stream_complete(new_summary_tmpl_str)

    context = { 
              "context": templates,
    }

    return response, context, None


def search_and_answer(res, question, engines):
    """Handle the search operation and reanswer the question with the search context."""
    keywords = res.get('keywords', [])
    print(f"Keywords: {keywords}")

    if not keywords:
        print("No keywords provided, and try answering the question directly.")
        response = Settings.llm.stream_complete(question)
        return response, None

    search_result = search(keywords, engines.get('nodes'))
    print(f"Length of search_result: {len(search_result)}")
    print(f"Search result: {search_result}")

     # check search_result
    search_result_modified = False

    MAX_LINES = 200

    if not search_result:
        print("No search result, and try answering the question directly.")
        response = Settings.llm.stream_complete(question)
    elif len(search_result) >= MAX_LINES:
        print("Search result is too long, and try trim it.")
        search_result_modified = True
        search_result_max = search_result[:MAX_LINES]
        print(f"search_result_max: {search_result_max}")


    search_prompt = (
        #"Answer the question with the context, and keep the answer concise."
        f"There are {len(search_result)} lines were matched by keywords: {keywords}.\n"
        f"{f'The context is too long, and it has been trimmed to {MAX_LINES} lines.' if search_result_modified else ''}\n"
        #f"Focus on the total number if the user asked a question about how many. For others, focus on context.\n"
        f"Focus on the total number if the user asks a question about how many.\n"
        f"Context: {search_result_max if search_result_modified else search_result}\n"
        f"Question: {question}."
    )

    prompt = create_prompt(search_prompt)

    print(prompt)

    response = Settings.llm.stream_complete(prompt)

    """
    response = Settings.llm.stream_complete(
        "Reanswer the question with the context, and keep the answer concise."
        f"{f'There are {len(search_result)} lines in the search result.' if not search_result_modified else 'The search result is too long, and it has been trimmed to 20 lines.'}"
        f"Question: {question}. Context: {search_result_20 if search_result_modified else search_result}"
    )
    """
    return response, search_result



def search_events_and_answer(res, question, engines):
    """Handle the search operation and reanswer the question with the search context."""
    events = res.get('events', [])
    print(f"Events: {events}")

    if not events:
        print("No events provided, and try answering the question directly.")
        response = Settings.llm.stream_complete(question)
        return response, None

    log_path = engines['log_path']
    template_path = get_template_path(log_path)

    df = pd.read_csv(template_path, header=0)

    # Assuming the templates file has a column named 'EventId' that contains the log lines
    # and a column named 'event' that contains the event names
    filtered_df = df[df['EventId'].isin(events)]

    structured_log_path = get_structured_log_path(log_path)

    search_result = search_events(events, log_path, structured_log_path)
    print(f"Length of search_result: {len(search_result)}")
    print(f"Search result: {search_result}")

     # check search_result
    search_result_modified = False

    MAX_LINES = 150

    if not search_result:
        print("No search result, and try answering the question directly.")
        response = Settings.llm.stream_complete(question)
    elif len(search_result) >= MAX_LINES:
        print("Search result is too long, and try trim it.")
        search_result_modified = True
        search_result_max = search_result[:MAX_LINES]
        print(f"search_result_max: {search_result_max}")


    search_prompt = (
        #"Answer the question with the context, and keep the answer concise."
        f"There are {len(search_result)} lines were matched by events: {events}.\n"
        f"{f'The context is too long, and it has been trimmed to {MAX_LINES} lines.' if search_result_modified else ''}\n"
        #f"Focus on the total number if the user asked a question about how many. For others, focus on context.\n"
        f"All relevant events are: {filtered_df.values.tolist()}"
        f"Focus on the total number if the user asks a question about how many.\n"
        f"Context: {search_result_max if search_result_modified else search_result}\n"
        f"Question: {question}."
    )

    prompt = create_prompt(search_prompt)

    print(prompt)

    response = Settings.llm.stream_complete(prompt)

    """
    response = Settings.llm.stream_complete(
        "Reanswer the question with the context, and keep the answer concise."
        f"{f'There are {len(search_result)} lines in the search result.' if not search_result_modified else 'The search result is too long, and it has been trimmed to 20 lines.'}"
        f"Question: {question}. Context: {search_result_20 if search_result_modified else search_result}"
    )
    """


    return response, search_result


def query_with_engines_partial(question, engines):
  """Query the appropriate stage based on the routing result and measure time taken for the operation."""
  start_time = time.time()

  try:
      # Get routing choice and act accordingly
      res = partial_router_query(question)
      choice = res["choice"]
      print(f"[Level 2 Router: Partial] Routing to '{choice}' stage")

      keywords = res.get('keywords', [])
      print(f"keywords: {keywords}")

      events = res.get('events', [])
      print(f"events: {events}")

      # Define actions for each choice
      actions = {
          #"anomaly": lambda: anomaly(res, question, engines),
          "retrieve": lambda: (engines.get('retrieve').query(question), None),
          "search": lambda: search_and_answer(res, question, engines),
          "event": lambda: search_events_and_answer(res, question, engines)
      }

      # Get response based on choice
      response, context = actions.get(choice, lambda: (Settings.llm.stream_complete(question), None))()

      context = { 
              "context": context, 
              "keywords": keywords,
              "events": events,
      }

  except Exception as e:
      # Log the error and provide a fallback response
      print(f"Error: {e}")
      response = Settings.llm.stream_complete(question)
      context = None

  end_time = time.time()
  print(f"Time taken: {end_time - start_time:.2f} seconds.\nLLM:", end=" ")

  return response, context, choice


def query_with_engines(question, engines):
    """Query the appropriate stage based on the routing result and measure time taken for the operation."""
    start_time = time.time()

    second_choice = None

    try:
        # Print user question
        print("-" * 50)
        print(f"User: {question}")
        print("-" * 50)

        # Get routing choice and act accordingly
        res = router_query(question)
        choice = res["choice"]
        print(f"[Level 1 Router] Routing to '{choice}' stage")

        # Define actions for each choice
        actions = {
            "generation": lambda: (Settings.llm.stream_complete(create_prompt(question)), None, None),
            #"all": lambda: (engines.get('summary').query(question), None, None),
            "all": lambda: summary(question, engines),
            "partial": lambda: query_with_engines_partial(question, engines)
        }

        # Get response based on choice
        response, context, second_choice = actions.get(choice, lambda: (Settings.llm.stream_complete(question), None, None))()

    except Exception as e:
        # Log the error and provide a fallback response
        print(f"Error: {e}")
        response = Settings.llm.stream_complete(question)
        context = None

    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds.\nLLM:", end=" ")

    if second_choice is not None:
        choice = choice + " > " + second_choice

    return choice, response, context