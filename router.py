import json
from settings import Settings

# Define a custom prompt template
ROOT_ROUTER_TEMPLATE = (
    """You are an expert at routing user questions to the 'all', 'partial', or 'generation' stage.
To answer the user's question, you must analyze how the question relates to the log content and decide whether to retrieve all, partial, or no logs from the log file.

- Use 'all' for questions requiring the entire log file to answer it. Example: "Summarize the log.", "Generate a report based on the log file.", "Anomaly Detection"
- Use 'partial' for questions requiring only a specific part or chunk of the log file to answer it. Example: "What kind of log file is this?", "What happened at 02:04:59?", and "What is the event Event5?"
- Use 'generation' for questions that can be answered without needing to retrieve any logs. Example: "What is a brute force attack?", "Who are you?"

Return a JSON as plaintext with a single key 'choice' based on the question without any preamble, special characters, or explanation.

Question to route: {question} \n"""
)


def router_query(question):
    """Route a question to either generation, summary, or retrive stage"""
    print("router_query with ", question)
    response = Settings.llm.complete(ROOT_ROUTER_TEMPLATE.format(question=question))
    print(str(response))
    return json.loads(str(response)) # Output example: {'choice': 'generation'}


# Define a custom prompt template
PARTIAL_ROUTER_TEMPLATE = (
    """You are an expert at routing user questions to the 'search', 'event', or 'retrieve' stage.

- Use 'search' if the question requires using a search tool to find relevant information. If the question does not have clear keywords, try using 'retrieve' instead. Example: "What happened at 03:28:22?", "What is 'ALERT' in this log?"
- Use 'event' if the question is related to a specific event or log template in the log file. Example: "What is the event5?"
- Use 'retrieve' if the question asks for specific information that can be retrieved from a vector database. Example: "What kind of log file is this?", "What does each column in this log mean?"

Return a JSON as plaintext with a single key 'choice' based on the question without any preamble, special characters, or explanation. If 'search' is chosen, also return 'keywords' as a list.  If 'event' is chosen, also return 'events' as a list. 

Question to route: {question} \n"""
)


def partial_router_query(question):
    """Route a question to either generation, summary, or retrive stage"""
    print("partial_router_query with ", question)
    response = Settings.llm.complete(PARTIAL_ROUTER_TEMPLATE.format(question=question))
    print(str(response))
    return json.loads(str(response)) # Output example: {'choice': 'generation'}
