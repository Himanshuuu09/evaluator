# %%
from langgraph.graph import Graph
from pydantic import BaseModel
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain_community.tools import TavilySearchResults
from typing import TypedDict
from langgraph.graph import END, START, StateGraph, MessagesState
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from typing import Union
import asyncio
import uvicorn

from fastapi import FastAPI
load_dotenv()

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"]=os.getenv("TAVILY_API_KEY")
llm=ChatGroq(model="llama3-groq-70b-8192-tool-use-preview")


# %%
class InputState(TypedDict):
    user_question:str
    user_answer:str
    gen_context:str
    question_type:str
class OutputState(TypedDict):
    feedback:str
    Total_Score:int
    Accuracy: int
    Clarity_and_Grammar: int
    Relevance_to_the_Question:int
    percentage:float
class OverallState(InputState,OutputState):
    pass


# %%
tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
    
)
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# %%
import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, chain

today = datetime.datetime.today().strftime("%D")
prompt = ChatPromptTemplate(
    [
        ("system", f"You are a helpful assistant. The date today is {today}."),
        ("human", "{user_input}"),
        ("placeholder", "{messages}"),
    ]
)

llm_with_tools = llm.bind_tools([tool,wikipedia])

llm_chain = prompt | llm_with_tools


@chain
def tool_chain(state:InputState):
    user_input=state["user_question"]
    input_ = {"user_input": user_input}
    ai_msg = llm_chain.invoke(input_)
    # print("ai_msg",ai_msg)
    tool_msgs = tool.batch(ai_msg.tool_calls)
    # print("tool_msgs",tool_msgs)
    z=llm_chain.invoke({**input_, "messages": [ai_msg, *tool_msgs]})
    # print("toolchain",z)
    return {"gen_context":z}




# %%


class EvaluationOutput(BaseModel):
    Accuracy: int
    Clarity_and_Grammar: int
    Total_Score: int
    feedback: str
    Relevance_to_the_Question:int
    percentage:float
    



# %%
def evaluate(state: InputState):
    """Evaluate the question and answer using the generated context."""
    prompt = f"""
You are an expert evaluator for a Q&A system. Your task is to evaluate the user's answer to a given question based on the following input details:

- **Reference Answer**: "{state['gen_context']}" (if available; otherwise, base your evaluation on the question and subject knowledge)
- **User's Answer**: "{state['user_answer']}"
- **Question**: "{state['user_question']}"
- **Question Type**: "{state['question_type']}"

### Evaluation Criteria:
1. **Relevance to the Question (10 points)**: How well the user's answer addresses the question.
2. **Accuracy (10 points)**: Whether the information provided is factually correct compared to the reference answer or subject knowledge.
3. **Clarity and Grammar (10 points)**: How clear and grammatically correct the answer is.

### Instructions:
- if the answer is not relevance directly give  0 marks as the answer is not relevance
- Provide a score for each criterion out of 10.
- Calculate the total score out of 30 by summing the individual scores.
- Calculate the percentage as `(Total Score / 30) * 100` and round it to two decimal places.
- Provide constructive feedback on how the user can improve their answer.

### Output Format:
Provide your evaluation strictly in the following JSON format:
```json
{{
    "Relevance_to_the_Question": (integer between 0-10),
    "Accuracy": (integer between 0-10),
    "Clarity_and_Grammar": (integer between 0-10),
    "Total_Score": (integer between 0-30),
    "percentage": (float rounded to two decimal places),
    "feedback": "Constructive feedback about the answer."
}}



    Relevance_to_the_Question
    # prompt = f"""
    # Here is a reference answer: "{state["gen_context"]}"
    # Here is the user’s answer: "{state["user_answer"]}"
    # Here is the question: "{state["user_question"]}"
    # You are a highly skilled evaluator and check if the user answer is correct as per the question give full marks otherwise. Evaluate the user's answer using the following criteria:
    # - Factual Accuracy: Rate from 1-10.
    # - Clarity_and_Grammar: Rate from 1-10.
    # Provide the following structured output:
    # {{
    #     "factual_accuracy": (integer between 1-10),
    #     "Clarity_and_Grammar": (integer between 1-10),
    #     "total_score": (integer between 0-30),
    #     "justification": (string)
    # }}
    # Ensure your response strictly adheres to this JSON format.
    # """
    response = llm.invoke(prompt)
    try:
        parsed_response = EvaluationOutput.parse_raw(response.content)
        return {
            "Total_Score": parsed_response.Total_Score,
            "feedback": parsed_response.feedback,
            "Accuracy":parsed_response.Accuracy,
            "Clarity_and_Grammar":parsed_response.Clarity_and_Grammar,
            "Relevance_to_the_Question":parsed_response.Relevance_to_the_Question,
            "percentage":parsed_response.percentage

        }
    except Exception as e:
        print("Error parsing response:", e)
        return {
            "Total_Score": 0,
            "feedback": "Unable to parse response. Please ensure the format is correct.",
        }


# %%
builder=StateGraph(OverallState, input=InputState, output=OutputState)
builder.add_node("tailvy_tool",tool_chain)
builder.add_node("evalulate",evaluate)
builder.add_edge(START,"tailvy_tool")
builder.add_edge("tailvy_tool","evalulate")
builder.add_edge("evalulate",END)
graph=builder.compile()


# # %%
# from IPython.display import Image, display

# try:
#     display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
# except Exception:
#     # This requires some extra dependencies and is optional
#     pass

# %%
# data=graph.invoke({"user_question":"who is the founder of apple company?","user_answer":"steve jobs ","gen_context:str":"None","question_type":"Short"})

# %%
# data

# %%
import nest_asyncio
import uvicorn
import subprocess
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException, status


# %%

app = FastAPI()
class Item(BaseModel):
    user_question: str
    user_answer: str
    question_type: str =None

@app.post("/evaluate")
def evaluate(item: Item):
    try:
        # Simulating the invocation of your external graph or model.
        data = graph.invoke({"user_question": item.user_question, "user_answer": item.user_answer, "gen_context": "None", "question_type": item.question_type})

        # Return the response with the obtained data
        return JSONResponse(
            content={"message": "Item evaluated successfully", "data": data},
            status_code=status.HTTP_200_OK  # HTTP 201 Created
        )
    except Exception as e:
        # Catch any exception and return an error response with a 400 or 500 status code
        return JSONResponse(
            content={"message": "An error occurred during evaluation", "error": str(e)},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR  # HTTP 500 Internal Server Error
        )

@app.get("/hello")
def create_item():
    # Example of using the status code for 'Created' (201)
    
    return JSONResponse(
        content={"message": "Hello World"},
        status_code=status.HTTP_201_CREATED  # HTTP 201 Created
    )




