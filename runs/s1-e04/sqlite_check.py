"""SqliteSaver checks with a scripted model (no model server): threads, processes and check_same_thread.

Run from s1-langgraph/: uv run python ../runs/s1-e04/sqlite_check.py   (output sqlite-check.txt)
"""
import inspect
import os
import sqlite3
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from research_agent.agent import build_graph


class Scripted(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def model(*texts):
    return Scripted(messages=iter([AIMessage(content=t) for t in texts]))


def ask(graph, thread: str, text: str):
    return graph.invoke({"messages": [{"role": "user", "content": text}]}, {"configurable": {"thread_id": thread}})


def case(name: str, fn) -> None:
    try:
        print(f"{name}: {fn()}")
    except Exception as e:
        print(f"{name}: {type(e).__name__}: {e}")


tmp = tempfile.mkdtemp()

# 1. The connection as the LangGraph docs write it: sqlite3.connect(path), check_same_thread left at True.
db1 = os.path.join(tmp, "one.db")
g1 = build_graph(model("a1", "a2"), checkpointer=SqliteSaver(sqlite3.connect(db1)))
case("1a connect(path), invoke in the creating thread", lambda: len(ask(g1, "t", "q1")["messages"]))
with ThreadPoolExecutor(1) as pool:
    case("1b same saver, invoke from another thread", lambda: len(pool.submit(ask, g1, "t", "q2").result()["messages"]))

# 2. check_same_thread=False
db2 = os.path.join(tmp, "two.db")
g2 = build_graph(model("b1", "b2"), checkpointer=SqliteSaver(sqlite3.connect(db2, check_same_thread=False)))
case("2a check_same_thread=False, creating thread", lambda: len(ask(g2, "t", "q1")["messages"]))
with ThreadPoolExecutor(1) as pool:
    case("2b check_same_thread=False, another thread", lambda: len(pool.submit(ask, g2, "t", "q2").result()["messages"]))

# 3. SqliteSaver.from_conn_string: how it opens the connection, and where the failing put runs
src = inspect.getsource(SqliteSaver.from_conn_string)
i = src.index("sqlite3.connect(")
print("3  from_conn_string opens:", " ".join(src[i:i + 200].split()).split(") as conn")[0] + ")")
import traceback
g3 = build_graph(model("c1"), checkpointer=SqliteSaver(sqlite3.connect(os.path.join(tmp, "three.db"))))
try:
    ask(g3, "t", "q")
except Exception as e:
    frames = [f"{fr.filename.split('site-packages/')[-1]}:{fr.name}" for fr in traceback.extract_tb(e.__traceback__)[-5:]]
    print("3  where 1a fails:", " -> ".join(frames))

# 4. A second process continues the thread from the same file (from_conn_string)
db4 = os.path.join(tmp, "four.db")
child = """
import sqlite3, sys
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from research_agent.agent import build_graph
class Scripted(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self
with SqliteSaver.from_conn_string(sys.argv[1]) as saver:
    g = build_graph(Scripted(messages=iter([AIMessage(content=sys.argv[3])])), checkpointer=saver)
    out = g.invoke({"messages": [{"role": "user", "content": sys.argv[2]}]}, {"configurable": {"thread_id": "p"}})
print([m.content for m in out["messages"]])
"""
for q, a in (("first process question", "first answer"), ("second process question", "second answer")):
    r = subprocess.run([sys.executable, "-c", child, db4, q, a], capture_output=True, text=True)
    print(f"4  process {q.split()[0]}: {r.stdout.strip() or r.stderr.strip()[-200:]}")
with sqlite3.connect(db4) as c:
    tables = [t for (t,) in c.execute("select name from sqlite_master where type='table'")]
    counts = {t: c.execute(f"select count(*) from {t}").fetchone()[0] for t in tables}
print(f"4  tables {counts}, file {os.path.getsize(db4)} bytes")
