"""The docs server with one extra line: every start appends "<pid> <time>" to counting_server.starts.

session_check.py launches this file as an MCPAdapter Path target (stdio subprocess) and counts the lines, i.e. how
many server processes a pattern started. The server itself is unchanged (docs_mcp.server).
"""

import os
import time
from pathlib import Path

from docs_mcp.server import mcp

STARTS = Path(__file__).with_suffix(".starts")

if __name__ == "__main__":
    with STARTS.open("a") as f:
        f.write(f"{os.getpid()} {time.time():.3f}\n")
    mcp.run(show_banner=False)
