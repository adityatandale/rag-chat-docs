"""UI smoke test (no network, no key): python tests/ui_smoke.py"""
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.pop("LLM_API_KEY", None)

from streamlit.testing.v1 import AppTest  # noqa: E402

import rag.embeddings as emb  # noqa: E402
import rag  # noqa: E402

# force the offline embedder so the test never downloads a model
rag.get_embedder = lambda prefer="auto": emb.HashingEmbedder()
import rag.llm  # noqa: E402,F401

at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
assert not at.exception, at.exception
assert at.chat_input[0].disabled, "chat should be disabled before indexing"

at.sidebar.checkbox[0].set_value(True)          # "Also load the sample documents"
idx = [b for b in at.sidebar.button if "Index documents" in b.label][0]
idx.click().run()
assert not at.exception, at.exception
print("index messages:", [s.value for s in at.sidebar.success])
assert len(at.sidebar.success) == 3

at.chat_input[0].set_value("How many days of paid annual leave do I get?").run()
assert not at.exception, at.exception
msgs = [(m.name, m.markdown[0].value[:60]) for m in at.chat_message]
print(msgs)
assert msgs[0][0] == "user" and "Retrieval-only" in msgs[1][1]
print("UI OK")
