"""Start an interactive local CLI session using an existing Qwen API key."""
import argparse
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = "832ca220f6955497f36abdbf"


def main():
    parser = argparse.ArgumentParser(description="Interactive Qwen RAG chat")
    parser.add_argument("--index", default=INDEX)
    parser.add_argument("--session", default=None, help="Reuse a previous session ID")
    parser.add_argument("--strategy", default="history", choices=("current", "history", "rewrite", "rewrite_checked"))
    parser.add_argument("--hybrid", action="store_true", help="Use ANN, keyword search and Qwen reranker")
    args = parser.parse_args()
    env = os.environ.copy()
    key = env.get("QWEN_API_KEY", "").strip()
    if not key and sys.platform == "win32":
        import winreg
        for hive, path in ((winreg.HKEY_CURRENT_USER, r"Environment"), (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment")):
            try:
                with winreg.OpenKey(hive, path) as reg:
                    key = str(winreg.QueryValueEx(reg, "QWEN_API_KEY")[0]).strip()
                if key:
                    break
            except OSError:
                pass
    if not key:
        print("未找到 QWEN_API_KEY，请先在系统或用户环境变量中配置。", file=sys.stderr)
        return 1
    base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    env.update({"PYTHONPATH": str(ROOT / "src"), "PYTHONUTF8": "1", "LLM_BASE_URL": base,
        "LLM_API_KEY": key, "LLM_MODEL": "qwen-plus", "EMBEDDING_BASE_URL": base,
        "EMBEDDING_API_KEY": key, "EMBEDDING_MODEL": "text-embedding-v4",
        "RERANK_BASE_URL": "https://dashscope.aliyuncs.com/api/v1", "RERANK_API_KEY": key,
        "RERANK_MODEL": "qwen3-rerank"})
    session = args.session or uuid.uuid4().hex
    print(f"会话 ID：{session}；输入 /quit 或 /exit 退出。")
    print("使用现有 QReCC 英文候选池，回答取决于知识库覆盖范围。")
    while True:
        try:
            question = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in ("/quit", "/exit"):
            break
        if not question:
            continue
        cmd = [sys.executable, "-m", "rag.cli", "ask", question, "--index", args.index,
               "--session", session, "--strategy", args.strategy]
        if args.hybrid:
            cmd.extend(["--backend", "ann", "--retrieval-mode", "hybrid", "--rerank", "--candidate-k", "10"])
        result = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode:
            print("运行失败：", result.stderr.strip() or "请检查模型服务配置。")
            continue
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            print("无法解析系统输出。")
            continue
        print("\n助手：", data.get("answer", ""))
        for evidence in data.get("evidence", []):
            print(f"  [{evidence['id']}] {evidence['source']}（字符 {evidence['start']}-{evidence['end']}）")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
