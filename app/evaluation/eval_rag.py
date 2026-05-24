"""
RAG 评测脚本 —— 使用自有 LLM 做评判（不依赖 RAGAS）
=====================================================
用法:
  python -m app.evaluation.eval_rag          # 跑评测
  python -m app.evaluation.eval_rag --baseline  # 保存基线
  python -m app.evaluation.eval_rag --compare    # 对比基线
"""

import json
import sys
from pathlib import Path
from app.core.llm_manager import get_llm
from app.core.config import settings
from langchain_core.messages import SystemMessage, HumanMessage

EVAL_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "eval" / "test_questions.json"
BASELINE_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "eval" / "baseline.json"

# 评判 LLM（temperature=0 保证一致性）
_judge = None

def _get_judge():
    global _judge
    if _judge is None:
        _judge = get_llm(provider="deepseek", temperature=0.0, streaming=False)
    return _judge


def score_faithfulness(question: str, answer: str, contexts: list[str]) -> float:
    """
    忠实度：回答是否有检索片段的支撑？
    返回 0~1，越高越好。
    """
    if not answer or not contexts:
        return 0.0

    context_text = "\n\n".join([f"[{i+1}] {c}" for i, c in enumerate(contexts)])
    prompt = f"""你是一位严格的 RAG 评测员。请判断以下"AI回答"有多大程度上来自于"参考片段"。

评分标准:
  1.0 — 回答完全基于参考片段，没有编造任何事实
  0.7 — 大部分基于片段，有少量合理推断（如举了教材外的通俗例子）
  0.4 — 有较多内容无法从片段中找到依据
  0.0 — 几乎全部是编造的

问题: {question}

参考片段:
{context_text}

AI回答:
{answer}

请只输出一个 0.0 到 1.0 之间的数字，如 0.85。不要输出任何其他文字。"""
    return _judge_score(prompt)


def score_context_precision(question: str, answer: str, contexts: list[str]) -> float:
    """
    上下文精确度：检索出的片段是否都对回答有用？
    返回 0~1，越高越好（说明噪音少）。
    """
    if not contexts:
        return 0.0

    # 考察前 2 个片段是否相关
    context_sample = "\n\n".join([f"[{i+1}] {c[:200]}" for i, c in enumerate(contexts[:2])])
    prompt = f"""你是一位 RAG 检索评测员。请判断以下"检索出的参考片段"是否与"问题"相关。

问题: {question}

检索到的参考片段:
{context_sample}

评分标准:
  1.0 — 片段非常相关，直接回答了问题
  0.5 — 部分相关，但不直接
  0.0 — 完全无关

请只输出一个 0.0 到 1.0 之间的数字。不要输出其他文字。"""
    return _judge_score(prompt)


def score_answer_quality(question: str, answer: str, ground_truth: str) -> float:
    """
    答案质量：回答与标准答案的一致性。
    返回 0~1，越高越好。
    """
    if not answer or not ground_truth:
        return 0.0

    prompt = f"""你是一位考研辅导老师。请判断"AI回答"与"标准答案"的知识一致性。

问题: {question}

标准答案:
{ground_truth}

AI回答:
{answer}

评分标准:
  1.0 — AI回答与标准答案完全一致，涵盖了所有关键知识点
  0.7 — 基本一致，遗漏了次要细节或表述略有不同
  0.4 — 只有部分一致，核心概念有偏差
  0.0 — 完全错误或不相关

请只输出一个 0.0 到 1.0 之间的数字。不要输出其他文字。"""
    return _judge_score(prompt)


def _judge_score(prompt: str) -> float:
    """调用评判 LLM 输出分数"""
    judge = _get_judge()
    messages = [
        SystemMessage(content="你是一个评测系统。只输出一个 0.0 到 1.0 之间的数字。"),
        HumanMessage(content=prompt),
    ]
    try:
        response = judge.invoke(messages)
        text = response.content.strip()
        # 提取数字
        import re
        match = re.search(r"(\d+\.?\d*)", text)
        if match:
            return float(match.group(1))
        return 0.5  # 解析失败，默认中等
    except Exception:
        return 0.5


def load_questions() -> list[dict]:
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def run_evaluation() -> dict:
    questions = load_questions()
    print(f"加载评测数据集: {EVAL_FILE}")
    print(f"共 {len(questions)} 道评测题目\n")

    f_scores, cp_scores, aq_scores = [], [], []

    for i, item in enumerate(questions):
        question = item["question"]
        ground_truth = item.get("ground_truth", "")
        print(f"[{i+1}/{len(questions)}] {question[:55]}...")

        # 检索上下文
        from app.services.vector_store import hybrid_search
        retrieved_docs = hybrid_search(query=question, collection_name="高数教材", k=4)
        contexts = [doc.page_content for doc in retrieved_docs]

        # 生成回答
        from app.agent.qa_chain import ask_question
        answer = ask_question(question=question, collection_name="高数教材")

        # 逐题打分
        f = score_faithfulness(question, answer, contexts)
        cp = score_context_precision(question, answer, contexts)
        aq = score_answer_quality(question, answer, ground_truth)

        f_scores.append(f)
        cp_scores.append(cp)
        aq_scores.append(aq)
        print(f"    忠实度={f:.2f}  检索精确={cp:.2f}  答案质量={aq:.2f}")

    avg_f = round(sum(f_scores) / len(f_scores), 4)
    avg_cp = round(sum(cp_scores) / len(cp_scores), 4)
    avg_aq = round(sum(aq_scores) / len(aq_scores), 4)

    return {
        "faithfulness": avg_f,
        "context_precision": avg_cp,
        "answer_quality": avg_aq,
    }


def print_results(scores: dict, label: str = "当前"):
    print(f"\n{'='*60}")
    print(f"  评测结果 —— {label}")
    print(f"{'='*60}")
    for name, val in scores.items():
        bar = "█" * int(val * 40) + "░" * (40 - int(val * 40))
        print(f"  {name:20s}: {val:.4f}  {bar}")

    f = scores.get("faithfulness", 0)
    cp = scores.get("context_precision", 0)
    aq = scores.get("answer_quality", 0)
    print(f"\n  诊断:")
    if f < 0.60: print(f"  ⚠️  忠实度 {f:.2%} — 幻觉严重，急需 Rerank + 阈值过滤")
    elif f < 0.80: print(f"  ⚡ 忠实度 {f:.2%} — 建议加强反幻觉 Prompt")
    else: print(f"  ✅ 忠实度 {f:.2%} — 良好")
    if cp < 0.50: print(f"  ⚠️  检索精确度 {cp:.2%} — 大量噪音，Rerank 急需")
    elif cp < 0.70: print(f"  ⚡ 检索精确度 {cp:.2%} — 需优化检索策略")
    else: print(f"  ✅ 检索精确度 {cp:.2%} — 良好")
    if aq < 0.60: print(f"  ⚠️  答案质量 {aq:.2%} — 知识遗漏严重")
    elif aq < 0.80: print(f"  ⚡ 答案质量 {aq:.2%} — 可能有细节遗漏")
    else: print(f"  ✅ 答案质量 {aq:.2%} — 良好")

    return scores


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true", help="保存当前结果为基线")
    parser.add_argument("--compare", action="store_true", help="与基线对比")
    args = parser.parse_args()

    result = run_evaluation()
    scores = print_results(result)

    if args.baseline:
        BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(BASELINE_FILE, "w", encoding="utf-8") as f:
            json.dump(scores, f, ensure_ascii=False, indent=2)
        print(f"\n基线已保存到 {BASELINE_FILE}")

    if args.compare:
        if not BASELINE_FILE.exists():
            print("\n未找到基线文件，请先运行 --baseline 建立基线")
            return
        with open(BASELINE_FILE, "r", encoding="utf-8") as f:
            baseline = json.load(f)
        print(f"\n{'='*60}")
        print(f"  与基线对比")
        print(f"{'='*60}")
        for metric, current_val in scores.items():
            base_val = baseline.get(metric, 0)
            diff = current_val - base_val
            arrow = "↑" if diff > 0 else "↓" if diff < 0 else "→"
            print(f"  {metric:20s}: {base_val:.4f} → {current_val:.4f}  {arrow} {diff:+.4f}")


if __name__ == "__main__":
    main()

