#!/usr/bin/env python3

import json
import re
import multiprocessing
import os
import argparse


def strip_thinking_tags(text: str) -> str:
    match = re.search(r'^(from|import|def|class)\b', text, re.MULTILINE)
    if match:
        text = text[match.start():]
    text = re.sub(r'\n?\s*</?assistant>', '', text)
    text = re.sub(r'\n?\s*</?(?:thinking|Thoughts?)>', '', text, flags=re.IGNORECASE)
    return text


def extract_code_block(raw: str) -> str:
    match = re.search(r'```python\s*(.*?)\s*```', raw, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    match = re.search(r'```\s*(.*?)\s*```', raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw.strip()


def has_function_def(text: str, entry_point: str) -> bool:
    return bool(re.search(r'\bdef\s+' + re.escape(entry_point) + r'\b', text))


def extract_from_tag(resp: str, tag: str) -> str:
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    last_start = resp.rfind(start_tag)
    if last_start == -1:
        return ""
    end_pos = resp.find(end_tag, last_start + len(start_tag))
    if end_pos == -1:
        return ""
    return resp[last_start + len(start_tag):end_pos].strip()


def _is_chinese_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    chinese = sum(1 for c in stripped if '一' <= c <= '鿿')
    ascii_ch = sum(1 for c in stripped if c.isascii() and not c.isspace())
    if chinese > 0 and chinese * 2 >= (ascii_ch + chinese):
        return True
    if stripped.startswith('#') and any('一' <= c <= '鿿' for c in stripped):
        return True
    return False


def _extract_python_blocks(text: str, entry_point: str = "") -> str:
    lines = text.split('\n')
    all_blocks = []
    current_block = []
    in_code = False

    def _save_block():
        nonlocal current_block
        # Remove trailing blank lines and incomplete assert
        while current_block:
            last = current_block[-1].strip()
            if last == 'assert' or last.endswith('assert ') or not last:
                current_block.pop()
            else:
                break
        if current_block:
            all_blocks.append('\n'.join(current_block).strip())
        current_block = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_code:
                current_block.append(line)
            continue

        is_code_start = re.match(r'^(from|import|def|class|@)\b', stripped)
        is_indented = line[0] in (' ', '\t') and not is_code_start
        is_closing = stripped in (')', ']', '}')
        is_test_line = stripped.startswith('assert ') or stripped.startswith('print(')

        if not in_code:
            if is_code_start or is_closing:
                in_code = True
                current_block = [line]
        else:
            if is_code_start or is_indented or is_closing:
                current_block.append(line)
            elif _is_chinese_line(line):
                in_code = False
                _save_block()
            elif is_test_line and not re.search(r'\bdef\s+', stripped):
                # assert/print in test code - might be testing a helper, keep it
                current_block.append(line)
            else:
                # Unknown non-indented line, end code block
                in_code = False
                _save_block()

    if in_code:
        _save_block()

    # Pick the best block: prefer ones that have the entry_point function
    # and have balanced triple quotes (complete docstrings).
    # Search from last to first, return first matching one.
    for block in reversed(all_blocks):
        has_def = has_function_def(block, entry_point)
        if not has_def:
            continue
        # Check balanced triple quotes
        td = block.count('"""')
        ts = block.count("'''")
        if td % 2 == 0 and ts % 2 == 0:
            return block

    # Fallback: last block regardless
    if all_blocks:
        return all_blocks[-1]
    return ""


def extract_code_from_responses(responses: list, entry_point: str, original_prompt: str) -> tuple:
    if not responses:
        return "", "empty_responses"

    for resp in reversed(responses):
        if not isinstance(resp, str) or not resp.strip():
            continue

        if '<submit>' in resp:
            raw = extract_from_tag(resp, 'submit')
            if raw:
                code = strip_thinking_tags(raw)
                code = extract_code_block(code)
                if code:
                    return code, "submit"

        if '<execute>' in resp:
            raw = extract_from_tag(resp, 'execute')
            if raw:
                code = strip_thinking_tags(raw)
                code = extract_code_block(code)
                if code and has_function_def(code, entry_point):
                    return code, "execute_with_def"

    for resp in reversed(responses):
        if not isinstance(resp, str) or not resp.strip():
            continue
        if '<execute>' in resp:
            raw = extract_from_tag(resp, 'execute')
            if raw:
                code = strip_thinking_tags(raw)
                code = extract_code_block(code)
                if code:
                    if original_prompt and not has_function_def(code, entry_point):
                        return original_prompt.strip() + "\n" + code, "execute+original_prompt"
                    return code, "execute_only"

    for resp in reversed(responses):
        if not isinstance(resp, str) or not resp.strip():
            continue
        code = _extract_python_blocks(resp, entry_point)
        if code and has_function_def(code, entry_point):
            return code, "raw_response"

    return "", "no_code_found"


def execute_test_case(full_code: str, result_queue):
    try:
        exec_globals = {}
        exec(full_code, exec_globals)
        result_queue.put("passed")
    except AssertionError:
        result_queue.put("failed: AssertionError")
    except Exception as e:
        result_queue.put(f"failed: {type(e).__name__}")


def evaluate_single_task(item: dict, timeout: float = 3.0) -> str:
    entry_point = item.get("entry_point", "")
    original_prompt = item.get("original_prompt", "")

    code, source = extract_code_from_responses(
        item.get("responses", []), entry_point, original_prompt
    )

    if not code and item.get("final_code"):
        code = item["final_code"].strip()
        source = "final_code"

    if not code:
        return f"failed: no valid code found (source: {source})"

    test_code = item.get("test", "")
    full_code = f"{code}\n\n{test_code}\n\ncheck({entry_point})\n"

    result_queue = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=execute_test_case,
        args=(full_code, result_queue)
    )
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        return "timed out"

    if not result_queue.empty():
        return result_queue.get()

    return "failed: unknown error"


def evaluate_file(file_path: str, timeout: float) -> dict:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    details = []
    passed = failed = skipped = timed_out = 0
    total = len(data)

    for idx, item in enumerate(data):
        task_id = item.get("id", item.get("task_id", "unknown"))
        used_turns = item.get("response_to_turn", item.get("used_turns", "?"))
        res = evaluate_single_task(item, timeout=timeout)

        if res == "passed":
            passed += 1
            status = "PASS"
        elif res == "timed out":
            timed_out += 1
            status = "TIMEOUT"
        elif "empty" in res or "no valid" in res:
            skipped += 1
            status = "SKIP"
        else:
            failed += 1
            status = f"FAIL ({res.replace('failed: ', '')})"

        details.append({
            "task_id": task_id,
            "used_turns": used_turns,
            "status": status,
            "detail": res,
        })

        if (idx + 1) % 10 == 0 or idx == total - 1:
            print(f"  [{idx + 1}/{total}] {task_id.ljust(20)} {status}")

    valid = total - skipped
    acc_valid = passed / valid * 100 if valid > 0 else 0
    acc_total = passed / total * 100 if total > 0 else 0

    return {
        "file": file_path,
        "total": total,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "timed_out": timed_out,
        "valid": valid,
        "pass_rate_valid": round(acc_valid, 2),
        "pass_rate_total": round(acc_total, 2),
        "details": details,
    }


def main():
    parser = argparse.ArgumentParser(description="HumanEval")
    parser.add_argument("input_file")
    parser.add_argument("--run-range", type=int, nargs=2, metavar=("START", "END"),)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--output", "-o", type=str)
    args = parser.parse_args()

    if args.run_range:
        start, end = args.run_range
        files = [args.input_file.format(run_id=i) for i in range(start, end + 1)]
    else:
        files = [args.input_file]

    for f in files:
        if not os.path.exists(f):
            print(f"Not find: {f}")
    files = [f for f in files if os.path.exists(f)]

    if not files:
        print("no files")
        return


    all_results = []
    for file_path in files:
        print("=" * 70)
        print(f"eval: {file_path}")
        print("=" * 70)

        result = evaluate_file(file_path, timeout=args.timeout)
        all_results.append(result)

        r = result
        print("-" * 70)
        print(f"  Total       : {r['total']}")
        print(f"  Skipped   : {r['skipped']}")
        print(f"  Valid     : {r['valid']}")
        print(f"  Pass         : {r['passed']}")
        print(f"  Fail         : {r['failed']}")
        print(f"  Timeout         : {r['timed_out']}")
        print(f"  Pass@1 (valid) : {r['pass_rate_valid']:.2f}% ({r['passed']}/{r['valid']})")
        print(f"  Pass@1 (total) : {r['pass_rate_total']:.2f}% ({r['passed']}/{r['total']})")

    if len(all_results) > 1:
        for r in all_results:
            print(f"  {r['file']}: Pass@1={r['pass_rate_valid']:.2f}% ({r['passed']}/{r['valid']})")

        total_all = sum(r['total'] for r in all_results)
        passed_all = sum(r['passed'] for r in all_results)
        skipped_all = sum(r['skipped'] for r in all_results)
        valid_all = total_all - skipped_all
        avg_acc = passed_all / valid_all * 100 if valid_all > 0 else 0
        print(f"  Total: Pass@1={avg_acc:.2f}% ({passed_all}/{valid_all})")
        print("=" * 70)

    if args.output:
        output_data = {
            "files_evaluated": len(all_results),
            "timeout_per_task": args.timeout,
            "per_file": all_results,
            "summary": {
                "total_files": len(all_results),
                "total_tasks": sum(r['total'] for r in all_results),
                "total_passed": sum(r['passed'] for r in all_results),
                "total_failed": sum(r['failed'] for r in all_results),
                "total_skipped": sum(r['skipped'] for r in all_results),
                "total_timed_out": sum(r['timed_out'] for r in all_results),
                "pass_rate_valid": round(
                    sum(r['passed'] for r in all_results) /
                    (sum(r['total'] for r in all_results) - sum(r['skipped'] for r in all_results)) * 100, 2
                ) if (sum(r['total'] for r in all_results) - sum(r['skipped'] for r in all_results)) > 0 else 0,
            }
        }
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
