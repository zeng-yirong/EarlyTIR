import json
import re
import subprocess
import sys
import os
import argparse
import tempfile


def extract_python_code(text: str) -> str:
    match = re.search(r'```python\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        return match.group(1)
    return text


def clean_code(code: str) -> str:
    code = extract_python_code(code)
    code = '\n'.join(line.rstrip() for line in code.split('\n'))
    return code.strip()


def fix_test_code_indent(test_code: str) -> str:
    lines = test_code.split('\n')
    in_check = False
    fixed = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('def check('):
            in_check = True
            fixed.append(line)
            continue
        if in_check:
            # 8 spaces before assert -> 4 spaces
            if line.startswith('        assert'):
                fixed.append('    ' + stripped)
                continue
            # Non-indented top-level line ends check() scope
            if line and not line[0].isspace():
                in_check = False
        fixed.append(line)
    return '\n'.join(fixed)


def fix_test_code_quotes(test_code: str) -> str:
    result = test_code
    # """"text"""" -> """text"""
    result = re.sub(r'""""(.*?)""""', r'"""\1"""', result, flags=re.DOTALL)
    # """"text""" -> """text"""
    result = re.sub(r'""""(.*?)"""', r'"""\1"""', result, flags=re.DOTALL)
    return result


def fix_test_code(test_code: str) -> str:
    test_code = fix_test_code_quotes(test_code)
    test_code = fix_test_code_indent(test_code)
    return test_code


def detect_test_type(test_code: str) -> str:
    if 'subprocess' in test_code:
        return 'stdin'
    return 'functional'


def _run_in_subprocess(full_code: str, timeout: float) -> str:
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(full_code)
            tmp_path = f.name

        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=timeout
        )

        if result.returncode == 0:
            return "passed"
        elif result.returncode == 1:
            stderr = result.stderr.strip()
            if 'AssertionError' in stderr:
                return "failed: AssertionError"
            for line in stderr.split('\n'):
                line = line.strip()
                for err in ('SyntaxError', 'NameError', 'TypeError', 'ValueError',
                            'KeyError', 'IndexError', 'AttributeError', 'RuntimeError',
                            'ZeroDivisionError', 'ImportError', 'IndentationError',
                            'TimeoutExpired'):
                    if line.startswith(err) or line.endswith(err):
                        return "failed: " + line.split(':')[0]
            return "failed: RuntimeError"
        else:
            return "failed: exit_code_" + str(result.returncode)
    except subprocess.TimeoutExpired:
        return "timed out"
    except Exception as e:
        return "failed: " + type(e).__name__
    finally:
        try:
            os.unlink(tmp_path)
        except (NameError, OSError):
            pass


def evaluate_item(item: dict, timeout: float) -> str:
    code = item.get("final_code", "")
    if not code or not code.strip():
        return "failed: final_code is empty"

    code = clean_code(code)
    if not code:
        return "failed: no valid code found"

    test_code = fix_test_code(item.get("test", ""))
    entry_point = item.get("entry_point", "")
    test_type = detect_test_type(test_code)

    if test_type == 'functional':
        full_code = code + "\n\n" + test_code + "\n\ncheck(" + entry_point + ")\n"
    else:
        full_code = "candidate = " + repr(code) + "\n\n" + test_code + "\n\ncheck(candidate)\n"

    return _run_in_subprocess(full_code, timeout)


def evaluate_file(file_path: str, timeout: float) -> dict:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    details = []
    passed = failed = skipped = timed_out = 0
    total = len(data)
    stats = {"functional": {"total": 0, "passed": 0}, "stdin": {"total": 0, "passed": 0}}

    for idx, item in enumerate(data):
        task_id = item.get("id", "unknown")
        used_turns = item.get("used_turns", "?")
        test_code = item.get("test", "")
        test_type = detect_test_type(test_code)
        stats[test_type]["total"] += 1

        res = evaluate_item(item, timeout=timeout)

        if res == "passed":
            passed += 1
            stats[test_type]["passed"] += 1
            status = "PASS"
        elif res == "timed out":
            timed_out += 1
            status = "TIMEOUT"
        elif "empty" in res or "no valid" in res:
            skipped += 1
            status = "SKIP"
        else:
            failed += 1
            err = res.replace("failed: ", "")
            status = "FAIL (" + err + ")"

        details.append({
            "task_id": task_id,
            "used_turns": used_turns,
            "test_type": test_type,
            "status": status,
            "detail": res,
        })

        if (idx + 1) % 50 == 0 or idx == total - 1:
            print("  [" + str(idx + 1) + "/" + str(total) + "] " +
                  task_id.ljust(20) + " [" + test_type.ljust(10) + "] " + status)

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
        "by_type": {
            "functional": {
                "total": stats["functional"]["total"],
                "passed": stats["functional"]["passed"],
                "rate": round(stats["functional"]["passed"] / stats["functional"]["total"] * 100, 2)
                if stats["functional"]["total"] > 0 else 0,
            },
            "stdin": {
                "total": stats["stdin"]["total"],
                "passed": stats["stdin"]["passed"],
                "rate": round(stats["stdin"]["passed"] / stats["stdin"]["total"] * 100, 2)
                if stats["stdin"]["total"] > 0 else 0,
            },
        },
        "details": details,
    }


def main():
    parser = argparse.ArgumentParser(description="LiveCodeBench tool")
    parser.add_argument("input_file")
    parser.add_argument("--run-range", type=int, nargs=2, metavar=("START", "END"))
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--output", "-o", type=str)
    args = parser.parse_args()

    if args.run_range:
        start, end = args.run_range
        files = [args.input_file.format(run_id=i) for i in range(start, end + 1)]
    else:
        files = [args.input_file]

    missing = [f for f in files if not os.path.exists(f)]
    for f in missing:
        print("Not found: " + f)
    files = [f for f in files if os.path.exists(f)]

    if not files:
        print("No files")
        return

    all_results = []
    for file_path in files:

        result = evaluate_file(file_path, timeout=args.timeout)
        all_results.append(result)

        r = result
        print("-" * 75)
        print("  Total Tasks    : " + str(r['total']))
        print("  Skipped (Empty): " + str(r['skipped']))
        print("  Valid Evals    : " + str(r['valid']))
        print("  Passed         : " + str(r['passed']))
        print("  Failed         : " + str(r['failed']))
        print("  Timed Out      : " + str(r['timed_out']))
        print("  Pass@1 (Valid) : " + "{:.2f}".format(r['pass_rate_valid']) + "% (" + str(r['passed']) + "/" + str(
            r['valid']) + ")")
        print("  Pass@1 (Total) : " + "{:.2f}".format(r['pass_rate_total']) + "% (" + str(r['passed']) + "/" + str(
            r['total']) + ")")
        print("  Statistics by Category:")
        ft = r['by_type']['functional']
        st = r['by_type']['stdin']
        print("    Functional   : " + str(ft['passed']) + "/" + str(ft['total']) + " (" + "{:.2f}".format(
            ft['rate']) + "%)")
        print("    Stdin        : " + str(st['passed']) + "/" + str(st['total']) + " (" + "{:.2f}".format(
            st['rate']) + "%)")
        print()

    # Multi-round summary
    if len(all_results) > 1:
        print("=" * 75)
        print("Multi-round Summary")
        print("=" * 75)
        for r in all_results:
            print("  " + os.path.basename(r['file']) + ": Pass@1=" +
                  "{:.2f}".format(r['pass_rate_valid']) + "% (" +
                  str(r['passed']) + "/" + str(r['valid']) + ")")

        total_all = sum(r['total'] for r in all_results)
        passed_all = sum(r['passed'] for r in all_results)
        skipped_all = sum(r['skipped'] for r in all_results)
        valid_all = total_all - skipped_all
        avg_acc = passed_all / valid_all * 100 if valid_all > 0 else 0

        ft_total = sum(r['by_type']['functional']['total'] for r in all_results)
        ft_passed = sum(r['by_type']['functional']['passed'] for r in all_results)
        st_total = sum(r['by_type']['stdin']['total'] for r in all_results)
        st_passed = sum(r['by_type']['stdin']['passed'] for r in all_results)

        print("  Total: Pass@1=" + "{:.2f}".format(avg_acc) + "% (" +
              str(passed_all) + "/" + str(valid_all) + ")")
        if ft_total > 0:
            print("    Functional: " + str(ft_passed) + "/" + str(ft_total) +
                  " (" + "{:.2f}".format(ft_passed / ft_total * 100) + "%)")
        if st_total > 0:
            print("    Stdin: " + str(st_passed) + "/" + str(st_total) +
                  " (" + "{:.2f}".format(st_passed / st_total * 100) + "%)")
        print("=" * 75)

    # Save output
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
        print("\nResults saved to: " + args.output)


if __name__ == '__main__':
    main()