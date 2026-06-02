import os
import re
import fnmatch
import json
import shutil
import signal
import tempfile
import threading
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Callable, Optional


# ============================================================
# constants
# ============================================================

SERVER_TOOL_READ_FILE_MAX_SIZE = 16 * 1024
SERVER_TOOL_FILE_SEARCH_MAX_RESULTS = 100
SERVER_TOOL_GREP_SEARCH_MAX_RESULTS = 100

SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_OUTPUT_SIZE = 16 * 1024
SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_TIMEOUT = 60


# ============================================================
# utility helpers
# ============================================================

def string_format(fmt: str, *args) -> str:
    return fmt % args


def string_join(items: List[str], sep: str) -> str:
    return sep.join(items)


def json_value(obj: Dict[str, Any], key: str, default: Any):
    return obj.get(key, default)


def safe_json_to_str(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2)
    except Exception as e:
        return json.dumps({
            "error": f"json serialization failed: {str(e)}"
        })


def glob_match(pattern: str, path: str) -> bool:
    if pattern == "**":
        return True

    pattern = pattern.replace("\\", "/")
    path = path.replace("\\", "/")

    return fnmatch.fnmatch(path, pattern)


def _function_definition(name: str, description: str, parameters: Dict[str, Any]):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


    return {
        "error": {
            "message": message,
            "type": error_type
        }
    }


ERROR_TYPE_SERVER = "server_error"
ERROR_TYPE_INVALID_REQUEST = "invalid_request"


# ============================================================
# subprocess handling
# ============================================================

@dataclass
class RunProcResult:
    output: str = ""
    exit_code: int = -1
    timed_out: bool = False


def run_process(
    args: List[str],
    max_output: int,
    timeout_secs: int
) -> RunProcResult:

    result = RunProcResult()

    try:
        process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

    except Exception as e:
        result.output = f"failed to spawn process: {str(e)}"
        return result

    output_chunks = []
    output_size = 0
    truncated = False
    timed_out = False

    def timeout_killer():
        nonlocal timed_out

        try:
            process.wait(timeout=timeout_secs)
        except subprocess.TimeoutExpired:
            timed_out = True

            try:
                process.kill()
            except Exception:
                pass

    timeout_thread = threading.Thread(target=timeout_killer)
    timeout_thread.start()

    try:
        if process.stdout:
            for line in process.stdout:

                if not truncated:
                    encoded_len = len(line.encode())

                    if output_size + encoded_len <= max_output:
                        output_chunks.append(line)
                        output_size += encoded_len
                    else:
                        remaining = max_output - output_size

                        if remaining > 0:
                            partial = line.encode()[:remaining].decode(
                                errors="ignore"
                            )
                            output_chunks.append(partial)

                        truncated = True

    except Exception as e:
        output_chunks.append(f"\n[read error: {str(e)}]")

    timeout_thread.join()

    try:
        result.exit_code = process.wait()
    except Exception:
        pass

    result.output = "".join(output_chunks)

    if truncated:
        result.output += "\n[output truncated]"

    result.timed_out = timed_out

    return result


# ============================================================
# HTTP wrappers
# ============================================================

@dataclass
class ServerHttpReq:
    body: str = ""


@dataclass
class ServerHttpRes:
    status: int = 200
    data: str = ""


# ============================================================
# base tool
# ============================================================

class ServerTool:

    def __init__(self):
        self.name = ""
        self.display_name = ""
        self.permission_write = False

    def get_definition(self):
        raise NotImplementedError()

    def invoke(self, params: Dict[str, Any]):
        raise NotImplementedError()

    def to_json(self):
        return {
            "display_name": self.display_name,
            "tool": self.name,
            "type": "builtin",
            "permissions": {
                "write": self.permission_write
            },
            "definition": self.get_definition()
        }


# ============================================================
# read_file
# ============================================================

class ServerToolReadFile(ServerTool):

    def __init__(self):
        super().__init__()

        self.name = "read_file"
        self.display_name = "Read file"
        self.permission_write = False

    def get_definition(self):
        return _function_definition(
            self.name,
            "Read file contents from disk.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative file path"},
                    "start_line": {
                        "type": "integer",
                        "description": "First line to read (1-based)",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Last line to read (-1 for EOF)",
                    },
                    "append_loc": {
                        "type": "boolean",
                        "description": "Prefix each line with its line number",
                    },
                },
                "required": ["path"],
            },
        )

    def invoke(self, params):

        path = params["path"]

        start_line = json_value(params, "start_line", 1)
        end_line = json_value(params, "end_line", -1)
        append_loc = json_value(params, "append_loc", False)

        try:
            file_size = os.path.getsize(path)
        except Exception as e:
            return {
                "error": f"cannot stat file: {str(e)}"
            }

        if (
            file_size > SERVER_TOOL_READ_FILE_MAX_SIZE
            and end_line == -1
        ):
            return {
                "error": (
                    f"file too large ({file_size} bytes). "
                    f"use start_line/end_line"
                )
            }

        try:
            result = []
            lineno = 0
            current_size = 0

            with open(path, "r", encoding="utf-8") as f:

                for line in f:
                    lineno += 1

                    if lineno < start_line:
                        continue

                    if end_line != -1 and lineno > end_line:
                        break

                    line = line.rstrip("\n")

                    if append_loc:
                        out_line = f"{lineno}→ {line}\n"
                    else:
                        out_line = line + "\n"

                    size = len(out_line.encode())

                    if current_size + size > SERVER_TOOL_READ_FILE_MAX_SIZE:
                        result.append("[output truncated]")
                        break

                    result.append(out_line)
                    current_size += size

            return {
                "plain_text_response": "".join(result)
            }

        except Exception as e:
            return {
                "error": f"failed to open file: {str(e)}"
            }


# ============================================================
# file glob search
# ============================================================

class ServerToolFileGlobSearch(ServerTool):

    def __init__(self):
        super().__init__()

        self.name = "file_glob_search"
        self.display_name = "File search"

    def get_definition(self):
        return _function_definition(
            self.name,
            "Search for files under a directory using glob patterns.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory to search"},
                    "include": {
                        "type": "string",
                        "description": "Glob pattern for files to include",
                    },
                    "exclude": {
                        "type": "string",
                        "description": "Glob pattern for files to exclude",
                    },
                },
                "required": ["path"],
            },
        )

    def invoke(self, params):

        base = params["path"]
        include = json_value(params, "include", "**")
        exclude = json_value(params, "exclude", "")

        results = []
        count = 0

        try:
            for root, _, files in os.walk(base):

                for file in files:

                    full_path = os.path.join(root, file)

                    rel = os.path.relpath(full_path, base)
                    rel = rel.replace("\\", "/")

                    if not glob_match(include, rel):
                        continue

                    if exclude and glob_match(exclude, rel):
                        continue

                    results.append(full_path)

                    count += 1

                    if count >= SERVER_TOOL_FILE_SEARCH_MAX_RESULTS:
                        break

            text = "\n".join(results)
            text += f"\n---\nTotal matches: {count}\n"

            return {
                "plain_text_response": text
            }

        except Exception as e:
            return {
                "error": str(e)
            }


# ============================================================
# grep search
# ============================================================

class ServerToolGrepSearch(ServerTool):

    def __init__(self):
        super().__init__()

        self.name = "grep_search"
        self.display_name = "Grep search"

    def get_definition(self):
        return _function_definition(
            self.name,
            "Search file contents with a regular expression.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File or directory path"},
                    "pattern": {"type": "string", "description": "Regular expression"},
                    "include": {
                        "type": "string",
                        "description": "Glob pattern for files to include",
                    },
                    "exclude": {
                        "type": "string",
                        "description": "Glob pattern for files to exclude",
                    },
                    "return_line_numbers": {
                        "type": "boolean",
                        "description": "Include line numbers in matches",
                    },
                },
                "required": ["path", "pattern"],
            },
        )

    def invoke(self, params):

        path = params["path"]
        pattern = params["pattern"]

        include = json_value(params, "include", "**")
        exclude = json_value(params, "exclude", "")
        return_line_numbers = json_value(
            params,
            "return_line_numbers",
            False
        )

        try:
            regex = re.compile(pattern)

        except re.error as e:
            return {
                "error": f"invalid regex: {str(e)}"
            }

        results = []
        total = 0

        def search_file(fpath):

            nonlocal total

            try:
                with open(
                    fpath,
                    "r",
                    encoding="utf-8",
                    errors="ignore"
                ) as f:

                    for lineno, line in enumerate(f, start=1):

                        if total >= SERVER_TOOL_GREP_SEARCH_MAX_RESULTS:
                            break

                        if regex.search(line):

                            line = line.rstrip("\n")

                            if return_line_numbers:
                                results.append(
                                    f"{fpath}:{lineno}:{line}"
                                )
                            else:
                                results.append(
                                    f"{fpath}:{line}"
                                )

                            total += 1

            except Exception:
                pass

        if os.path.isfile(path):

            search_file(path)

        elif os.path.isdir(path):

            for root, _, files in os.walk(path):

                for file in files:

                    if total >= SERVER_TOOL_GREP_SEARCH_MAX_RESULTS:
                        break

                    full_path = os.path.join(root, file)

                    rel = os.path.relpath(full_path, path)
                    rel = rel.replace("\\", "/")

                    if not glob_match(include, rel):
                        continue

                    if exclude and glob_match(exclude, rel):
                        continue

                    search_file(full_path)

        else:
            return {
                "error": f"path does not exist: {path}"
            }

        output = "\n".join(results)
        output += f"\n\n---\nTotal matches: {total}\n"

        return {
            "plain_text_response": output
        }


# ============================================================
# exec shell command
# ============================================================

class ServerToolExecShellCommand(ServerTool):

    def __init__(self):
        super().__init__()

        self.name = "exec_shell_command"
        self.display_name = "Execute shell command"
        self.permission_write = True

    def get_definition(self):
        return _function_definition(
            self.name,
            "Execute a shell command and return stdout/stderr output.",
            {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run"},
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds",
                    },
                    "max_output_size": {
                        "type": "integer",
                        "description": "Maximum output bytes to capture",
                    },
                },
                "required": ["command"],
            },
        )

    def invoke(self, params):

        command = params["command"]

        timeout = min(
            json_value(params, "timeout", 10),
            SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_TIMEOUT
        )

        max_output_size = min(
            json_value(
                params,
                "max_output_size",
                SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_OUTPUT_SIZE
            ),
            SERVER_TOOL_EXEC_SHELL_COMMAND_MAX_OUTPUT_SIZE
        )

        if os.name == "nt":
            args = ["cmd", "/c", command]
        else:
            args = ["sh", "-c", command]

        res = run_process(
            args,
            max_output_size,
            timeout
        )

        text_output = res.output
        text_output += f"\n[exit code: {res.exit_code}]"

        if res.timed_out:
            text_output += " [exit due to timed out]"

        return {
            "plain_text_response": text_output
        }


# ============================================================
# write file
# ============================================================

class ServerToolWriteFile(ServerTool):

    def __init__(self):
        super().__init__()

        self.name = "write_file"
        self.display_name = "Write file"
        self.permission_write = True

    def get_definition(self):
        return _function_definition(
            self.name,
            "Write text content to a file on disk.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write"},
                    "content": {"type": "string", "description": "File contents"},
                },
                "required": ["path", "content"],
            },
        )

    def invoke(self, params):

        path = params["path"]
        content = params["content"]

        try:
            parent = os.path.dirname(path)

            if parent:
                os.makedirs(parent, exist_ok=True)

            with open(path, "wb") as f:
                f.write(content.encode())

            return {
                "result": "file written successfully",
                "path": path,
                "bytes": len(content.encode())
            }

        except Exception as e:
            return {
                "error": f"failed to write file: {str(e)}"
            }


# ============================================================
# get datetime
# ============================================================

class ServerToolGetDatetime(ServerTool):

    def __init__(self):
        super().__init__()

        self.name = "get_datetime"
        self.display_name = "Get Date & Time"

    def get_definition(self):
        return _function_definition(
            self.name,
            "Return the current local date and time.",
            {
                "type": "object",
                "properties": {},
            },
        )

    def invoke(self, params):
        return {
            "result": datetime.now().ctime()
        }


# ============================================================
# server tools manager
# ============================================================

class ServerTools:

    def __init__(self):

        self.tools: List[ServerTool] = []

        self.handle_get = None
        self.handle_post = None

    def build_tools(self):

        return [
            ServerToolReadFile(),
            ServerToolFileGlobSearch(),
            ServerToolGrepSearch(),
            ServerToolExecShellCommand(),
            ServerToolWriteFile(),
            ServerToolGetDatetime(),
        ]

    def setup(self, enabled_tools: List[str]):

        all_tools = self.build_tools()

        if enabled_tools:

            enabled_set = set(enabled_tools)

            known_names = [
                t.name for t in all_tools
            ]

            for name in enabled_tools:

                if name == "all":
                    continue

                if name not in known_names:
                    raise RuntimeError(
                        string_format(
                            'unknown tool "%s". available tools: %s',
                            name,
                            string_join(known_names, ", ")
                        )
                    )

            self.tools = []

            for tool in all_tools:

                if (
                    tool.name in enabled_set
                    or "all" in enabled_set
                ):
                    self.tools.append(tool)

        else:
            self.tools = all_tools

        def handle_get(req: ServerHttpReq):

            res = ServerHttpRes()

            try:
                result = []

                for tool in self.tools:
                    result.append(tool.to_json())

                res.data = safe_json_to_str(result)

            except Exception as e:

                res.status = 500

                res.data = safe_json_to_str(
                    format_error_response(
                        str(e),
                        ERROR_TYPE_SERVER
                    )
                )

            return res

        def handle_post(req: ServerHttpReq):

            res = ServerHttpRes()

            try:
                body = json.loads(req.body)

                tool_name = body["tool"]
                params = body.get("params", {})

                result = self.invoke(tool_name, params)

                res.data = safe_json_to_str(result)

            except json.JSONDecodeError as e:

                res.status = 400

                res.data = safe_json_to_str(
                    format_error_response(
                        str(e),
                        ERROR_TYPE_INVALID_REQUEST
                    )
                )

            except Exception as e:

                res.status = 500

                res.data = safe_json_to_str(
                    format_error_response(
                        str(e),
                        ERROR_TYPE_SERVER
                    )
                )

            return res

        self.handle_get = handle_get
        self.handle_post = handle_post

    def invoke(self, name: str, params: Dict[str, Any]):

        for tool in self.tools:

            if tool.name == name:
                return tool.invoke(params)

        return {
            "error": f"unknown tool: {name}"
        }


# ============================================================
# example usage
# ============================================================

if __name__ == "__main__":

    tools = ServerTools()

    tools.setup(["all"])

    req = ServerHttpReq(
        body=json.dumps({
            "tool": "get_datetime",
            "params": {}
        })
    )

    res = tools.handle_post(req)

    print(res.status)
    print(res.data)