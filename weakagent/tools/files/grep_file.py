import re

from typing import Optional, List, Dict, Any
from pathlib import Path

from weakagent.tools.base import BaseTool, ToolExecutionResult


class GrepTool(BaseTool):
    name: str = "grep"
    description: str = "Use when you need to search for a pattern in a file or directory"
    parameters: dict = {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "The pattern to search for (supports regular expressions)"
                    },
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file or directory to search (data directory: F:/_Work/ai-dashboard/data)"
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "If file_path is a directory, whether to recursively search subdirectories, default false",
                        "default": False
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "Whether to distinguish between uppercase and lowercase, default true",
                        "default": True
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of return results, default 100",
                        "default": 100
                    },
                    "file_extensions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "If searching a directory, only search files with specified extensions (e.g. ['.py', '.txt']), optional"
                    }
                },
        "required": ["pattern", "file_path"],
    }
    

    async def execute(self, pattern: str,
                            file_path: str,
                            recursive: bool = False,
                            case_sensitive: bool = True,
                            max_results: int = 100,
                            file_extensions: Optional[List[str]] = None) -> ToolExecutionResult:
        try:
            path = Path(file_path)
            if not path.exists():
                return self.fail_response(f"File or directory not found: {file_path}")
            
            # 编译正则表达式
            flags = 0 if case_sensitive else re.IGNORECASE
            try:
                regex = re.compile(pattern, flags)
            except re.error as e:
                return self.fail_response(f"Regular expression error: {str(e)}")
            
            results: List[Dict[str, Any]] = []
            
            # 如果是文件
            if path.is_file():
                results.extend(self._search_in_file(path, regex, max_results - len(results)))
            # 如果是目录
            elif path.is_dir():
                if recursive:
                    files = self._get_files_recursive(path, file_extensions)
                else:
                    files = [f for f in path.iterdir() if f.is_file()]
                    if file_extensions:
                        files = [f for f in files if f.suffix in file_extensions]
                
                for file in files:
                    if len(results) >= max_results:
                        break
                    results.extend(self._search_in_file(file, regex, max_results - len(results)))
            else:
                return self.fail_response(f"The path is neither a file nor a directory: {file_path}")
            
            # 格式化结果
            if not results:
                return self.fail_response(f"No matching content found for '{pattern}'")
            
            # 构建返回字符串
            output_lines = [f"Found {len(results)} matching results:\n"]
            
            for result in results[:max_results]:
                file_rel_path = result["file"]
                line_num = result["line_number"]
                line_content = result["line_content"].rstrip()
                output_lines.append(f"{file_rel_path}:{line_num}: {line_content}")
            
            if len(results) > max_results:
                output_lines.append(f"\n(Only displaying the first {max_results} results, found {len(results)} in total)")
            
            return self.success_response("\n".join(output_lines))
            
        except Exception as e:
            return self.fail_response(f"Error searching: {str(e)}")
    
    def _search_in_file(self, file_path: Path, regex: re.Pattern, max_results: int) -> List[Dict[str, Any]]:
        """Search in a single file"""
        results = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, start=1):
                    if len(results) >= max_results:
                        break
                    if regex.search(line):
                        results.append({
                            "file": str(file_path),
                            "line_number": line_num,
                            "line_content": line
                        })
        except Exception as e:
            # Ignore unreadable files (e.g. binary files, permission issues, etc.)
            pass
        return results
    
    def _get_files_recursive(self, directory: Path, file_extensions: Optional[List[str]] = None) -> List[Path]:
        """Get all files in a directory recursively"""
        files = []
        try:
            for item in directory.rglob('*'):
                if item.is_file():
                    if file_extensions is None or item.suffix in file_extensions:
                        files.append(item)
        except PermissionError:
            # Ignore permission errors
            pass
        return files
