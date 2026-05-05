from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime

from weakagent.tools.base import BaseTool, ToolExecutionResult


class ListFilesTool(BaseTool):
    """Tool to list files and directories, similar to the ls command"""

    name: str = "list_files"
    description: str = "Use when you need to list files and directories in a directory"
    parameters: dict = {
                "type": "object",
                "properties": {
                    "directory_path": {
                        "type": "string",
                        "description": "The path to the directory to list, default is the current working directory (data directory: F:/_Work/ai-dashboard/data)"
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Whether to list subdirectories recursively, default false",
                        "default": False
                    },
                    "show_details": {
                        "type": "boolean",
                        "description": "Whether to show detailed information (file size, modification time), default true",
                        "default": True
                    },
                    "file_extensions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Only list files with specified extensions (e.g. ['.py', '.txt', '.xlsx']), optional"
                    },
                    "include_directories": {
                        "type": "boolean",
                        "description": "Whether to include directories in the results, default true",
                        "default": True
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of return results, default 500",
                        "default": 500
                    }
                },
        "required": [],
    }

    async def execute(self, directory_path: str,
                            recursive: bool = False,
                            show_details: bool = True,
                            file_extensions: Optional[List[str]] = None,
                            include_directories: bool = True,
                            max_results: int = 500) -> ToolExecutionResult:
        if not directory_path:
            return self.fail_response("Directory path is required")
        
        try:
            # 转换为 Path 对象
            path = Path(directory_path)
            
            # 检查路径是否存在
            if not path.exists():
                return f"路径不存在: {directory_path}"
            
            # 如果不是目录，返回文件信息
            if path.is_file():
                return self._format_file_info(path, show_details)
            
            if not path.is_dir():
                return f"路径不是目录: {directory_path}"
            
            # 获取文件列表
            items = []
            if recursive:
                items = self._get_items_recursive(path, file_extensions, include_directories, max_results)
            else:
                items = self._get_items_in_directory(path, file_extensions, include_directories, max_results)
            
            if not items:
                return f"目录 '{directory_path}' 为空或没有匹配的文件"
            
            # 格式化输出
            output_lines = [f"目录: {path.absolute()}\n"]
            output_lines.append(f"找到 {len(items)} 个项目:\n")
            
            # 先列出目录，再列出文件
            dirs = [item for item in items if item["is_directory"]]
            files = [item for item in items if not item["is_directory"]]
            
            if dirs:
                output_lines.append("\n[目录]")
                for item in dirs:
                    output_lines.append(self._format_item(item, show_details))
            
            if files:
                output_lines.append("\n[文件]")
                for item in files:
                    output_lines.append(self._format_item(item, show_details))
            
            if len(items) >= max_results:
                output_lines.append(f"\n(仅显示前 {max_results} 个项目)")
            
            return "\n".join(output_lines)
            
        except Exception as e:
            return f"列出文件时出错: {str(e)}"
    
    def _format_file_info(self, file_path: Path, show_details: bool) -> str:
        """格式化单个文件信息"""
        try:
            stat = file_path.stat()
            size = self._format_size(stat.st_size)
            mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            
            if show_details:
                return f"{file_path.name}\n  路径: {file_path.absolute()}\n  大小: {size}\n  修改时间: {mtime}"
            else:
                return str(file_path.name)
        except Exception:
            return str(file_path.name)
    
    def _format_item(self, item: Dict[str, Any], show_details: bool) -> str:
        """Format single item information"""
        name = item["name"]
        if show_details:
            if item["is_directory"]:
                return f"  📁 {name}/"
            else:
                size = item.get("size", "?")
                mtime = item.get("mtime", "?")
                return f"  📄 {name} ({size}, {mtime})"
        else:
            if item["is_directory"]:
                return f"  {name}/"
            else:
                return f"  {name}"
    
    def _get_items_in_directory(
        self, 
        directory: Path, 
        file_extensions: Optional[List[str]], 
        include_directories: bool,
        max_results: int
    ) -> List[Dict[str, Any]]:
        """Get items in a directory"""
        items = []
        try:
            for item in directory.iterdir():
                if len(items) >= max_results:
                    break
                
                if item.is_dir():
                    if include_directories:
                        items.append({
                            "name": item.name,
                            "path": str(item),
                            "is_directory": True
                        })
                elif item.is_file():
                    if file_extensions is None or item.suffix in file_extensions:
                        stat = item.stat()
                        items.append({
                            "name": item.name,
                            "path": str(item),
                            "is_directory": False,
                            "size": self._format_size(stat.st_size),
                            "mtime": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                        })
        except PermissionError:
            pass
        return items
    
    def _get_items_recursive(
        self, 
        directory: Path, 
        file_extensions: Optional[List[str]], 
        include_directories: bool,
        max_results: int
    ) -> List[Dict[str, Any]]:
        """Get all items in a directory recursively"""
        items = []
        try:
            for item in directory.rglob('*'):
                if len(items) >= max_results:
                    break
                
                if item.is_dir():
                    if include_directories:
                        items.append({
                            "name": str(item.relative_to(directory)),
                            "path": str(item),
                            "is_directory": True
                        })
                elif item.is_file():
                    if file_extensions is None or item.suffix in file_extensions:
                        stat = item.stat()
                        items.append({
                            "name": str(item.relative_to(directory)),
                            "path": str(item),
                            "is_directory": False,
                            "size": self._format_size(stat.st_size),
                            "mtime": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                        })
        except PermissionError:
            pass
        return items
    
    def _format_size(self, size_bytes: int) -> str:
        """Format file size"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"
