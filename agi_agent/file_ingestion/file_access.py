import os
import re
import urllib.request
import urllib.error
from typing import List, Dict, Optional, Tuple
from datetime import datetime

SUPPORTED_PROTOCOLS = ['file://', 'http://', 'https://']


class FileAccessManager:
    def __init__(self, logger=None):
        self.logger = logger
        self.access_log = []
        self._validate_paths()

    def _validate_paths(self):
        self.base_dirs = {
            'local': os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')),
            'uploads': os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'webui', 'uploads'))
        }
        for dir_path in self.base_dirs.values():
            os.makedirs(dir_path, exist_ok=True)

    def read_local_file(self, file_path: str) -> Tuple[bool, str, Optional[bytes]]:
        try:
            abs_path = os.path.abspath(file_path)
            
            if not os.path.exists(abs_path):
                return False, f"File not found: {file_path}", None
            
            if not os.path.isfile(abs_path):
                return False, f"Not a file: {file_path}", None

            if not self._is_safe_path(abs_path):
                return False, f"Path traversal detected: {file_path}", None

            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext in ['.txt', '.md', '.json', '.csv']:
                with open(abs_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return True, content, None
            else:
                with open(abs_path, 'rb') as f:
                    content = f.read()
                return True, file_ext, content

        except UnicodeDecodeError:
            return False, f"Encoding error: {file_path}", None
        except PermissionError:
            return False, f"Permission denied: {file_path}", None
        except Exception as e:
            return False, f"Read error: {str(e)}", None

    def _is_safe_path(self, path: str) -> bool:
        normalized = os.path.normpath(path)
        
        if '..' in path.replace('\\', '/').split('/'):
            return False
        
        for base_dir in self.base_dirs.values():
            base_normalized = os.path.normpath(base_dir)
            if normalized == base_normalized:
                return True
            if normalized.startswith(base_normalized + os.sep):
                return True
        
        return False

    def download_from_url(self, url: str, save_dir: str = None) -> Tuple[bool, str, Optional[str]]:
        try:
            if not any(url.startswith(p) for p in SUPPORTED_PROTOCOLS):
                return False, f"Unsupported protocol: {url}", None

            if not save_dir:
                save_dir = self.base_dirs['uploads']
            os.makedirs(save_dir, exist_ok=True)

            filename = os.path.basename(url).split('?')[0]
            if not filename:
                filename = f"downloaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.dat"

            save_path = os.path.join(save_dir, filename)

            with urllib.request.urlopen(url, timeout=30) as response:
                with open(save_path, 'wb') as f:
                    f.write(response.read())

            self._log_access('download', url, save_path)
            return True, f"Downloaded to {save_path}", save_path

        except urllib.error.HTTPError as e:
            return False, f"HTTP error: {str(e)}", None
        except urllib.error.URLError as e:
            return False, f"URL error: {str(e)}", None
        except TimeoutError:
            return False, "Request timed out", None
        except Exception as e:
            return False, f"Download error: {str(e)}", None

    def list_directory(self, dir_path: str, extensions: List[str] = None) -> Tuple[bool, str, Optional[List[Dict]]]:
        try:
            abs_path = os.path.abspath(dir_path)
            
            if not os.path.exists(abs_path):
                return False, f"Directory not found: {dir_path}", None
            
            if not os.path.isdir(abs_path):
                return False, f"Not a directory: {dir_path}", None

            files = []
            for entry in os.listdir(abs_path):
                entry_path = os.path.join(abs_path, entry)
                if os.path.isfile(entry_path):
                    ext = os.path.splitext(entry)[1].lower()
                    if extensions and ext not in extensions:
                        continue
                    
                    stat = os.stat(entry_path)
                    files.append({
                        'name': entry,
                        'path': entry_path,
                        'extension': ext,
                        'size': stat.st_size,
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })

            files.sort(key=lambda x: x['modified'], reverse=True)
            return True, f"Found {len(files)} files", files

        except PermissionError:
            return False, f"Permission denied: {dir_path}", None
        except Exception as e:
            return False, f"List error: {str(e)}", None

    def save_file(self, content: bytes, filename: str, save_dir: str = None) -> Tuple[bool, str, Optional[str]]:
        try:
            if not save_dir:
                save_dir = self.base_dirs['uploads']
            os.makedirs(save_dir, exist_ok=True)

            sanitized_name = re.sub(r'[\\/:*?"<>|]', '_', filename)
            if not sanitized_name:
                sanitized_name = f"file_{datetime.now().strftime('%Y%m%d_%H%M%S')}.dat"

            save_path = os.path.join(save_dir, sanitized_name)
            
            with open(save_path, 'wb') as f:
                f.write(content)

            self._log_access('save', filename, save_path)
            return True, f"Saved to {save_path}", save_path

        except Exception as e:
            return False, f"Save error: {str(e)}", None

    def _log_access(self, action: str, source: str, destination: str):
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'source': source,
            'destination': destination
        }
        self.access_log.append(log_entry)
        if len(self.access_log) > 100:
            self.access_log = self.access_log[-100:]
        
        if self.logger:
            self.logger.info(f"File access: {action} from {source} to {destination}")

    def get_access_log(self, limit: int = 20) -> List[Dict]:
        return self.access_log[-limit:]

    def get_base_dirs(self) -> Dict[str, str]:
        return self.base_dirs