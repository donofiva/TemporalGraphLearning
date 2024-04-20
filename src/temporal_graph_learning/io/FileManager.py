import os
from typing import List


class FileManager:

    @staticmethod
    def directory_create(path: str):
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def directory_resolve(directories: List[str]):
        return '/'.join(directories)

    @staticmethod
    def file_resolve_file_path(directory: str, file: str):
        return '/'.join([directory, file])