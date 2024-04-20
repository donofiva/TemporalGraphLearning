import os


class FileManager:

    @staticmethod
    def directory_create(path: str):
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def file_resolve_file_path(directory: str, file: str):
        return '/'.join([directory, file])