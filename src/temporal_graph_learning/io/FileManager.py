import os


class FileManager:

    @staticmethod
    def directory_create(path: str):
        os.makedirs(path, exist_ok=True)
