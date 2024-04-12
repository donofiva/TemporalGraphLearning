import os

from pathlib import Path
from tgl.io.Directory import Directory
from tgl.io.File import File


class FileManager:

    @staticmethod
    def get_cwd() -> Path:
        return Path(os.getcwd()).resolve()

    @staticmethod
    def resolve_directory(*directories: Directory) -> Path:
        return FileManager.get_cwd().joinpath(*[directory.value for directory in directories])

    @staticmethod
    def resolve_file(directory: Path, file: File) -> Path:
        return directory.joinpath(file.value)
