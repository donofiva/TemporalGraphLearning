import os
from multiprocessing.dummy import Pool


class MultiThreadingManager:

    def __init__(self, number_of_workers: int = None):

        # Define number of workers
        self.number_of_workers = number_of_workers or os.cpu_count()

    def parallelize_pool(self, values, function):
        with Pool(self.number_of_workers) as pool:
            return pool.map(function, values)