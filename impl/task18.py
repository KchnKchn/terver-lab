import math
import random
import argparse
import numpy as np
import scipy.integrate

class task18:

    __device_count = None
    __q = None
    __d = None
    __a = None
    __k = None

    def set_parameters(self, device_count: int, q: float, d: float):
        self.__device_count = device_count
        self.__q = q
        self.__d = d
        self.__a = self.__calculate_a()
        self.__k = self.__calculate_k()

    def set_seed(self, seed: int):
        random.seed(seed)

    def make_experemets(self, experiment_count: int):
        result = np.zeros(shape=(experiment_count), dtype=float)
        for i in range(experiment_count):
            job_time = self.__make_experiment()
            job_time = self.__truncate(job_time)
            result[i] = job_time
        result.sort()
        return result

    def get_metrics(self, result: np.array):
        e = self.__device_count * (self.__a + 1 / self.__k)
        x = np.mean(result)
        d = self.__device_count / (self.__k ** 2)
        s = np.std(result) ** 2
        me = np.median(result)
        r = result[-1] - result[0]
        return np.asarray([e, x, abs(e-x), d, s, abs(d-s), me, r], dtype=float)

    def get_graphics(self, results: np.array):
        n = results.shape[0]
        F = np.zeros(shape=(n), dtype=float)
        Fc = np.zeros(shape=(n), dtype=float)
        norm = 0
        for i in range(n):
            F_elem = self.__F(results[i])
            Fc_elem = i / n
            norm = max(norm, abs(F_elem - Fc_elem))
            F[i] = F_elem
            Fc[i] = Fc_elem
        return F, Fc, norm

    def get_histogram(self, results: np.array, borders: np.array):
        n = results.shape[0]
        z_array = np.zeros(shape=(len(borders)-1), dtype=float)
        f_array = np.zeros(shape=(len(borders)-1), dtype=float)
        n_array = np.zeros(shape=(len(borders)-1), dtype=float)
        norm = 0.0
        for i in range(borders.shape[0] - 1):
            z_array[i] = (borders[i] + borders[i + 1]) / 2
            f_array[i] = self.__f(z_array[i])
            for result in results:
                if borders[i] <= result < borders[i + 1]:
                    n_array[i] += 1
            length = borders[i + 1] - borders[i]
            n_array[i] = n_array[i] / (n * length)
            norm = max(abs(n_array[i] - f_array[i]), norm)
        return z_array, f_array, n_array, norm

    def __truncate(self, x: float):
        count = 4
        return int(x / 10 ** -count) * 10 ** -count

    def __f(self, y: float):
        return math.exp(-(y - self.__q * self.__device_count) ** 2 \
            / (2 * self.__d * self.__device_count)) \
            / math.sqrt(2 * math.pi * self.__d * self.__device_count)

    def __F(self, y: float):
        return 0.5 + 0.5 * math.erf((y - self.__q * self.__device_count) \
            / math.sqrt(2 * self.__d * self.__device_count))

    def __get_device_time(self):
        return self.__a - math.log(1 - random.random()) / self.__k

    def __calculate_a(self):
        return self.__q - math.sqrt(self.__d)

    def __calculate_k(self):
        return 1 / math.sqrt(self.__d)
    
    def __make_experiment(self):
        job_time = 0.0
        for i in range(self.__device_count):
            job_time += self.__get_device_time()
        return job_time
