import math
import random
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

    def get_metrics(self, result: np.ndarray):
        e = self.__device_count * (self.__a + 1 / self.__k)
        x = np.mean(result)
        d = self.__device_count / (self.__k ** 2)
        s = np.std(result) ** 2
        me = np.median(result)
        r = result[-1] - result[0]
        return np.asarray([e, x, abs(e-x), d, s, abs(d-s), me, r], dtype=float)

    def get_graphics(self, results: np.ndarray):
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

    def get_qj(self, borders: np.ndarray):
        n = borders.shape[0]
        qj = np.zeros(shape=(n+1), dtype=float)
        qj[0] = self.__F(borders[0]) - self.__F(-math.inf)
        for i in range(1, n):
            qj[i] = self.__F(borders[i]) - self.__F(borders[i-1])
        qj[n] = self.__F(math.inf) - self.__F(borders[n-1])
        return qj

    def get_r0(self, results: np.ndarray, borders: np.ndarray, qj: np.ndarray):
        k = borders.shape[0]
        n = results.shape[0]
        n_array = np.zeros(shape=(k+1), dtype=int)
        for i in range(1, k-1):
            for result in results:
                if result < borders[0]:
                    n_array[0] += 1
                elif borders[i] <= result < borders[i + 1]:
                    n_array[i] += 1
                elif borders[k-1] <= result:
                    n_array[k] += 1
        print(n_array)
        print(qj)
        r0 = 0.0
        for i in range(k + 1):
            r0 += ((n_array[i] - n * qj[i]) ** 2) / (n * qj[i])
            print(r0)
        return r0

    def get_fr0(self, r0: float, k: int):
        f = lambda x: (2**(-k/2))*(x**(k/2-1))*math.exp(-x/2)/math.gamma(k/2) if x > 0 else 0
        return 1 - scipy.integrate.quad(f, 0, r0)[0]

    def get_histogram(self, results: np.ndarray, borders: np.ndarray):
        n = results.shape[0]
        z_array = np.zeros(shape=(borders.shape[0]-1), dtype=float)
        f_array = np.zeros(shape=(borders.shape[0]-1), dtype=float)
        n_array = np.zeros(shape=(borders.shape[0]-1), dtype=float)
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
