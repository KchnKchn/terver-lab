import math
import random
import argparse
import numpy as np

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--experiment_count", 
        help="number of experiments for modeling",
        default=10, type=int, dest="experiment_count")
    parser.add_argument("--device_count",
        help="number of backup devices",
        default=1, type=int, dest="device_count")
    parser.add_argument("-q",
        help="expected value",
        default=1, type=float, dest="q")
    parser.add_argument("-d",
        help="dispersion",
        default=1, type=float, dest="d")
    parser.add_argument("-s", "--seed",
        help="seed for generate pseudo-random numbers",
        default=1, type=int, dest="seed")
    args = parser.parse_args()
    return args.experiment_count, args.device_count, args.q, args.d, args.seed

class task18:
    def __init__(self, device_count: int, q: float, d: float):
        self.__device_count = device_count
        self.__q = q
        self.__d = d
        self.__a = self.__calculate_a()
        self.__k = self.__calculate_k()
        self.__task_info = {
            "Количество приборов" : self.__device_count,
            "Среднее значение с.в" : self.__q,
            "Дисперсия с.в." : self.__d
        }
        self.__stats = None

    def set_parameters(self, device_count: int, q: float, d: float):
        self.__device_count = device_count
        self.__q = q
        self.__d = d
        self.__a = self.__calculate_a()
        self.__k = self.__calculate_k()
        self.__task_info = {
            "Количество приборов" : self.__device_count,
            "Среднее значение с.в" : self.__q,
            "Дисперсия с.в." : self.__d
        }
        self.__stats = None

    def set_seed(self, seed: int):
        random.seed(seed)

    def make_experemets(self, experiment_count: int):
        result = []
        for i in range(experiment_count):
            job_time = self.__make_experiment()
            job_time = self.__truncate(job_time)
            result.append(job_time)
        result.sort()
        self.__create_statistic(experiment_count, result)

    def print_info(self):
        print("Информация по задаче")
        for key, value in self.__task_info.items():
            print("{0}: {1}".format(key, value))
    
    def print_statistic(self):
        print("Результаты экспериментов")
        for key, value in self.__stats.items():
            print("{0}: {1}".format(key, value))

    def __create_statistic(self, experiment_count: int, result: list):
        self.__stats = {
            "Количество проведенных экспериментов" : experiment_count,
            "Результаты экспериментов" : result
        }

    def return_result(self):
        return  self.__stats["Результаты экспериментов"]

    def generate_table(self, borders: list):
        results = self.__stats["Результаты экспериментов"]
        n = len(results)
        z_array = np.zeros(shape=(len(borders)-1), dtype=float)
        f_array = np.zeros(shape=(len(borders)-1), dtype=float)
        n_array = np.zeros(shape=(len(borders)-1), dtype=float)
        norm = 0.0
        for i in range(len(borders) - 1):
            z_array[i] = (borders[i] + borders[i + 1]) / 2
            f_array[i] = self.__f(z_array[i])
            for result in results:
                if borders[i] <= result < borders[i + 1]:
                    n_array[i] += 1
            length = borders[i + 1] - borders[i]
            n_array[i] = n_array[i] / (n * length)
            norm = max(abs(n_array[i] - f_array[i]), norm)
        return z_array, f_array, n_array, norm

    def print_table(self, borders: list):
        z_array, f_array, n_array, norm = self.generate_table(borders)
        print("max(Nj / N * |/\'j| - Fn(Zj)) = {0}".format(norm))
        for z in z_array:
            print("|{0:10.5f}".format(z), end="")
        print("|")
        for f in f_array:
            print("|{0:10.5f}".format(f), end="")
        print("|")
        for n in n_array:
            print("|{0:10.5f}".format(n), end="")
        print("|")

    def __truncate(self, x: float):
        count = 4
        return int(x / 10 ** -count) * 10 ** -count

    def __f(self, y):
        result = 0.0
        if (y >= self.__a):
            result = self.__k * math.exp(-self.__k * (y - self.__a)) 
        return result

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

def main():
    experiment_count, device_count, q, d, seed = parse()
    solver = task18(device_count, q, d)
    solver.set_seed(seed)
    solver.make_experemets(experiment_count)
    solver.print_info()
    solver.print_statistic()
    solver.print_table([0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0])

if __name__ == "__main__":
    main()