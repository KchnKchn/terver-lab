from  PyQt5 import QtWidgets, QtGui, QtCore
from impl.task18 import task18
import matplotlib.pyplot as plt
import numpy as np

class TaskParametersGroup(QtWidgets.QGroupBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Параметры задачи")
        self.__grid = QtWidgets.QGridLayout(self)
        self.__init_seed_parameter(0)
        self.__init_device_parameter(1)
        self.__init_q_parameter(2)
        self.__init_d_parameter(3)
        self.__init_button(4)
        self.__solver = None

    def set_solver(self, solver: task18):
        self.__solver = solver

    def __init_seed_parameter(self, str_number: int):
        label = QtWidgets.QLabel()
        label.setText("Значение для построения псевдо-случайной последовательности")
        self.__seed_input = QtWidgets.QLineEdit()
        self.__seed_input.setText("")
        self.__grid.addWidget(label, str_number, 0)
        self.__grid.addWidget(self.__seed_input, str_number, 1)

    def __init_device_parameter(self, str_number: int):
        label = QtWidgets.QLabel()
        label.setText("Количество устройств")
        self.__device_input = QtWidgets.QLineEdit()
        self.__device_input.setText("10")
        self.__grid.addWidget(label, str_number, 0)
        self.__grid.addWidget(self.__device_input, str_number, 1)

    def __init_q_parameter(self, str_number: int):
        label = QtWidgets.QLabel()
        label.setText("Математическое ожидание с.в.")
        self.__q_input = QtWidgets.QLineEdit()
        self.__q_input.setText("10")
        self.__grid.addWidget(label, str_number, 0)
        self.__grid.addWidget(self.__q_input, str_number, 1)

    def __init_d_parameter(self, str_number: int):
        label = QtWidgets.QLabel()
        label.setText("Дисперсия с.в.")
        self.__d_input = QtWidgets.QLineEdit()
        self.__d_input.setText("5")
        self.__grid.addWidget(label, str_number, 0)
        self.__grid.addWidget(self.__d_input, str_number, 1)

    def __button_clicked(self):
        device_count = int(self.__device_input.text())
        q = float(self.__q_input.text())
        d = float(self.__d_input.text())
        seed = self.__seed_input.text()
        self.__solver.set_parameters(device_count, q, d)
        if seed:
            self.__solver.set_seed(seed)

    def __init_button(self, str_number: int):
        button = QtWidgets.QPushButton()
        button.setText("Ввести параметры")
        button.clicked.connect(self.__button_clicked)
        self.__grid.addWidget(button, str_number, 0, 1, 2)

class ExperimentsParametersGroup(QtWidgets.QGroupBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Параметры эксперимента")
        self.__grid = QtWidgets.QGridLayout(self)
        self.__init_experiments_parameter(0)
        self.__init_borders_parameter(1)
        self.__init_button(2)
        self.__solver = None

    def set_solver(self, solver: task18):
        self.__solver = solver

    def __init_experiments_parameter(self, str_number: int):
        label = QtWidgets.QLabel()
        label.setText("Количество экспериментов")
        self.__experiments_input = QtWidgets.QLineEdit()
        self.__experiments_input.setText("1000")
        self.__grid.addWidget(label, str_number, 0)
        self.__grid.addWidget(self.__experiments_input, str_number, 1)

    def __init_borders_parameter(self, str_number: int):
        label = QtWidgets.QLabel()
        label.setText("Границы")
        self.__borders_input = QtWidgets.QLineEdit()
        self.__borders_input.setText("77.5 82.5 87.5 92.5 97.5 102.5 107.5 112.5 117.5 122.5 127.5")
        self.__grid.addWidget(label, str_number, 0)
        self.__grid.addWidget(self.__borders_input, str_number, 1)

    def set_update_tables(self, f):
        self.__update_tables = f

    def __button_clicked(self):
        #parse forms
        experiments_count = int(self.__experiments_input.text())
        borders = np.asarray([float(x) for x in (self.__borders_input.text()).split(" ")], dtype=float)

        #make experiments
        result = self.__solver.make_experemets(experiments_count)

        #print tables
        metrics = self.__solver.get_metrics(result)
        z, f, n, norm = self.__solver.get_histogram(result, borders)
        self.__update_tables(result, metrics, z, f, n, norm)

        #print graphics
        plt.close("all")
        F, Fc, norm = self.__solver.get_graphics(result)
        fig, axs = plt.subplots(2, 1, num="Графики")

        axs[0].set_title("Функция распределения и выборочная функция распределения")
        axs[0].set_xlabel("Значения случайной величины")
        axs[0].set_ylabel("Вероятность события")
        axs[0].grid(True)
        axs[0].plot(result, F, label="Аналитическая функция распределения")
        axs[0].step(result, Fc, label="Выборочная функция распределения")
        axs[0].legend(title="D = {0:0.3f}".format(norm), loc="lower right")

        axs[1].set_title("Гистограмма")
        axs[1].grid(True)
        len = np.asarray([borders[i + 1] - borders[i] for i in range(borders.shape[0] - 1)], dtype=float)
        axs[1].bar(borders[:-1], n, width=len, align="edge")
        plt.show()

    def __init_button(self, str_number: int):
        button = QtWidgets.QPushButton()
        button.setText("Запуск экспериментов")
        button.clicked.connect(self.__button_clicked)
        self.__grid.addWidget(button, str_number, 0, 1, 2)

class ResultTable(QtWidgets.QTableWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

    def __create_cell(self, text: str):
        cell = QtWidgets.QTableWidgetItem(text)
        cell.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
        return cell

    def update_table(self, result: np.array):
        self.clear()
        self.setRowCount(1)
        self.setColumnCount(len(result))
        for i in range(len(result)):
            self.setItem(0, i, self.__create_cell(str(result[i])))
        self.resizeColumnsToContents()

class MetricsTable(QtWidgets.QTableWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRowCount(1)
        self.setColumnCount(8)
        self.setHorizontalHeaderLabels(("Eη", "x", "|Eη - x|", "Dη", "S^2", "|Dη - S^2|", "Me", "R"))

    def __create_cell(self, text: str):
        cell = QtWidgets.QTableWidgetItem(text)
        cell.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
        return cell

    def update_table(self, metrics: np.array):
        for i in range(8):
            self.setItem(0, i, self.__create_cell(str(metrics[i])))
        self.resizeColumnsToContents()

class HistogramTable(QtWidgets.QTableWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

    def __create_cell(self, text: str):
        cell = QtWidgets.QTableWidgetItem(text)
        cell.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
        return cell

    def update_table(self, z_array: np.array, f_array: np.array, n_array: np.array):
        self.clear()
        self.setRowCount(3)
        self.setColumnCount(z_array.shape[0])
        for i in range(z_array.shape[0]):
            self.setItem(0, i, self.__create_cell(str(z_array[i])))
            self.setItem(1, i, self.__create_cell(str(f_array[i])))
            self.setItem(2, i, self.__create_cell(str(n_array[i])))
        self.resizeColumnsToContents()

class ResultGroup(QtWidgets.QGroupBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Результаты серии экспериментов")
        self.__grid = QtWidgets.QGridLayout(self)
        self.__init_result_table(0)
        self.__init_metrics_table(2)
        self.__init_result_histogram(4)
        self.__init_max(6)

    def __init_result_table(self, str_number: int):
        label = QtWidgets.QLabel()
        label.setText("Значения случайной величины")
        self.__result = ResultTable()
        self.__grid.addWidget(label, str_number, 0, 1, 2)
        self.__grid.addWidget(self.__result, str_number + 1, 0, 1, 2)

    def __init_metrics_table(self, str_number: int):
        label = QtWidgets.QLabel()
        label.setText("Метрики случайной величины")
        self.__metrics = MetricsTable()
        self.__grid.addWidget(label, str_number, 0, 1, 2)
        self.__grid.addWidget(self.__metrics, str_number + 1, 0, 1, 2)

    def __init_result_histogram(self, str_number: int):
        label = QtWidgets.QLabel()
        label.setText("Гистограмма")
        self.__histogram = HistogramTable()
        self.__grid.addWidget(label, str_number, 0, 1, 2)
        self.__grid.addWidget(self.__histogram, str_number + 1, 0, 1, 2)

    def __init_max(self, str_number: int):
        label = QtWidgets.QLabel()
        label.setText("max(Nj / N * |/\'j| - Fn(Zj)) =")
        self.__max = QtWidgets.QLineEdit()
        self.__max.setReadOnly(True)
        self.__grid.addWidget(label, str_number, 0)
        self.__grid.addWidget(self.__max, str_number, 1)
    
    def update_tables(self, result: np.array, metrics: np.array, z: np.array, f: np.array, n: np.array, norm: float):
        self.__result.update_table(result)
        self.__metrics.update_table(metrics)
        self.__histogram.update_table(z, f, n)
        self.__max.setText(str(norm))

class MainWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.__solver = task18()
        self.__grid = QtWidgets.QGridLayout(self)

        self.__task_parameters = TaskParametersGroup()
        self.__task_parameters.set_solver(self.__solver)
        self.__grid.addWidget(self.__task_parameters, 0, 0)

        self.__experiment_parameters = ExperimentsParametersGroup()
        self.__experiment_parameters.set_solver(self.__solver)
        self.__grid.addWidget(self.__experiment_parameters, 1, 0)

        self.__info = QtWidgets.QTextEdit()
        self.__info.setReadOnly(True)
        self.__info.setText(
        "Задача №18\n\
Устройство состоит из N >> 1 дублирующих приборов. \
Каждый следующий прибор включается после выхода из строя предыдущего. \
Время безотказной работы каждого прибора — положительная с.в. со средним \
Q и дисперсией R. С.в. η — время безотказной работы всего устройства.\n\
\n\
ξ - с.в. характеризующая время безотказной работы одного прибора\
\n\
              k * exp(-k * (y - a)), y >= a\n\
fξ(x) = \n\
              0, иначе\n\
\n\
a = Q - sqrt(D)\n\
k = 1 / sqrt(D)"
        )
        self.__grid.addWidget(self.__info, 0, 1, 2, 1)

        self.__result = ResultGroup()
        self.__grid.addWidget(self.__result, 2, 0, 1, 2)

        self.__experiment_parameters.set_update_tables(self.__result.update_tables)

class GUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(1280, 720)
        self.setWindowTitle("Лабораторная работа по теории вероятности")
        self.__central = MainWidget(self)
        self.setCentralWidget(self.__central)
