import math
import matplotlib.pyplot as plt


def get_real_output(binary_vector):
    if binary_vector[0] == 1 or binary_vector[1] == 1 or binary_vector[3] == 1:
        if binary_vector[2] == 1:
            return 1
        else:
            return 0
    return 0


# Вычисляю дельту весового коэффициента
def get_delta_weight(norma, error, output_rbf):
    return norma * error * output_rbf


# получаю двоичные векторы десятичных чисел от 0 до 16
def get_input_vectors():
    output = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    for it in range(0, 16):
        binary_number = bin(it)
        binary_number = int(binary_number[2:])
        if binary_number < 10:
            binary_number = str(binary_number)
            output[it].append(0)
            output[it].append(0)
            output[it].append(0)
            output[it].append(int(binary_number))
        elif binary_number < 100:
            binary_number = str(binary_number)
            output[it].append(0)
            output[it].append(0)
            output[it].append(int(binary_number[0]))
            output[it].append(int(binary_number[1]))
        elif binary_number < 1000:
            binary_number = str(binary_number)
            output[it].append(0)
            output[it].append(int(binary_number[0]))
            output[it].append(int(binary_number[1]))
            output[it].append(int(binary_number[2]))
        else:
            binary_number = str(binary_number)
            output[it].append(int(binary_number[0]))
            output[it].append(int(binary_number[1]))
            output[it].append(int(binary_number[2]))
            output[it].append(int(binary_number[3]))
    return output


class neuron:
    def __init__(self):
        self.weights = [0, 0, 0, 0, 0, 0, 0, 0]
        self.center_coordinate = [[], [0, 0, 1, 1], [0, 1, 1, 0], [0, 1, 1, 1], [1, 0, 1, 0], [1, 0, 1, 1],
                                  [1, 1, 1, 0],
                                  [1, 1, 1, 1]]

    # Корректирую весовые коэффициенты согласно правилу Видроу-Хоффа
    def correction_weights(self, delta_weights=None):
        for it in range(0, 8):
            self.weights[it] += delta_weights[it]

    # Вычисление выхода теневого нейрончика
    def get_output_rbf(self, input_vectors, index):
        summ = 0
        for it in range(0, 4):
            summ -= (input_vectors[it] - self.center_coordinate[index][it]) ** 2
        return math.exp(summ)

    # Вычисление сетевого входа
    def get_network(self, rbf_array):
        net = 0
        for it in range(0, 8):
            net += rbf_array[it] * self.weights[it]
        return net


def correction_neuron(my_neuron, error, rbf_array):
    norma = 0.3
    delta_array = []
    for index in range(0, 8):
        delta_array.append(get_delta_weight(norma, error, rbf_array[index]))
    my_neuron.correction_weights(delta_array)


# вычисляю выход нейронной сети
def get_neuron_output(my_neuron, binary_vector):
    rbf_array = [1]
    for index in range(1, 8):
        rbf_array.append(my_neuron.get_output_rbf(binary_vector, index))
    net = my_neuron.get_network(rbf_array)
    if net >= 0:
        return 1, rbf_array
    else:
        return 0, rbf_array


def print_output(my_neuron, arr_error):
    plt.plot(arr_error)
    plt.show()
    print(my_neuron.weights)


def start_learning():
    my_neuron = neuron()
    input_vectors = get_input_vectors()
    epoche = 0
    arr_error = []
    while True:
        quadratic_error = 0
        for it in range(0, 16):
            real_output = get_real_output(input_vectors[it])
            neuron_output, rbf_array = get_neuron_output(my_neuron, input_vectors[it])
            error = real_output - neuron_output
            if real_output != neuron_output:
                quadratic_error += 1
            correction_neuron(my_neuron, error, rbf_array)
        epoche += 1
        arr_error.append(quadratic_error)
        if quadratic_error == 0:
            break
    return my_neuron, arr_error


if __name__ == '__main__':
    my_neuron, arr_error = start_learning()
    print_output(my_neuron, arr_error)
