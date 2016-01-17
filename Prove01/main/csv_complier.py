import re
import os


def append_to_file(file_name, data):
    with open(file_name, 'a') as results:
        results.write(data)


def create_csv_file(file_name, write_file_name):
    if not os.path.isfile(write_file_name):
        result = "k,MyResults,BetterResults\n"
        append_to_file(write_file_name, result)

    with open(file_name, 'r') as results:
        line_number = 1
        for line in results:
            pattern = re.compile(r'\s+')
            line = re.sub(pattern, '', line)

            result = line.replace('=', ':').replace('%', '').split(':')[1]

            if line_number is not 3:
                result += ","
            else:
                result += "\n"

            append_to_file(write_file_name, result)

            line_number += 1

for index in range(1, 106):
    create_csv_file(os.getcwd() + os.sep + ".." + os.sep + str(index) + "-iris_results.txt", os.getcwd() + os.sep + ".." + os.sep + "iris-results.csv")
    create_csv_file(os.getcwd() + os.sep + ".." + os.sep + str(index) + "-car_results.txt", os.getcwd() + os.sep + ".." + os.sep + "car-results.csv")