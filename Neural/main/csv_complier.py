

def create_csv_file(data, file_name):
    with open(file_name, 'w') as f:
        index = 1
        for d in data:
            f.write(str(index) + "," + str(d) + '\n')
            index += 1
