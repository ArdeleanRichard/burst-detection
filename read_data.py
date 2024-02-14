# A comparison of computational methods for detecting bursts in neuronal spike trains and their application to human stem cell-derived neuronal networks
# https://github.com/ellesec/burstanalysis/tree/master/Simulation_results


import csv
import rdata
from constants import DATA_PATH


def write_csv(file_name, data):
    csv_file = open(f"{file_name}.csv", "w", newline='')
    writer = csv.writer(csv_file)
    for values in data:
        writer.writerow(values)


def read_and_save():
    parsed = rdata.parser.parse_file(DATA_PATH + "/sim_data.RData")
    converted = rdata.conversion.convert(parsed)

    for data_name in converted.keys():
        # sim.data
        # print(data_name)
        for data_type in converted[data_name].keys():
            # non.bursting
            # non.stationary
            # reg.bursting
            # long.bursts
            # high.freq
            # noisy.bursts
            # comp.time
            print(data_type)
            # if data_type == 'noisy.bursts':
            data = []
            for i in range(len(converted[data_name][data_type][0].keys())):
                data.append([])

                ### noisy.bursts doesnt create num.bursts
                # if i == len(converted[data_name][data_type][0].keys()) - 1:
                #     data.append([])

            for list_elem in converted[data_name][data_type]:
                # print(list_elem)
                # spks
                # burst.beg
                # burst.end
                # num.bursts

                for id, simulation_key in enumerate(list_elem.keys()):
                    data[id].append(list_elem[simulation_key])

                    ### noisy.bursts doesnt create num.bursts
                    # if id == len(list_elem.keys()) - 1:
                    #     data[id+1].append([len(list_elem[simulation_key])])

                # print(list_elem['spks'])
                # print(list_elem['burst.beg'])
                # print(list_elem['burst.end'])
                # print(list_elem['num.bursts'])

            auxs = ['spks', 'burst.beg', 'burst.end', 'num.bursts']
            for id, file_data in enumerate(data):
                write_csv(DATA_PATH+f"{data_type}.{auxs[id]}", file_data)


if __name__ == "__main__":
    read_and_save()


