from argument import configs
import random


def Generating_example(job_number=configs.job_number, operation_number=configs.operation_number, machine_number=configs.machine_number):
    total_fuzzy_pt = []
    for job in range(job_number):
        temp_operation = []
        for operation in range(operation_number):
            temp_machine = []
            # unavailable_machine = random.sample(list(range(machine_number)), random.randint(1, machine_number // 2))
            unavailable_machine = 0
            for machine in range(machine_number):
                if unavailable_machine != 0 and machine in unavailable_machine:
                    temp_machine.append([-1, -1, -1])
                else:
                    # s = random.randint(configs.threshold_time, configs.upper_limit_time)
                    fuzzy = []
                    while len(fuzzy) != 3:
                        s = random.randint(configs.threshold_time, configs.upper_limit_time)
                        while s in fuzzy:
                            s = random.randint(configs.threshold_time, configs.upper_limit_time)
                        fuzzy.append(s)
                    fuzzy.sort()
                    temp_machine.append(fuzzy)
            temp_operation.append(temp_machine)
        total_fuzzy_pt.append(temp_operation)
    return total_fuzzy_pt
