from typing import *
from copy import deepcopy
from argument import configs
from utils_DFFJSP import *


class Factor(object):
    def __init__(self, index, machines):
        self.TEC = [0, 0, 0]
        self.index = index
        self.machines: List[Machine] = machines
        self.makespan = [0, 0, 0]
        self.all_makespan = [0, 0, 0]

        self.plant_load = [0, 0, 0]
        self.one_makespan = [0, 0, 0]


class Job(object):
    def __init__(self, index, op_nb, total_fuzzy_pt):
        self.index = index
        self.op_nb = op_nb
        self.fuzzy_pt = total_fuzzy_pt[index]
        self.belong_factor = -1

        self.current_op = 0
        self.current_time = [0, 0, 0]

        self.finish = 0
        self.total_processing_time_avg = [0, 0, 0]
        for operation, item in enumerate(self.fuzzy_pt):
            self.total_processing_time_avg = addition(self.total_processing_time_avg, get_avg(item))
        self.processed_time = [0, 0, 0]

    def Process(self, start_time, process_time):
        self.current_op += 1
        self.current_time = addition(start_time, process_time)

        if self.current_op >= self.op_nb:
            self.finish = 1
        else:
            self.processed_time = addition(self.processed_time, process_time)


class Operation(object):
    def __init__(self, belong_job, index, total_fuzzy_pt_job, total_processed_time, finish_job):
        self.belong_job = belong_job
        self.index = index

        self.processing_time = total_fuzzy_pt_job
        self.finish_job = finish_job
        self.total_processed_time = total_processed_time
        self.finish = 0


class Machine(object):
    def __init__(self, index):
        self.speeds = [0.6, 0.8, 1, 1.2, 1.4]
        self.energy = [8, 5, 3, 2, 1]
        self.current_gear = 0
        self.gear_list = []
        self.index = index
        self.current_time = [0, 0, 0]
        self.op_start_time = []
        self.op_end_time = []
        self.job_index = []
        self.op_index = []
        self.pluggable_index = 0
        self.pluggable = 0
        self.total_energy = [0, 0, 0]

        self.process_time = [0, 0, 0]
        self.process_energy = [0, 0, 0]
        self.ready_time = [0, 0, 0]
        self.machine_load = [0, 0, 0]
        self.utilization_rate = 0.0

    def Process(self, start_time, job_index, op_index):
        self.job_index.append(job_index)
        self.op_index.append(op_index)
        self.op_start_time.append(start_time)
        self.op_end_time.append(addition(start_time, self.process_time))
        standby_time = subtract(start_time, self.current_time)
        self.gear_list.append(self.current_gear)
        if not self.pluggable:
            self.current_time = addition(start_time, self.process_time)
            self.total_energy = addition(self.total_energy, multiplication(self.process_time, configs.unit_energy_consumption * self.energy[self.current_gear]))
            self.total_energy = addition(self.total_energy, multiplication(standby_time, 1))
        else:
            self.total_energy = addition(self.total_energy, multiplication(self.process_time, configs.unit_energy_consumption * self.energy[self.current_gear]))
            self.total_energy = subtract(self.total_energy, multiplication(self.process_time, 1))
        self.ready_time = deepcopy(self.current_time)
        self.machine_load = addition(self.machine_load, self.process_time)
        self.utilization_rate = value(self.machine_load) / value(self.ready_time) * 100


class Shop(object):
    def __init__(self, factor_nb, job_nb, machine_nb, op_nb, total_fuzzy_pt):
        self.factor_nb = factor_nb
        self.job_nb = job_nb
        self.machine_nb = machine_nb
        self.op_nb = op_nb
        self.total_fuzzy_pt = total_fuzzy_pt
        self.factor_list: List[Factor] = []
        self.job_list: List[Job] = []
        self.finish = 0
        self.Create()
        self.legitimate_job = [True for _ in range(self.job_nb)]
        self.legitimate_machine = [True for _ in range(self.machine_nb)]
        self.legitimate_factor = [True for _ in range(self.factor_nb)]

        self.factory_features = []

        self.job_features = [[], []]
        for job in self.job_list:

            self.job_features[0].append(job.finish)
            self.job_features[0].append(value(job.total_processing_time_avg))
            self.job_features[0].append(value(job.current_time))
            self.job_features[0].append(value(job.processed_time))

            self.job_features[1].append(job.finish)
            self.job_features[1].append(value(job.total_processing_time_avg))
            self.job_features[1].append(value(job.current_time))
            self.job_features[1].append(value(job.processed_time))

        self.mac_features = []

        self.gov_features = []

        for f in self.factor_list:
            for j in self.job_list:
                f.makespan = addition(f.makespan, j.total_processing_time_avg)
                f.TEC = addition(f.TEC, multiplication(f.makespan, configs.unit_energy_consumption))

    def Create(self):
        self.factor_list = []
        self.job_list = []
        machine_list = []
        for i in range(self.job_nb):
            self.job_list.append(Job(i, self.op_nb, self.total_fuzzy_pt))
        for i in range(self.machine_nb):
            machine_list.append(Machine(i))
        for i in range(self.factor_nb):
            self.factor_list.append(Factor(i, deepcopy(machine_list)))

    def Step(self, job_index, machine_index):
        job = self.job_list[job_index]
        factor = self.factor_list[job.belong_factor]
        machine = factor.machines[machine_index // 5]
        machine.current_gear = machine_index % 5
        process_time = job.fuzzy_pt[job.current_op][machine.index]
        process_time = multiplication(process_time, machine.speeds[machine.current_gear])
        machine.process_time = deepcopy(process_time)
        start_time = rank(job.current_time, machine.current_time)
        machine_time = machine.current_time
        machine.pluggable = 0
        if equality(start_time, machine_time):
            for fragment in range(0, len(machine.op_start_time)):
                if insertable(job.current_time, machine.op_end_time[fragment - 1]):
                    if insertable(subtract(machine.op_start_time[fragment], job.current_time),
                                  process_time):
                        machine.pluggable = 1

        if sum(process_time) > 0:
            if machine.pluggable:
                machine.Process(job.current_time, job_index, job.current_op)
                job.Process(job.current_time, process_time)
            else:
                machine.Process(start_time, job_index, job.current_op)
                job.Process(start_time, process_time)
        else:
            job.Process(start_time, process_time)

        if job.finish == 1: self.legitimate_job[job.index] = False

        self.job_features = [[], []]
        for job in self.job_list:
            self.job_features[0].append(job.finish)
            self.job_features[0].append(value(job.total_processing_time_avg))
            self.job_features[0].append(value(job.current_time))
            self.job_features[0].append(value(job.processed_time))

            self.job_features[1].append(job.finish)
            self.job_features[1].append(value(job.total_processing_time_avg))
            self.job_features[1].append(value(job.current_time))
            self.job_features[1].append(value(job.processed_time))

        self.finish = 1
        for j in self.job_list:
            if j.finish == 0:
                self.finish = 0
                break

        select_makespan = deepcopy(factor.makespan)
        select_TEC = deepcopy(factor.TEC)

        for f in self.factor_list:
            f.makespan = get_max([m.current_time for m in f.machines])
            f.TEC = get_sum([m.total_energy for m in f.machines])
            for job_ in self.job_list:
                if job_.belong_factor == -1:
                    f.makespan = addition(f.makespan, job_.total_processing_time_avg)
                    f.TEC = addition(f.TEC, multiplication(job_.total_processing_time_avg, configs.unit_energy_consumption))
                if job_.belong_factor == f.index:
                    temp = [0, 0, 0]
                    for i in range(job_.current_op, len(job_.fuzzy_pt)):
                        temp = addition(temp, get_avg(job_.fuzzy_pt[i]))
                    f.makespan = addition(f.makespan, temp)
                    f.TEC = addition(f.TEC, multiplication(temp, configs.unit_energy_consumption))

        for f in self.factor_list:
            f.makespan = get_max([m.current_time for m in f.machines])
            f.TEC = get_sum([m.total_energy for m in f.machines])
            for m in f.machines:
                f.TEC = addition(f.TEC, multiplication(subtract(f.makespan, m.current_time), 1))

        reward_makespan = subtract(select_makespan, factor.makespan)
        reward_TEC = subtract(select_TEC, factor.TEC)
        return [sum(reward_makespan), sum(reward_TEC)]

    def flash_mac(self, job_index):
        factor = self.factor_list[self.job_list[job_index].belong_factor]
        job = self.job_list[job_index]
        self.legitimate_machine = []
        for index, item in enumerate(job.fuzzy_pt[job.current_op]):
            factor.machines[index].process_time = item
            if sum(item) > 0:
                for _ in range(len(factor.machines[index].speeds)):
                    self.legitimate_machine.append(True)
            else:
                for _ in range(len(factor.machines[index].speeds)):
                    self.legitimate_machine.append(False)

        self.mac_features = [[], []]
        for machine in factor.machines:
            for gear in range(len(machine.speeds)):
                self.mac_features[0].append(machine.utilization_rate)
                self.mac_features[0].append(sum(machine.current_time))
                self.mac_features[0].append(sum(machine.machine_load))
                self.mac_features[1].append(machine.utilization_rate)
                self.mac_features[1].append(sum(machine.current_time))
                self.mac_features[1].append(sum(machine.machine_load))
                self.mac_features[0].append(sum(multiplication(machine.process_time, machine.speeds[gear])))
                self.mac_features[1].append(sum(multiplication(multiplication(machine.process_time, machine.speeds[gear]), machine.energy[gear] * configs.unit_energy_consumption)))
        pass

    def flash_gov(self, job_index, machine_index):
        factor = self.factor_list[self.job_list[job_index].belong_factor]
        machine = factor.machines[machine_index]
        self.gov_features = []
        for gear in range(len(machine.energy)):
            self.gov_features.append(sum(multiplication(machine.process_time, machine.speeds[gear])))
            self.gov_features.append(sum(multiplication(multiplication(machine.process_time, machine.speeds[gear]), machine.energy[gear] * configs.unit_energy_consumption)))
        pass

    def gov_reward(self, job_index, machine_index, gear):
        job = self.job_list[job_index]
        factor = self.factor_list[job.belong_factor]
        machine = factor.machines[machine_index]
        process_time = job.fuzzy_pt[job.current_op][machine.index]
        TEC = multiplication(process_time, configs.unit_energy_consumption * machine.energy[machine.current_gear])

        process_time_G = multiplication(process_time, machine.speeds[gear])
        TEC_G = multiplication(process_time_G, configs.unit_energy_consumption * machine.energy[gear])
        machine.current_gear = gear
        return [round(value(process_time) - value(process_time_G), 2), round(value(TEC) - value(TEC_G), 2)]

    def Refresh_makespan(self, choose_job_index):
        shop_ = deepcopy(self)

        for job_index in range(choose_job_index):
            if shop_.job_list[job_index].belong_factor != -1:
                factor = shop_.factor_list[shop_.job_list[job_index].belong_factor]
                job = shop_.job_list[job_index]
                while not job.finish:
                    usable_machines_index = [index for index in range(len(job.fuzzy_pt[job.current_op])) if
                                             sum(job.fuzzy_pt[job.current_op][index]) >= 0]
                    machine = factor.machines[usable_machines_index[0]]
                    min_load_machines_index = []
                    for index in usable_machines_index:
                        if not equality(rank(machine.machine_load, factor.machines[index].machine_load),
                                        factor.machines[index].machine_load):
                            machine = factor.machines[index]

                    for index in usable_machines_index:
                        if equality(rank(machine.machine_load, factor.machines[index].machine_load),
                                    machine.machine_load):
                            min_load_machines_index.append(index)

                    if len(min_load_machines_index) == 1:
                        shop_.Step(job.index, machine.index)
                    else:
                        machine = factor.machines[min_load_machines_index[0]]
                        for index in min_load_machines_index:
                            if not equality(rank(job.fuzzy_pt[job.current_op][machine.index],
                                                 job.fuzzy_pt[job.current_op][index]),
                                            job.fuzzy_pt[job.current_op][index]):
                                machine = factor.machines[index]
                        shop_.Step(job.index, machine.index)

        temp = deepcopy(shop_.job_list[choose_job_index])
        for factor in shop_.factor_list:
            job = shop_.job_list[choose_job_index]
            job.belong_factor = factor.index
            while not job.finish:
                usable_machines_index = [index for index in range(len(job.fuzzy_pt[job.current_op])) if sum(job.fuzzy_pt[job.current_op][index]) >= 0]
                machine = factor.machines[usable_machines_index[0]]
                min_load_machines_index = []

                for index in usable_machines_index:
                    if not equality(rank(machine.machine_load, factor.machines[index].machine_load), factor.machines[index].machine_load):
                        machine = factor.machines[index]

                for index in usable_machines_index:
                    if equality(rank(machine.machine_load, factor.machines[index].machine_load), machine.machine_load):
                        min_load_machines_index.append(index)

                if len(min_load_machines_index) == 1:
                    shop_.Step(job.index, machine.index)
                else:
                    machine = factor.machines[min_load_machines_index[0]]
                    for index in min_load_machines_index:
                        if not equality(rank(job.fuzzy_pt[job.current_op][machine.index], job.fuzzy_pt[job.current_op][index]), job.fuzzy_pt[job.current_op][index]):
                            machine = factor.machines[index]
                    shop_.Step(job.index, machine.index)
            self.factor_list[factor.index].one_makespan = factor.makespan
            shop_.job_list[choose_job_index] = deepcopy(temp)

        job_list_ = deepcopy(shop_.job_list)
        for factor in shop_.factor_list:
            for job_index in range(choose_job_index + 1, len(shop_.job_list)):
                job = shop_.job_list[job_index]
                job.belong_factor = factor.index
                while not job.finish:
                    usable_machines_index = [index for index in range(len(job.fuzzy_pt[job.current_op])) if
                                             sum(job.fuzzy_pt[job.current_op][index]) >= 0]
                    machine = factor.machines[usable_machines_index[0]]
                    min_load_machines_index = []

                    for index in usable_machines_index:
                        if not equality(rank(machine.machine_load, factor.machines[index].machine_load),
                                        factor.machines[index].machine_load):
                            machine = factor.machines[index]

                    for index in usable_machines_index:
                        if equality(rank(machine.machine_load, factor.machines[index].machine_load),
                                    machine.machine_load):
                            min_load_machines_index.append(index)

                    if len(min_load_machines_index) == 1:
                        shop_.Step(job.index, machine.index)
                    else:
                        machine = factor.machines[min_load_machines_index[0]]
                        for index in min_load_machines_index:
                            if not equality(rank(job.fuzzy_pt[job.current_op][machine.index],
                                                 job.fuzzy_pt[job.current_op][index]),
                                            job.fuzzy_pt[job.current_op][index]):
                                machine = factor.machines[index]
                        shop_.Step(job.index, machine.index)
            self.factor_list[factor.index].all_makespan = factor.makespan
            shop_.job_list = deepcopy(job_list_)
        self.flash_fac()

    def flash_fac(self):
        self.factory_features = []
        for factor in self.factor_list:
            self.factory_features.append(sum(factor.plant_load))
            self.factory_features.append(sum(factor.one_makespan))

    def reset(self):
        self.factor_list: List[Factor] = []
        self.job_list: List[Job] = []
        self.finish = 0
        self.Create()

    def decode(self, SSE):
        self.reset()
        FV = SSE[0]
        JV = SSE[1]
        SV = SSE[2]
        MV = SSE[3]
        for index in range(len(JV)):
            job = self.job_list[JV[index]]
            if job.belong_factor == -1:
                job.belong_factor = FV[job.index]
            factor = self.factor_list[job.belong_factor]
            machine = factor.machines[MV[job.index * job.op_nb + job.current_op]]
            machine.current_gear = SV[job.index * job.op_nb + job.current_op]
            self.Step(job.index, machine.index * 5 + machine.current_gear)

        return value(get_max([f.makespan for f in self.factor_list])), value(get_sum([f.TEC for f in self.factor_list]))

    def energy_saving(self, SSE):
        shop_ = deepcopy(self)
        Cmax = value(get_max([f.makespan for f in self.factor_list]))
        for i in range(len(SSE[1])):
            if SSE[2][i] < 4:
                temp_SSE = deepcopy(SSE)
                temp_cmax = Cmax
                while temp_cmax == Cmax and temp_SSE[2][i] < 4:
                    temp_SSE[2][i] += 1
                    shop_.decode(temp_SSE)
                    temp_cmax = value(get_max([f.makespan for f in shop_.factor_list]))
                    pass
                if temp_cmax > Cmax: temp_SSE[2][i] -= 1
                SSE = deepcopy(temp_SSE)
        self.decode(SSE)
