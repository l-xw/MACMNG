from argument import up_configs
import torch
from Instance import *
from utils_DFFJSP import *


if __name__ == '__main__':
    path = 'example/'
    data_list = ['data/data1.xlsx', 'data/data2.xlsx', 'data/data3.xlsx', 'data/data4.xlsx', 'data/data5.xlsx']
    PMK_list = ['PMK/MK01.fjs', 'PMK/MK02.fjs', 'PMK/MK03.fjs', 'PMK/MK04.fjs', 'PMK/MK05.fjs', 'PMK/MK06.fjs',
                'PMK/MK07.fjs', 'PMK/MK08.fjs', 'PMK/MK09.fjs', 'PMK/MK10.fjs']
    FMK_list = ['FMK/FMK01.txt', 'FMK/FMK02.txt', 'FMK/FMK03.txt', 'FMK/FMK04.txt', 'FMK/FMK05.txt', 'FMK/FMK06.txt',
                'FMK/FMK07.txt', 'FMK/FMK08.txt', 'FMK/FMK10.txt']
    reman_list = ['remanu/remanu01.txt', 'remanu/remanu02.txt', 'remanu/remanu03.txt', 'remanu/remanu04.txt',
                  'remanu/remanu05.txt', 'remanu/remanu06.txt', 'remanu/remanu07.txt', 'remanu/remanu08.txt']
    example_list = [FMK_list]
    example_list = [item for sublist in example_list for item in sublist]

    show_PMK = ['PMK01', 'PMK02', 'PMK03', 'PMK04', 'PMK05', 'PMK06', 'PMK07', 'PMK08', 'PMK09', 'PMK10']
    show_data = ['data1', 'data2', 'data3', 'data4', 'data5']
    show_FMK = ['FMK01', 'FMK02', 'FMK03', 'FMK04', 'FMK05', 'FMK06', 'FMK07', 'FMK08', 'FMK10']
    show_reman = ['remanu01', 'remanu02', 'remanu03', 'remanu04', 'remanu05', 'remanu06', 'remanu07', 'remanu08']
    show_list = [show_FMK]
    show_list = [item for sublist in show_list for item in sublist]

    for fa in range(2, 3):
        for index in range(1, 2):
            print(f"{show_list[index]}_{fa}")
            factor_number, job_number, operation_number, machine_number, total_fuzzy_pt = read_example(rf'{path + example_list[index]}', example_list[index][:3])
            factor_number = fa
            MK_W = 1
            res = []
            MK = []
            TEC = []
            while MK_W >= 0:
                up_configs(factor_number, job_number, operation_number, machine_number)
                path_fac = rf'res/{show_list[index]}_{factor_number}/factor_agent{MK_W}.pt'
                path_job = rf'res/{show_list[index]}_{factor_number}/job_agent{MK_W}.pt'
                path_mac = rf'res/{show_list[index]}_{factor_number}/machine_agent{MK_W}.pt'
                fac_agent = torch.load(path_fac)
                job_agent = torch.load(path_job)
                mac_agent = torch.load(path_mac)
                fac_agent.epsilon = 1
                job_agent.epsilon = 1
                mac_agent.epsilon = 1

                print(f"Objective weights: {MK_W}")

                shop = Shop(factor_number, job_number, machine_number, operation_number, total_fuzzy_pt)
                action_job = 0
                SSE = [[-1 for _ in range(job_number)], [], [-1 for _ in range(job_number * operation_number)],
                       [-1 for _ in range(job_number * operation_number)]]
                while not shop.finish:
                    action_job = job_agent.get_action(shop.job_features, shop.legitimate_job)
                    SSE[1].append(action_job)
                    action_factory = -1
                    reward_ = 0
                    if shop.job_list[action_job].belong_factor == -1:
                        shop.Refresh_makespan(action_job)
                        action_factory = fac_agent.get_action(shop.factory_features, shop.legitimate_factor)
                        shop.job_list[action_job].belong_factor = action_factory
                        SSE[0][action_job] = action_factory
                    shop.flash_mac(action_job)
                    action_machine = mac_agent.get_action(shop.mac_features, shop.legitimate_machine)
                    SSE[2][action_job * operation_number + shop.job_list[action_job].current_op] = action_machine % 5
                    SSE[3][action_job * operation_number + shop.job_list[action_job].current_op] = action_machine // 5
                    reward = shop.Step(action_job, action_machine)
                shop.energy_saving(SSE)
                res.append([value(get_max([f.makespan for f in shop.factor_list])), value(get_sum([f.TEC for f in shop.factor_list]))])
                MK.append(get_max([f.makespan for f in shop.factor_list]))
                TEC.append(get_sum([f.TEC for f in shop.factor_list]))
                MK_W = round(MK_W - 0.01, 2)

            res_index = find_pareto_front(res)
            res = [res[i] for i in res_index]

            print(f"Pareto solution:{res}")

            for i in range(len(MK)):
                if MK[i] == get_min(MK):
                    print(f"Minimum makespan : MK = {MK[i]} TEC = {TEC[i]}")
                    break
            for i in range(len(TEC)):
                if TEC[i] == get_min(TEC):
                    print(f"Minimum TEC : MK = {MK[i]} TEC = {TEC[i]}")
                    break
