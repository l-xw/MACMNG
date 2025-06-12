import os
from Data_generation import Generating_example
from argument import up_configs
from ReplayBuffer import *
import torch
from Instance import *
from DQN import DQN, MODQN
import numpy as np
from tqdm import tqdm


"""
                                                      ___====-_  _-====___
                                                _--^^^#####//      \\#####^^^--_
                                             _-^##########// (    ) \\##########^-_
                                            -############//  |\^^/|  \\############-
                                          _/############//   (@::@)   \\############\_
                                         /#############((     \\//     ))#############\
                                        -###############\\    (oo)    //###############-
                                       -#################\\  / VV \  //#################-
                                      -###################\\/      \//###################-
                                     _#/|##########/\######(   /\   )######/\##########|\#_
                                     |/ |#/\#/\#/\/  \#/\##\  |  |  /##/\#/  \/\#/\#/\#| \|
                                     `  |/  V  V  `   V  \#\| |  | |/#/  V   '  V  V  \|  '
                                        `   `  `      `   / | |  | | \   '      '  '   '
                                                         (  | |  | |  )
                                                        __\ | |  | | /__
                                                       (vvv(VVV)(VVV)vvv)
                                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                                    神兽保佑       永无BUG
                                                        作者——李小伟
"""


if __name__ == '__main__':
    path = 'example/'
    data_list = ['data/data1.xlsx', 'data/data2.xlsx', 'data/data3.xlsx', 'data/data4.xlsx', 'data/data5.xlsx']
    PMK_list = ['PMK/MK01.fjs', 'PMK/MK02.fjs', 'PMK/MK03.fjs', 'PMK/MK04.fjs', 'PMK/MK05.fjs', 'PMK/MK06.fjs',
                'PMK/MK07.fjs', 'PMK/MK08.fjs', 'PMK/MK09.fjs', 'PMK/MK10.fjs']
    FMK_list = ['FMK/FMK01.txt', 'FMK/FMK02.txt', 'FMK/FMK03.txt', 'FMK/FMK04.txt', 'FMK/FMK05.txt', 'FMK/FMK06.txt',
                'FMK/FMK07.txt', 'FMK/FMK08.txt', 'FMK/FMK09.txt', 'FMK/FMK10.txt']
    reman_list = ['remanu/remanu01.txt', 'remanu/remanu02.txt', 'remanu/remanu03.txt', 'remanu/remanu04.txt',
                  'remanu/remanu05.txt', 'remanu/remanu06.txt', 'remanu/remanu07.txt', 'remanu/remanu08.txt']
    example_list = [FMK_list]
    example_list = [item for sublist in example_list for item in sublist]

    show_PMK = ['PMK01', 'PMK02', 'PMK03', 'PMK04', 'PMK05', 'PMK06', 'PMK07', 'PMK08', 'PMK09', 'PMK10']
    show_data = ['data1', 'data2', 'data3', 'data4', 'data5']
    show_FMK = ['FMK01', 'FMK02', 'FMK03', 'FMK04', 'FMK05', 'FMK06', 'FMK07', 'FMK08', 'FMK09', 'FMK10']
    show_reman = ['remanu01', 'remanu02', 'remanu03', 'remanu04', 'remanu05', 'remanu06', 'remanu07', 'remanu08']
    show_list = [show_FMK]
    show_list = [item for sublist in show_list for item in sublist]

    for index in range(0, len(show_list)):
        factor_number, job_number, operation_number, machine_number, verify_total_fuzzy_pt = read_example(
            rf'{path + example_list[index]}', example_list[index][:3])
        factor_number = 2
        up_configs(factor_number, job_number, operation_number, machine_number)

        fac_agent = DQN(floors_number=configs.floors_number,
                        n_states=2 * configs.factor_number,
                        n_hidden=configs.n_hidden,
                        n_actions=configs.factor_number,
                        learning_rate=configs.learning_rate,
                        gamma=configs.gamma,
                        epsilon=configs.epsilon,
                        target_update=configs.target_update,
                        device=configs.device)

        job_agent = MODQN(floors_number=configs.floors_number,
                          n_states=4 * configs.job_number,
                          n_hidden=configs.n_hidden,
                          n_actions=configs.job_number,
                          learning_rate=configs.learning_rate,
                          gamma=configs.gamma,
                          epsilon=configs.epsilon,
                          target_update=configs.target_update,
                          device=configs.device)

        mac_agent = MODQN(floors_number=configs.floors_number,
                          n_states=5 * 4 * configs.machine_number,
                          n_hidden=configs.n_hidden * 2,
                          n_actions=configs.machine_number * 5,
                          learning_rate=configs.learning_rate,
                          gamma=configs.gamma,
                          epsilon=0.7,
                          target_update=configs.target_update,
                          device=configs.device)

        replay_buff_fac = ReplayBuffer(configs.capacity)
        replay_buff_job = ReplayBuffer(configs.capacity)
        replay_buff_mac = ReplayBuffer(configs.capacity)

        fitness_list = []
        job_loss_list = []
        factory_loss_list = []
        machine_loss_list = []

        nondominating = []
        temp_solution = []

        makespan_list = []
        epochs = 400
        MK_W = 1

        best_list = []
        while MK_W >= 0:
            best = [0x7fffff, 0x7fffff]

            print(MK_W)
            for epoch in tqdm(range(1, epochs)):
                job_loss = []
                factory_loss = []
                machine_loss = []
                total_reward = 0
                total_fuzzy_pt = Generating_example(job_number=configs.job_number,
                                                    operation_number=configs.operation_number,
                                                    machine_number=configs.machine_number)
                shop = Shop(factor_number, job_number, machine_number, operation_number, total_fuzzy_pt)
                reward_list = []
                action_job = 0
                fac_reward = 0

                while not shop.finish:
                    if sum([job.current_op for job in shop.job_list]) == 0:
                        action_job = job_agent.get_action(shop.job_features, shop.legitimate_job)
                        shop.flash_mac(action_job)

                    action_factory = -1
                    reward_ = 0
                    if shop.job_list[action_job].belong_factor == -1:
                        shop.Refresh_makespan(action_job)
                        action_factory = fac_agent.get_action(shop.factory_features, shop.legitimate_factor)
                        shop.job_list[action_job].belong_factor = action_factory

                        fac_state = deepcopy(shop.factory_features)
                        fac_next_state = deepcopy(fac_state)
                        done = 1
                        virtual_actions = []
                        for j in shop.job_list:
                            if j.belong_factor == -1:
                                done = 0
                                virtual_actions.append(j.index)

                        if not done:
                            shop.Refresh_makespan(np.random.choice(virtual_actions))
                            fac_next_state = deepcopy(shop.factory_features)

                        for factor in shop.factor_list:
                            reward_ += value(subtract(factor.all_makespan, shop.factor_list[action_factory].all_makespan))
                        if len(shop.factor_list) > 1:
                            reward_ = reward_ / (len(shop.factor_list) - 1)
                        replay_buff_fac.add(fac_state, action_factory, reward_, fac_next_state, done)
                        fac_reward += reward_

                    action_machine = mac_agent.get_action(shop.mac_features, shop.legitimate_machine)

                    pre_shop = deepcopy(shop)

                    reward = shop.Step(action_job, action_machine)
                    reward_ = reward[0] * MK_W + (reward[1]) * (1 - MK_W)
                    reward_list.append(reward_)
                    total_reward += reward_
                    if replay_buff_fac.size() > 50:
                        s, a, r, ns, d = replay_buff_fac.sample(32)
                        transition_dict = {
                            'states': s,
                            'actions': a,
                            'next_states': ns,
                            'rewards': r,
                            'dones': d,
                        }
                        factory_loss.append(fac_agent.update(transition_dict))

                    replay_buff_job.add(pre_shop.job_features, action_job, reward, shop.job_features, shop.finish)
                    if replay_buff_job.size() > configs.min_capacity:
                        s, a, r, ns, d = replay_buff_job.sample(configs.batch_size)
                        transition_dict = {
                            'states': s,
                            'actions': a,
                            'next_states': ns,
                            'rewards': r,
                            'dones': d,
                        }
                        job_loss.append(job_agent.update(transition_dict))

                    if shop.finish != 1:
                        action_job = job_agent.get_action(shop.job_features, shop.legitimate_job)
                        shop.flash_mac(action_job)

                    replay_buff_mac.add(pre_shop.mac_features, action_machine, reward, shop.mac_features, shop.finish)
                    if replay_buff_mac.size() > configs.min_capacity:
                        s, a, r, ns, d = replay_buff_mac.sample(configs.batch_size)
                        transition_dict = {
                            'states': s,
                            'actions': a,
                            'next_states': ns,
                            'rewards': r,
                            'dones': d,
                        }
                        machine_loss.append(mac_agent.update(transition_dict))
                print(f"example: {show_list[index]} epoch-->{epoch + 1}:", end="")
                for f in shop.factor_list:
                    print(f" f{f.index + 1}: {f.makespan}", end="")
                makespan_list.append([value(get_max([f.makespan for f in shop.factor_list]))])
                job_loss_list.append(sum(job_loss) / (len(job_loss) + 1))
                factory_loss_list.append(sum(factory_loss) / (len(factory_loss) + 1))
                machine_loss_list.append(sum(machine_loss) / (len(machine_loss) + 1))
                print(f" total_reward: {total_reward}")

            path_fac = rf'res/{show_list[index]}_{factor_number}/factor_agent{MK_W}.pt'
            path_job = rf'res/{show_list[index]}_{factor_number}/job_agent{MK_W}.pt'
            path_mac = rf'res/{show_list[index]}_{factor_number}/machine_agent{MK_W}.pt'
            os.makedirs(os.path.dirname(path_fac), exist_ok=True)
            os.makedirs(os.path.dirname(path_job), exist_ok=True)
            os.makedirs(os.path.dirname(path_mac), exist_ok=True)
            torch.save(fac_agent, path_fac)
            torch.save(job_agent, path_job)
            torch.save(mac_agent, path_mac)
            MK_W = round(MK_W - 0.01, 2)
            mac_agent.MK_W = MK_W
            job_agent.MK_W = MK_W
            epochs = 20
            torch.cuda.empty_cache()
