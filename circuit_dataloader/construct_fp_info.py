from .parser import parse_blk_tml, map_tml, parse_net, parse_blk_xyz
from .construct_partner import construct_partner_blk
from .construct_layer import assign_layer
from .construct_pre_placed_module import construct_preplaced_modules
import torch
import fp_env
import pandas as pd
from typing import Tuple
from typing import List
from copy import deepcopy
from collections import defaultdict


def construct_fp_info_func(circuit:str, area_util:float, num_grid_x:int, num_grid_y:int, num_alignment:int, 
                           alignment_rate:float, alignment_sort:str, num_preplaced_module:int, add_virtual_block:bool, num_layer:int, read_fp: bool, set_z_only: bool, add_halo: bool, halo_width: float, halo_height: float, ratio_range : List[float] ) -> Tuple[fp_env.FPInfo, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    alignment_rate = 1.0 if alignment_rate is None else alignment_rate
    
    # read block and terminal (w,h,virtual)
    # (add_halo, halo_width, halo_height)
    if not add_halo:
        halo_width = 0.0    
        halo_height = 0.0
    
    # parse_blk_tml (blk_wh_dict, tml_xy_dict)
    blk_wh_dict, blk_AR_dict, tml_xy_dict, outline_width, outline_height = parse_blk_tml(circuit, area_util, add_halo, halo_width, halo_height)
    
    tml_xy_dict = map_tml(tml_xy_dict, outline_width, outline_height)

    # assign layer (w,h,virtual,z)
    first_blk_wh = next(iter(blk_wh_dict.values()))

    # initialize the levels (first_blk_wh)
    if "z" not in first_blk_wh.keys():
        # balance the area between the two layers
        print("[INFO] Constructing layer information")
        blk_wh_dict = assign_layer(blk_wh_dict, num_layer)
    
    # construct preplaced modules (w,h,virtual,z,preplaced,x,y)
    first_blk_wh = next(iter(blk_wh_dict.values()))

    if "preplaced" not in first_blk_wh.keys():
        print("[INFO] Constructing preplaced modules")
        blk_wh_dict = construct_preplaced_modules(num_preplaced_module, blk_wh_dict, outline_height)

    # add virtual block
    # not add the halo_width and halo_height
    # virtual_block
    if add_virtual_block:
        print("[INFO] Adding virtual block")
        blk_wh_dict["virtual_block"] = {
            "w": 1,
            "h": 1,
            "realw": 1,
            "realh": 1,
            "type": "virtual",
            "virtual": True,
            "z": 0,
            "preplaced": False,
            "x": 0,
            "y": 0,
        }

    # read and get the floorplan results from the fp.txt
    blk_xyz_dict = {}
    if read_fp:
        blk_xyz_dict = parse_blk_xyz(circuit)

    # all blocks, preplaced_blocks + movable_blocks
    preplaced_blocks = []
    movable_blocks = []
    
    # The number of the blocks on two layers
    layer_dict = {0: 0, 1: 0}

    for blk_name, blk_info in blk_wh_dict.items():
        if blk_info['preplaced']: # PPM
            preplaced_blocks.append(fp_env.Block(blk_info['x'], blk_info['y'], blk_info['z'], blk_info['w'], blk_info['h'], blk_info["realw"], blk_info["realh"], blk_name, blk_info["type"], True, blk_info['virtual']))
        else: # movable or virtual
            movable_blocks.append(fp_env.Block(0, 0, blk_info['z'], blk_info['w'], blk_info['h'], blk_info["realw"], blk_info["realh"], blk_name, blk_info["type"], False, blk_info['virtual']))
            # blk_xyz_dict
            if blk_name in list(blk_xyz_dict.keys()):
                xyz_list = blk_xyz_dict[blk_name]
                x_, y_, z_ = xyz_list[0], xyz_list[1], xyz_list[2]
                # print(x_, y_, z_)
                if set_z_only:
                    movable_blocks[-1].set_z(int(z_))
                    layer_dict[int(z_)] += 1
                else:
                    movable_blocks[-1].set_xyz(x_, y_, int(z_))
                    movable_blocks[-1].placed = True
            # blk_AR_dict
            if blk_name in list(blk_AR_dict.keys()):
               min_AR = blk_AR_dict[blk_name]["min_AR"]
               max_AR = blk_AR_dict[blk_name]["max_AR"]
            else:
               min_AR = ratio_range[0]
               max_AR = ratio_range[1]
            
            movable_blocks[-1].set_max_min_ratio(min_AR, max_AR)

    
    # The number of blocks on each layer
    print("layer 0: ", layer_dict[0], "layer 1: ", layer_dict[1])
    
    # （preplaced_blocks，movable_blocks）
    block_info = preplaced_blocks + movable_blocks

    # blocks
    # print("preplaced_blocks: ", len(preplaced_blocks))
    # print("movable_blocks", len(movable_blocks))
    # print("blocks: ", len(block_info))

    # terminal_info
    terminal_info = []
    for terminal_name in tml_xy_dict.keys():
        # terminal_name
        terminal_xy = tml_xy_dict[terminal_name]
        # All the z-position are 0
        terminal_info.append(fp_env.Terminal(terminal_xy['x'], terminal_xy['y'], 0, terminal_name))


    # discretize block and terminal
    block_info, terminal_info, grid_width, grid_height = fp_env.discretize(block_info, terminal_info, num_grid_x, num_grid_y, outline_width, outline_height)
    
    print("block info: ", len(block_info))
    print("tml_info: ", len(terminal_info))

    # print(grid_width, grid_height)

    # read nets and construct net_info, nets is List[List[str]]
    # also construct adjacency matrix
    nets_str = parse_net(circuit)
    # name to objects
    name2obj = {obj.name: obj for obj in block_info + terminal_info}

    net_info = []
    net_weight = 1.0

    for net_connectors_str in nets_str:
        connector_list = [name2obj[connector_name] for connector_name in net_connectors_str]
        net = fp_env.Net(connector_list, net_weight, read_fp and (not set_z_only))
        # initialize the number of the pins in each layer for different nets
        net.init_layer_num_pin(num_layer)
        # read_from_floorplan
        if read_fp and not set_z_only:
            net.fill_layer_num_pin_withfp()
        net_info.append(net)

    # load to fp_info
    fp_info = fp_env.FPInfo(block_info, terminal_info, net_info, outline_width, outline_height, num_grid_x, num_grid_y)
    episode_len = fp_info.movable_block_num

    # construct adjacency matrix
    adjacency_matrix = torch.zeros(fp_info.block_num + fp_info.termimal_num, fp_info.block_num + fp_info.termimal_num)
    for net_idx, net_connectors_str in enumerate(nets_str):
        for i in range(len(net_connectors_str)):
            for j in range(i+1, len(net_connectors_str)):
                adjacency_matrix[name2obj[net_connectors_str[i]].idx, name2obj[net_connectors_str[j]].idx] = fp_info.net_info[net_idx].get_net_weight()
                adjacency_matrix[name2obj[net_connectors_str[j]].idx, name2obj[net_connectors_str[i]].idx] = fp_info.net_info[net_idx].get_net_weight()
            # add connector to net
            fp_info.net_info[net_idx].add_connector(name2obj[net_connectors_str[i]])

    #print(adjacency_matrix)
    fp_info.set_adjacency_matrix(adjacency_matrix)

    # construct alignment partner, and set to fp_info
    fp_info.set_alignment_sort(alignment_sort)
    # construct_partner_blk

    df_partner = construct_partner_blk(fp_info, num_alignment, alignment_sort)
    #print(df_partner)
    # print("df_partner: ", df_partner)

    for row in df_partner.itertuples():
        blk0_name, blk1_name = row.blk0, row.blk1
        blk0, blk1 = name2obj[blk0_name], name2obj[blk1_name]
        blk0_idx, blk1_idx = blk0.idx, blk1.idx
        alignment_area = min(blk0.area, blk1.area) * alignment_rate
        # set the partner
        fp_info.set_partner(blk0_idx, blk1_idx, alignment_area)
    
    # fp_info
    print("fp_info: ", fp_info.name2alignment_group.items())

    # name2alignment_group
    for blk_name, aln_group in fp_info.name2alignment_group.items():
        #print(fp_info.name2alignment_group)
        df_partner.loc[df_partner['blk0'] == blk_name, 'alignment_group'] = aln_group
    
    # # df_partner
    # print("df_partner: ", df_partner)

    # print(df_partner['blk0'])
    # print(df_partner['blk1'])
    # print(alignment_area)

    # alignment group, blk0, blk1
    if len(fp_info.name2alignment_group.items()) > 0:
        df_partner.sort_values(by=['alignment_group', 'blk0', 'blk1'], inplace=True)
        df_partner["alignment_group"] = df_partner["alignment_group"].astype(int)
    else:
        df_partner.sort_values(by=['blk0', 'blk1'], inplace=True)

    return fp_info, df_partner