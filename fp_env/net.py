from typing import List, Dict, Tuple, Any, Union
from .block import Block
from .terminal import Terminal
import math

class Net:
    def __init__(self, connector_list:List[Union[Block, Terminal]], weight:float=1.0, read_fp:bool=False):
        """
        connector_list: list of connector id's that are connected to the net.
        For instance, the full id of preplaced block, movable block, and terminal.
        """
        self._weight = weight
        self.read_fp = read_fp

        # add net to connected connectors
        for connector in connector_list:
            connector.connected_nets.append(self)


        # number of preplaced block and terminal
        self.num_preplaced_fixed_connector = 0
        for connector in connector_list:
            if isinstance(connector, Block) and connector.preplaced:
                self.num_preplaced_fixed_connector += 1
            elif isinstance(connector, Terminal):
                self.num_preplaced_fixed_connector += 1
        

        # set init range for preplaced or fixed connectors
        self.init_x_min, self.init_x_max, self.init_y_min, self.init_y_max = math.inf, -math.inf, math.inf, -math.inf
        if self.num_preplaced_fixed_connector > 0:
            for connector in connector_list:
                if isinstance(connector, Block) and connector.preplaced:
                    # a block, use the center of the block
                    self.init_x_min = min(self.init_x_min, connector.grid_x + connector.grid_w / 2)
                    self.init_x_max = max(self.init_x_max, connector.grid_x + connector.grid_w / 2)
                    self.init_y_min = min(self.init_y_min, connector.grid_y + connector.grid_h / 2)
                    self.init_y_max = max(self.init_y_max, connector.grid_y + connector.grid_h / 2)
                elif isinstance(connector, Terminal):
                    # a terminal, use the terminal position
                    self.init_x_min = min(self.init_x_min, connector.grid_x)
                    self.init_x_max = max(self.init_x_max, connector.grid_x)
                    self.init_y_min = min(self.init_y_min, connector.grid_y)
                    self.init_y_max = max(self.init_y_max, connector.grid_y)
                
            self.init_x_min, self.init_x_max, self.init_y_min, self.init_y_max = round(self.init_x_min), round(self.init_x_max), round(self.init_y_min), round(self.init_y_max)
        
        if self.read_fp:
            self.connector_list = connector_list
            for connector in connector_list:
                if isinstance(connector, Block) and (not connector.preplaced):
                    self.init_x_min = min(self.init_x_min, connector.grid_x + connector.grid_w / 2)
                    self.init_x_max = max(self.init_x_max, connector.grid_x + connector.grid_w / 2)
                    self.init_y_min = min(self.init_y_min, connector.grid_y + connector.grid_h / 2)
                    self.init_y_max = max(self.init_y_max, connector.grid_y + connector.grid_h / 2)
            self.init_x_min, self.init_x_max, self.init_y_min, self.init_y_max = round(self.init_x_min), round(self.init_x_max), round(self.init_y_min), round(self.init_y_max)
        
        # set init net range
        self.reset()
    
    def get_net_weight(self) -> float:
        return self._weight

    def init_layer_num_pin(self, num_layer)->List[int]:
        self.pin_layer = [0 for layer_id in range(num_layer)]
    
    def add_layer_num_pin(self, layer_id):
        self.pin_layer[layer_id] += 1
    
    def show_layer_num_pin(self):
        for layer_id in range(len(self.pin_layer)):
            print("layer: ", layer_id, "pin number: ", self.pin_layer[layer_id])
    
    # init_flag
    def is_init_status(self) -> bool:
        init_flag = True
        for layer_id in range(len(self.pin_layer)):
            if not self.pin_layer[layer_id] == 0:
                init_flag = False
        return init_flag
    
    def fill_layer_num_pin_withfp(self):
        for connector in self.connector_list:
            if isinstance(connector, Block) and (not connector.preplaced):
                self.add_layer_num_pin(connector.z)
    
    def is_cut(self):
        num_layer_with_pins = 0
        for pin_layer_per in self.pin_layer:
            if pin_layer_per > 0:
                num_layer_with_pins += 1
            if num_layer_with_pins > 1:
                return True

        if self.num_preplaced_fixed_connector > 0 and (num_layer_with_pins == 1 and self.pin_layer[0] == 0):
            return True
        else:
            return False

    def add_connector(self, connector:Union[Block, Terminal]):
        """Add a connector to the net."""
        assert hasattr(connector, "idx"), "connector {} should have idx attribute.".format(connector)
        # connector
        if not hasattr(self, "connector_list"):
            self.connector_list = []
        
        # add connector
        if isinstance(connector, Block):
            self.connector_list.append({"type":Block, "id":connector.idx})
        elif isinstance(connector, Terminal):
            self.connector_list.append({"type":Terminal, "id":connector.idx})
        else:
            raise ValueError("connector should be Block or Terminal, but got {}".format(type(connector)))

    def reset(self):
        """If there is no preplaced block or terminal, reset the net range to inf, -inf."""
        if (self.num_preplaced_fixed_connector > 0 or self.read_fp):
            self.x_min, self.x_max, self.y_min, self.y_max = self.init_x_min, self.init_x_max, self.init_y_min, self.init_y_max
            self.num_placed_connector = self.num_preplaced_fixed_connector
        else:
            self.x_min, self.x_max, self.y_min, self.y_max = math.inf, -math.inf, math.inf, -math.inf
            self.num_placed_connector = 0
    

    def update(self, block:Block):
        """After placing a block, update the net range."""
        assert not block.preplaced, "The block is preplaced, should not be updated."
        self.num_placed_connector += 1
        x_center = block.grid_x + block.grid_w / 2
        y_center = block.grid_y + block.grid_h / 2
        self.x_min = round(min(self.x_min, x_center))
        self.x_max = round(max(self.x_max, x_center))
        self.y_min = round(min(self.y_min, y_center))
        self.y_max = round(max(self.y_max, y_center))
        
    
    def __repr__(self) -> str:
        return "Net(x_min={}, x_max={}, y_min={}, y_max={}, num_placed_connector={})".format(self.x_min, self.x_max, self.y_min, self.y_max, self.num_placed_connector)


    def calc_hpwl(self) -> int:
        return max(self.x_max - self.x_min, 0) + max(self.y_max - self.y_min, 0)

    def calc_stride(self) -> Tuple[int, int]:
        """return x_stride and y_stride."""
        return max(self.x_max - self.x_min, 0), max(self.y_max - self.y_min, 0)