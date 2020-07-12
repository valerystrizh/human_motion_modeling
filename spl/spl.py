import numpy as np
import torch
import torch.nn as nn

SMPL_SKELETON =  {
            0: (-1, 'root'),
            3: [0, 'spine1'],
            6: [3, 'spine2'],
            9: [6, 'spine3'],
            12: [9, 'neck'],
            15: [12, 'head'],
            2: [0, 'l_hip'],
            5: [2, 'l_knee'],
            8: [5, 'l_heel'],
            11: [8, 'l_foot'],
            1: [0, 'r_hip'],
            4: [1, 'r_knee'],
            7: [4, 'r_heel'],
            10: [7, 'r_foot'],
            14: [9, 'l_collar'],
            17: [14, 'l_shoulder'],
            19: [17, 'l_elbow'],
            21: [19, 'l_wrist'],
            13: [9, 'r_collar'],
            16: [13, 'r_shoulder'],
            18: [16, 'r_elbow'],
            20: [18, 'r_wrist']
        }
            
class SPL(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_size_spl, device):
        super().__init__()
        def mlp(hidden_size):
            return nn.Sequential(
                nn.Linear(hidden_size, hidden_size_spl),
                nn.ReLU(),
                nn.Linear(hidden_size_spl, output_size)
            ).to(device)

        self.prediction_order = np.arange(22)
        
        self.kinematic_tree = dict()
        for joint_id in sorted(SMPL_SKELETON.keys()):
            joint_entry = SMPL_SKELETON[joint_id]
            parent_list_ = [joint_entry[0]] if joint_entry[0] > -1 else []
            self.kinematic_tree[joint_id] = [parent_list_, joint_id, joint_entry[1]]
            
        self.indexed_skeleton = dict()

        def get_all_parents(parent_list, parent_id, tree):
            if parent_id not in parent_list:
                parent_list.append(parent_id)
                for parent in tree[parent_id][0]:
                    get_all_parents(parent_list, parent, tree)
            
        for joint_id in self.prediction_order:
            joint_entry = self.kinematic_tree[joint_id]
            parent_list_ = list()
            if len(joint_entry[0]) > 0:
                get_all_parents(parent_list_, joint_entry[0][0], self.kinematic_tree)
            new_entry = [parent_list_, joint_entry[1], joint_entry[2]]
            self.indexed_skeleton[joint_id] = new_entry
        
        self.spl_layers = {}

        for joint_id in self.indexed_skeleton:
            self.spl_layers[joint_id] = mlp(hidden_size+output_size*len(self.indexed_skeleton[joint_id][0]))
    
    def forward(self, input_data):
        joint_predictions = dict()

        for joint_id in self.prediction_order:
            parent_joint_ids, joint_id, joint_name = self.indexed_skeleton[joint_id]
            layer = self.spl_layers[joint_id]
            inputs = [input_data]

            for parent_joint_id in parent_joint_ids:
                inputs.append(joint_predictions[parent_joint_id])

            joint_predictions[joint_id] = layer(torch.cat(inputs, 2))

        joint_predictions_values_by_sorted_keys = []
        for key in sorted(joint_predictions):
            joint_predictions_values_by_sorted_keys.append(joint_predictions[key])
        
        return torch.cat(joint_predictions_values_by_sorted_keys, 2).permute(0, 2, 1)
    
    