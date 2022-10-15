import torch
import torch.nn.functional as F

def load_patial_state_dict(model, pretrained_dict):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict) 
    model.load_state_dict(model_dict)

def HP2Deg(pred):
    device = pred.device
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).to(device)
    pred = F.softmax(pred)
    degree = torch.sum(pred*idx_tensor, axis=1) * 3 - 99

    return degree

def rotation_matrix_x(theta):
    theta = theta.view(-1, 1, 1)
    z = torch.zeros_like(theta)
    o = torch.ones_like(theta)
    c = torch.cos(theta)
    s = torch.sin(theta)
    return torch.cat(
        [
            torch.cat([c, z, s], 2),
            torch.cat([z, o, z], 2),
            torch.cat([-s, z, c], 2),
        ],
        1,
    )

def rotation_matrix_y(theta):
    theta = theta.view(-1, 1, 1)
    z = torch.zeros_like(theta)
    o = torch.ones_like(theta)
    c = torch.cos(theta)
    s = torch.sin(theta)
    return torch.cat(
        [
            torch.cat([o, z, z], 2),
            torch.cat([z, c, -s], 2),
            torch.cat([z, s, c], 2),
        ],
        1,
    )

def rotation_matrix_z(theta):
    theta = theta.view(-1, 1, 1)
    z = torch.zeros_like(theta)
    o = torch.ones_like(theta)
    c = torch.cos(theta)
    s = torch.sin(theta)
    return torch.cat(
        [
            torch.cat([c, -s, z], 2),
            torch.cat([s, c, z], 2),
            torch.cat([z, z, o], 2),
        ],
        1,
    )

def GetRotMat(yaw, pitch, roll, t=None, input_mode=None):
    if input_mode is None:
        yaw = HP2Deg(yaw)
        pitch = HP2Deg(pitch)
        roll = HP2Deg(roll)
        
    yaw = torch.deg2rad(yaw).unsqueeze(1)
    pitch = torch.deg2rad(pitch).unsqueeze(1)
    roll = torch.deg2rad(roll).unsqueeze(1)

    if t is not None:
        m_dim = 4
    else:
        m_dim = 3
    rot_mat = torch.eye(m_dim).unsqueeze(0).repeat(yaw.size(0), 1, 1).to(yaw.device)
    R = rotation_matrix_x(roll) @ rotation_matrix_y(pitch) @ rotation_matrix_z(yaw)
    rot_mat[:, :3, :3] = R
    if t is not None:
        t = torch.FloatTensor(t).to(yaw.device)
        t = t.unsqueeze(0).unsqueeze(-1).repeat(yaw.size(0), 1, 1)
        t = R@t
        rot_mat[:, :3, -1] = t.squeeze(-1)

    return rot_mat