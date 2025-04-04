import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class STNKd(nn.Module):
    # T-Net a.k.a. Spatial Transformer Network
    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.conv1 = nn.Sequential(nn.Conv1d(k, 64, 1), nn.BatchNorm1d(64))
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024))

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, k * k),
        )

    def forward(self, x):
        """
        Input: [B,k,N]
        Output: [B,k,k]
        """
        B = x.shape[0]
        device = x.device
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2)[0]

        x = self.fc(x)
        
        # Followed the original implementation to initialize a matrix as I.
        identity = (
            Variable(torch.eye(self.k, dtype=torch.float))
            .reshape(1, self.k * self.k)
            .expand(B, -1)
            .to(device)
        )
        x = x + identity
        x = x.reshape(-1, self.k, self.k)
        return x


class PointNetFeat(nn.Module):
    """
    Corresponds to the part that extracts max-pooled features.
    """
    def __init__(
        self,
        input_transform: bool = False,
        feature_transform: bool = False,
    ):
        super().__init__()
        self.input_transform = input_transform
        self.feature_transform = feature_transform

        if self.input_transform:
            self.stn3 = STNKd(k=3)
        if self.feature_transform:
            self.stn64 = STNKd(k=64)

        # point-wise mlp
        # TODO : Implement point-wise mlp model based on PointNet Architecture.
        # Point-wise MLP layers (Conv1d)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud: [B,N,3]
        Output:
            - Global feature: [B,1024]
            - ...
        """

        # TODO : Implement forward function.
        # Transpose to [B, 3, N] for Conv1d
        x = pointcloud.transpose(2, 1)  # [B, 3, N]

        # Input transform
        trans_input = None
        if self.input_transform:
            trans_input = self.stn3(x)  # [B, 3, 3]
            x = torch.bmm(trans_input, x)  # [B, 3, N]

        # First MLP: 3 -> 64
        x = F.relu(self.bn1(self.conv1(x)))  # [B, 64, N]

        # Feature transform
        trans_feat = None
        if self.feature_transform:
            trans_feat = self.stn64(x)  # [B, 64, 64]
            x = torch.bmm(trans_feat, x)  # [B, 64, N]

        # Second and third MLP: 64 -> 128 -> 1024
        x = F.relu(self.bn2(self.conv2(x)))  # [B, 128, N]
        x = F.relu(self.bn3(self.conv3(x)))  # [B, 1024, N]

        # Max pool over N points
        x = torch.max(x, 2)[0]  # [B, 1024]

        return x, trans_input, trans_feat




class PointNetCls(nn.Module):
    def __init__(self, num_classes, input_transform, feature_transform):
        super().__init__()
        self.num_classes = num_classes
        
        # extracts max-pooled features
        self.pointnet_feat = PointNetFeat(input_transform, feature_transform)
        
        # returns the final logits from the max-pooled features.
        # TODO : Implement MLP that takes global feature as an input and return logits.
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, self.num_classes)


    def forward(self, pointcloud):
        """
        Input:
            - pointcloud [B,N,3]
        Output:
            - logits [B,num_classes]
            - ...
        """
        # TODO : Implement forward function.

        global_feat, trans_input, trans_feat = self.pointnet_feat(pointcloud)
        x = F.relu(self.bn1(self.fc1(global_feat)))
        x = F.relu(self.bn2(self.fc2(x)))
        logits = self.fc3(x)
        return logits, trans_input, trans_feat



class PointNetPartSeg(nn.Module):
    def __init__(self, m=50):
        super().__init__()

        # returns the logits for m part labels each point (m = # of parts = 50).
        # TODO: Implement part segmentation model based on PointNet Architecture.
        
        self.m = m  # <--- Define self.m here
 
        # 1) T-Nets for input and feature transforms
        self.input_transform = STNKd(k=3)
        self.feature_transform = STNKd(k=64)

        # 2) Point-wise MLP layers for local/global feature extraction
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)

        # 3) Segmentation head
        #    We'll concatenate local_feat(64) + global_feat(1024) = 1088
        self.conv4 = nn.Conv1d(1088, 512, 1)
        self.bn4 = nn.BatchNorm1d(512)

        self.conv5 = nn.Conv1d(512, 256, 1)
        self.bn5 = nn.BatchNorm1d(256)

        # final layer produces m part scores
        self.conv6 = nn.Conv1d(256, self.m, 1)

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud: [B,N,3]
        Output:
            - logits: [B,50,N] | 50: # of point labels
            - ...
        """
        # TODO: Implement forward function.

        B, N, _ = pointcloud.shape

        # (1) Input Transform
        x = pointcloud.transpose(2, 1)  # [B, 3, N]
        trans_input = self.input_transform(x)  # [B, 3, 3]
        x = torch.bmm(trans_input, x)  # [B, 3, N]

        # (2) First MLP: 3 -> 64
        x = F.relu(self.bn1(self.conv1(x)))  # [B, 64, N]

        # (3) Feature Transform
        trans_feat = self.feature_transform(x)  # [B, 64, 64]
        local_feat = torch.bmm(trans_feat, x)   # [B, 64, N]

        # (4) Second MLP: 64 -> 128 -> 1024
        x = F.relu(self.bn2(self.conv2(local_feat)))  # [B, 128, N]
        x = F.relu(self.bn3(self.conv3(x)))           # [B, 1024, N]

        # (5) Global Feature via max pool
        global_feat = torch.max(x, 2, keepdim=True)[0]  # [B, 1024, 1]
        global_feat = global_feat.repeat(1, 1, N)       # [B, 1024, N]

        # (6) Concatenate local (64) and global (1024) => 1088 channels
        seg_feat = torch.cat([local_feat, global_feat], dim=1)  # [B, 1088, N]

        # (7) Segmentation head
        x = F.relu(self.bn4(self.conv4(seg_feat)))  # [B, 512, N]
        x = F.relu(self.bn5(self.conv5(x)))         # [B, 256, N]
        logits = self.conv6(x)                      # [B, m, N]

        return logits, trans_input, trans_feat




class PointNetAutoEncoder(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        self.pointnet_feat = PointNetFeat()

        # Decoder is just a simple MLP that outputs N x 3 (x,y,z) coordinates.
        # TODO : Implement decoder.


        # ----------------------------
        # Encoder (no T-Nets)
        # ----------------------------
        # We'll implement a simplified PointNet encoder:
        #  - 3 -> 64 -> 128 -> 1024 (1D convs + BN + ReLU), then max-pool
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)

        # ----------------------------
        # Decoder
        # ----------------------------
        # Takes the 1024-d global feature and outputs N*3 values, then reshapes to [B, N, 3].
        self.fc1 = nn.Linear(1024, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, self.num_points * 3)

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud [B,N,3]
        Output:
            - pointcloud [B,N,3]
            - ...
        """
        # TODO : Implement forward function.
        
        # 1) Encoder
        #    - shape [B, 3, N] => conv => shape [B, 1024, N] => max-pool => shape [B, 1024]
        x = pointcloud.transpose(2, 1)  # [B, 3, N]
        x = F.relu(self.bn1(self.conv1(x)))     # [B, 64, N]
        x = F.relu(self.bn2(self.conv2(x)))     # [B, 128, N]
        x = F.relu(self.bn3(self.conv3(x)))     # [B, 1024, N]
        global_feat = torch.max(x, dim=2)[0]    # [B, 1024]

        # 2) Decoder
        #    - shape [B, 1024] => [B, N*3] => reshape => [B, N, 3]
        x = F.relu(self.bn_fc1(self.fc1(global_feat)))  # [B, 512]
        x = F.relu(self.bn_fc2(self.fc2(x)))            # [B, 256]
        x = self.fc3(x)                                 # [B, N*3]
        x = x.view(-1, self.num_points, 3)              # [B, N, 3]

        return x



def get_orthogonal_loss(feat_trans, reg_weight=1e-3):
    """
    a regularization loss that enforces a transformation matrix to be a rotation matrix.
    Property of rotation matrix A: A*A^T = I
    """
    if feat_trans is None:
        return 0

    B, K = feat_trans.shape[:2]
    device = feat_trans.device

    identity = torch.eye(K).to(device)[None].expand(B, -1, -1)
    mat_square = torch.bmm(feat_trans, feat_trans.transpose(1, 2))

    mat_diff = (identity - mat_square).reshape(B, -1)

    return reg_weight * mat_diff.norm(dim=1).mean()
