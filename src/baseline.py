# Copyright (C) 2018-2021  Ben Cardoen
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU Affero General Public License as published
#     by the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU Affero General Public License for more details.
#
#     You should have received a copy of the GNU Affero General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.
import torch.nn as nn
import torch.nn.functional as F


class SVRGBaseline(nn.Module):
    def __init__(self):
        super(SVRGBaseline, self).__init__()
        self.fc1 = nn.Linear(PIX**2, (PIX-1)**2)
        self.fc2 = nn.Linear((PIX-1)**2, (PIX-1)**2)
        self.fc3 = nn.Linear((PIX-1)**2, (PIX-2)**2)
        self.fc4 = nn.Linear((PIX-2)**2, (PIX-3)**2)
        self.fc5 = nn.Linear((PIX-3)**2, (PIX-4)**2)
        self.fc6 = nn.Linear((PIX-4)**2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x
