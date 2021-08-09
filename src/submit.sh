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
#!/bin/bash
#SBATCH --account=<YOURACCOUNT>
#SBATCH --mem-per-cpu=64G      # increase as needed
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-user=<YOUREMAIL>
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

#### Replace any variable in <> with your details

cd <SOURCE>
module purge

echo "Configuring ENV"
module load cuda
module load python/3.7
echo "Copying DATA"
mkdir /dev/shm/ertrain && tar -xf <WHEREIS>/training.tgz -C /dev/shm/ertrain


source <VENVLOCATION>/bin/activate
export CUDA_VISIBLE_DEVICES=0


python trainlstm.py --imageroot=/dev/shm/ertrain --loss=intmse --outputroot=<YOURDIR>/output --shuffle=False --mode=lstm --epochs=500 --csv=<GTPATH>/density.csv --mode=validate --batchweights=True

# Cleanup
echo "Removing memdisk"
rm -r /dev/shm/svrgtrain
