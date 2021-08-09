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
import logging
import os
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)
lgr = logging.getLogger('global')
lgr.setLevel(logging.INFO)

lgr = None


def initlogger(configuration):
    global lgr
    if lgr is None:
        lgr = logging.getLogger('global')
    if 'logdir' in configuration:
        fh = logging.FileHandler(os.path.join(configuration['logdir'], 'svrg.log'))
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s")
        fh.setFormatter(formatter)
        lgr.addHandler(fh)
    lgr.setLevel(logging.INFO)
    return lgr


def getlogger():
    global lgr
    if lgr is None:
        return initlogger({})
    return lgr
