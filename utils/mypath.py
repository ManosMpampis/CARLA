
import os


class MyPath(object):
    @staticmethod
    def db_root_dir(database=''):
        db_names = {'msl', 'smap', 'smd', 'power', 'yahoo', 'kpi', 'swat', 'wadi', 'gecco', 'swan', 'ucr'}
        assert(database in db_names)

        if database == 'msl' or database == 'smap':
            return '/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/datasets/MSL_SMAP'
        elif database == 'ucr':
            return '/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/datasets/UCR'
        elif database == 'yahoo':
            return '/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/datasets/Yahoo'
        elif database == 'smd':
            return '/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/datasets/SMD'
        elif database == 'swat':
            return '/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/datasets/SWAT'
        elif database == 'wadi':
            return '/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/datasets/WADI'
        elif database == 'kpi':
            return '/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/datasets/KPI'
        elif database == 'swan':
            return '/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/datasets/Swan'
        elif database == 'gecco':
            return '/home/manos/Documents/EKETA/HYPER_AI/gits/CARLA/datasets/GECCO'
        
        else:
            raise NotImplementedError
