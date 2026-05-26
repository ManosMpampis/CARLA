
import os


class MyPath(object):
    @staticmethod
    def db_root_dir(database=''):
        db_names = {'msl', 'smap', 'smd', 'new_smd', 'psm', 'power', 'yahoo', 'kpi', 'swat', 'wadi', 'gecco', 'swan', 'ucr'}
        assert(database in db_names)

        if database == 'msl' or database == 'smap':
            return 'datasets/MSL_SMAP'
        elif database == 'ucr':
            return 'datasets/UCR'
        elif database == 'yahoo':
            return 'datasets/Yahoo'
        elif database == 'smd':
            return 'datasets/SMD'
        elif database == 'new_smd':
            return 'datasets/new_smd'
        elif database == 'psm':
            return 'datasets/PSM'
        elif database == 'swat':
            return 'datasets/SWAT'
        elif database == 'wadi':
            return 'datasets/WADI'
        elif database == 'kpi':
            return 'datasets/KPI'
        elif database == 'swan':
            return 'datasets/Swan'
        elif database == 'gecco':
            return 'datasets/GECCO'
        
        else:
            raise NotImplementedError
