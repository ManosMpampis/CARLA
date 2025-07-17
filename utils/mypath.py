import os


class MyPath(object):
    @staticmethod
    def db_root_dir(database=''):
        db_names = {'msl', 'smap', 'smd', 'power', 'yahoo', 'kpi', 'swat', 'wadi', 'gecco', 'swan', 'ucr'}
        assert(database in db_names)

        project_dir = os.path.join(os.path.dirname(__file__), '..')
        if database == 'msl' or database == 'smap':
            return f'{project_dir}/datasets/MSL_SMAP'
        elif database == 'ucr':
            return f'{project_dir}/datasets/UCR'
        elif database == 'yahoo':
            return f'{project_dir}/datasets/Yahoo'
        elif database == 'smd':
            return f'{project_dir}/datasets/SMD'
        elif database == 'swat':
            return f'{project_dir}/datasets/SWAT'
        elif database == 'wadi':
            return f'{project_dir}/datasets/WADI'
        elif database == 'kpi':
            return f'{project_dir}/datasets/KPI'
        elif database == 'swan':
            return f'{project_dir}/datasets/Swan'
        elif database == 'gecco':
            return f'{project_dir}/datasets/GECCO'
        
        else:
            raise NotImplementedError
