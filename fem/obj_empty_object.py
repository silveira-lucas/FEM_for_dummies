import copy
#%%

class EmptyObject():
    def copy(self):
        return copy.deepcopy(self)

