import numpy as np
import pickle
import os

class EncoderResHandler: 
    def __init__(self, data_dir, info_dir, file_prefix="", data=None, info=None):
        self.data_dir = data_dir
        self.info_dir = info_dir
        self.file_prefix = file_prefix
        self.data = data
        self.info = info

    def save(self): 
        try:
            np.save(os.path.join(self.data_dir, self.file_prefix + ".npy"), self.data)
        except Exception as e:
            print(f"Error saving data: {str(e)}")

        try:
            with open(os.path.join(self.info_dir, self.file_prefix + ".info"), 'wb') as file:
                pickle.dump(self.info, file)
        except Exception as e:
            print(f"Error saving info: {str(e)}")

    def read(self): 
        try:
            self.data = np.load(os.path.join(self.data_dir, self.file_prefix + ".npy"))
        except Exception as e:
            print(f"Error loading data: {str(e)}")

        try:
            with open(os.path.join(self.info_dir, self.file_prefix + ".info"), 'rb') as file:
                self.info = pickle.load(file)
        except Exception as e:
            print(f"Error loading info: {str(e)}")


class AnnoEncoderResHandler: 
    def __init__(self, whole_res_dir, file_prefix="", res=None, tok=None, name=None):
        # whole_res_dir: dir to place whole res together, not for the purpose of individual plotting, but for general inspection. 
        self.whole_res_dir = whole_res_dir
        self.file_prefix = file_prefix
        self.res = res    # res, numpy
        self.tok = tok    # token, list
        self.name = name  # name, list, name of each corresponding point

    def save(self): 
        try:
            np.save(os.path.join(self.whole_res_dir, self.file_prefix + ".res"), 
                    self.res)
        except Exception as e:
            print(f"Error saving results: {str(e)}")

        try:
            with open(os.path.join(self.whole_res_dir, self.file_prefix + ".tok"), 'wb') as file:
                pickle.dump(self.tok, file)
        except Exception as e:
            print(f"Error saving tokens: {str(e)}")

        try:
            with open(os.path.join(self.whole_res_dir, self.file_prefix + ".name"), 'wb') as file:
                pickle.dump(self.name, file)
        except Exception as e:
            print(f"Error saving names: {str(e)}")

    def read(self): 
        try:
            self.res = np.load(os.path.join(self.whole_res_dir, self.file_prefix + ".res.npy"))
        except Exception as e:
            print(f"Error loading results: {str(e)}")

        try:
            with open(os.path.join(self.whole_res_dir, self.file_prefix + ".tok"), 'rb') as file:
                self.tok = pickle.load(file)
        except Exception as e:
            print(f"Error loading tokens: {str(e)}")

        try:
            with open(os.path.join(self.whole_res_dir, self.file_prefix + ".name"), 'rb') as file:
                self.name = pickle.load(file)
        except Exception as e:
            print(f"Error loading names: {str(e)}")

class AnnoEncoderMelResHandler: 
    # stores anno, encoder output as well as mel transformed input
    def __init__(self, whole_res_dir, file_prefix="", res=None, tok=None, transformed=None):
        # whole_res_dir: dir to place whole res together, not for the purpose of individual plotting, but for general inspection. 
        self.whole_res_dir = whole_res_dir
        self.file_prefix = file_prefix
        self.res = res    # res, list (of numpy arrays)
        self.tok = tok    # token, list
        self.transformed = transformed  # transformed, list

    def save(self): 
        try:
            with open(os.path.join(self.whole_res_dir, self.file_prefix + ".res"), 'wb') as file:
                pickle.dump(self.res, file)
        except Exception as e:
            print(f"Error saving results: {str(e)}")

        try:
            with open(os.path.join(self.whole_res_dir, self.file_prefix + ".input"), 'wb') as file:
                pickle.dump(self.transformed, file)
        except Exception as e:
            print(f"Error saving inputs: {str(e)}")

        try:
            with open(os.path.join(self.whole_res_dir, self.file_prefix + ".tok"), 'wb') as file:
                pickle.dump(self.tok, file)
        except Exception as e:
            print(f"Error saving tokens: {str(e)}")

    def read(self): 
        try:
            with open(os.path.join(self.whole_res_dir, self.file_prefix + ".res"), 'rb') as file:
                self.res = pickle.load(file)
        except Exception as e:
            print(f"Error loading results: {str(e)}")

        try:
            with open(os.path.join(self.whole_res_dir, self.file_prefix + ".input"), 'rb') as file:
                self.transformed = pickle.load(file)
        except Exception as e:
            print(f"Error loading inputs: {str(e)}")

        try:
            with open(os.path.join(self.whole_res_dir, self.file_prefix + ".tok"), 'rb') as file:
                self.tok = pickle.load(file)
        except Exception as e:
            print(f"Error loading tokens: {str(e)}")


class BndEncoderResHandler: 
    def __init__(self, whole_res_dir, file_prefix="", res=None, tok=None, name=None, match_status=None):
        # whole_res_dir: dir to place whole res together, not for the purpose of individual plotting, but for general inspection. 
        self.whole_res_dir = whole_res_dir
        self.file_prefix = file_prefix
        self.res = res    # res, list
        self.tok = tok    # token, list
        self.name = name  # name, list, name of each corresponding point
        # self.match_status = match_status

    def save(self): 
        try:
            with open(os.path.join(self.whole_res_dir, self.file_prefix + ".res"), 'wb') as file:
                pickle.dump(self.res, file)
        except Exception as e:
            print(f"Error saving tokens: {str(e)}")

        try:
            with open(os.path.join(self.whole_res_dir, self.file_prefix + ".tok"), 'wb') as file:
                pickle.dump(self.tok, file)
        except Exception as e:
            print(f"Error saving tokens: {str(e)}")

        try:
            with open(os.path.join(self.whole_res_dir, self.file_prefix + ".name"), 'wb') as file:
                pickle.dump(self.name, file)
        except Exception as e:
            print(f"Error saving names: {str(e)}")

        # try:
        #     with open(os.path.join(self.whole_res_dir, self.file_prefix + ".ms"), 'wb') as file:
        #         pickle.dump(self.match_status, file)
        # except Exception as e:
        #     print(f"Error saving match stati: {str(e)}")

    def read(self): 
        try:
            with open(os.path.join(self.whole_res_dir, self.file_prefix + ".res"), 'rb') as file:
                self.res = pickle.load(file)
        except Exception as e:
            print(f"Error loading tokens: {str(e)}")

        try:
            with open(os.path.join(self.whole_res_dir, self.file_prefix + ".tok"), 'rb') as file:
                self.tok = pickle.load(file)
        except Exception as e:
            print(f"Error loading tokens: {str(e)}")

        try:
            with open(os.path.join(self.whole_res_dir, self.file_prefix + ".name"), 'rb') as file:
                self.name = pickle.load(file)
        except Exception as e:
            print(f"Error loading names: {str(e)}")
        
        # try:
        #     with open(os.path.join(self.whole_res_dir, self.file_prefix + ".ms"), 'rb') as file:
        #         self.match_status = pickle.load(file)
        # except Exception as e:
        #     print(f"Error loading match stati: {str(e)}")


class ReconResHandler: 
    def __init__(self, whole_res_dir, file_prefix="", res=None, tok=None, name=None):
        # whole_res_dir: dir to place whole res together, not for the purpose of individual plotting, but for general inspection. 
        self.whole_res_dir = whole_res_dir
        self.file_prefix = file_prefix
        self.res = res    # res, list
        self.tok = tok    # token, list: here the token is the original mel
        self.name = name  # name, list, name of each corresponding point

    def save(self): 
        try:
            with open(os.path.join(self.whole_res_dir, self.file_prefix + ".res"), 'wb') as file:
                pickle.dump(self.res, file)
        except Exception as e:
            print(f"Error saving tokens: {str(e)}")

        try:
            with open(os.path.join(self.whole_res_dir, self.file_prefix + ".tok"), 'wb') as file:
                pickle.dump(self.tok, file)
        except Exception as e:
            print(f"Error saving tokens: {str(e)}")

        try:
            with open(os.path.join(self.whole_res_dir, self.file_prefix + ".name"), 'wb') as file:
                pickle.dump(self.name, file)
        except Exception as e:
            print(f"Error saving names: {str(e)}")

    def read(self): 
        try:
            with open(os.path.join(self.whole_res_dir, self.file_prefix + ".res"), 'rb') as file:
                self.res = pickle.load(file)
        except Exception as e:
            print(f"Error loading tokens: {str(e)}")

        try:
            with open(os.path.join(self.whole_res_dir, self.file_prefix + ".tok"), 'rb') as file:
                self.tok = pickle.load(file)
        except Exception as e:
            print(f"Error loading tokens: {str(e)}")

        try:
            with open(os.path.join(self.whole_res_dir, self.file_prefix + ".name"), 'rb') as file:
                self.name = pickle.load(file)
        except Exception as e:
            print(f"Error loading names: {str(e)}")

class WordEncodeResHandler: 
    def __init__(self, whole_res_dir, file_prefix="", res=None, name=None):
        # whole_res_dir: dir to place whole res together, not for the purpose of individual plotting, but for general inspection. 
        self.whole_res_dir = whole_res_dir
        self.file_prefix = file_prefix
        self.res = res    # res, list, each encoded rec an item (each item is np.array(?)) 
        self.name = name  # name, list, wuid of each corresponding encoded rec

    def save(self): 
        try:
            with open(os.path.join(self.whole_res_dir, self.file_prefix + ".res"), 'wb') as file:
                pickle.dump(self.res, file)
        except Exception as e:
            print(f"Error saving res: {str(e)}")

        try:
            with open(os.path.join(self.whole_res_dir, self.file_prefix + ".name"), 'wb') as file:
                pickle.dump(self.name, file)
        except Exception as e:
            print(f"Error saving names: {str(e)}")

    def read(self): 
        try:
            with open(os.path.join(self.whole_res_dir, self.file_prefix + ".res"), 'rb') as file:
                self.res = pickle.load(file)
        except Exception as e:
            print(f"Error loading res: {str(e)}")

        try:
            with open(os.path.join(self.whole_res_dir, self.file_prefix + ".name"), 'rb') as file:
                self.name = pickle.load(file)
        except Exception as e:
            print(f"Error loading names: {str(e)}")


class DictResHandler: 
    def __init__(self, whole_res_dir, file_prefix=""):
        # whole_res_dir: dir to place whole res together, not for the purpose of individual plotting, but for general inspection. 
        self.whole_res_dir = whole_res_dir
        self.file_prefix = file_prefix
        self.res = {}

    def save(self): 
        try:
            with open(os.path.join(self.whole_res_dir, self.file_prefix + ".res"), 'wb') as file:
                pickle.dump(self.res, file)
        except Exception as e:
            print(f"Error saving res: {str(e)}")

    def read(self): 
        try:
            with open(os.path.join(self.whole_res_dir, self.file_prefix + ".res"), 'rb') as file:
                self.res = pickle.load(file)
        except Exception as e:
            print(f"Error loading res: {str(e)}")


class ClusterHandler: 
    def __init__(self, put_path, data=None):
        self.put_path = put_path
        self.data = data

    def save(self): 
        try:
            with open(self.put_path, 'wb') as file:
                pickle.dump(self.data, file)
        except Exception as e:
            print(f"Error saving info: {str(e)}")
    
    def read(self): 
        try:
            with open(self.put_path, 'rb') as file:
                self.data = pickle.load(file)
        except Exception as e:
            print(f"Error loading info: {str(e)}")