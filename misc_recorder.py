import pickle

# Define recorders of training hists, for ease of extension
class Recorder: 
    def __init__(self, IOPath): 
        self.record = []
        self.IOPath = IOPath

    def save(self): 
        pass
    
    def append(self, content): 
        self.record.append(content)
    
    def get(self): 
        return self.record
    

class ListRecorder(Recorder): 
    def read(self): 
        # only used by loss hists 
        with open(self.IOPath, 'rb') as f:
            self.record = pickle.load(f)
    
    def save(self): 
        with open(self.IOPath, 'wb') as file:
            pickle.dump(self.record, file)


class DictRecorder(Recorder):
    def __init__(self, IOPath): 
        self.record = {}
        self.IOPath = IOPath

    def read(self): 
        # only used by loss hists 
        with open(self.IOPath, 'rb') as f:
            self.record = pickle.load(f)
    
    def save(self): 
        with open(self.IOPath, 'wb') as file:
            pickle.dump(self.record, file)

    def append(self, content:tuple): 
        key, value = content
        self.record[key] = value


class DfRecorder(): 
    def __init__(self, IOPath): 
        self.record = pd.DataFrame()
        self.IOPath = IOPath
    def read(self): 
        self.record = pd.read_csv(self.IOPath)
    
    def save(self): 
        self.record.to_csv(self.IOPath, index=False)

    def append(self, content): 
        self.record = pd.concat([self.record, pd.DataFrame([content])], ignore_index=True)

    def get(self): 
        return self.__convert_lists_to_arrays(self.record.to_dict('list'))
    
    @staticmethod
    def __convert_lists_to_arrays(input_dict):
        """
        Convert all lists in a dictionary to NumPy arrays.

        Args:
        - input_dict (dict): Dictionary with lists as values.

        Returns:
        - dict: Dictionary with lists converted to NumPy arrays.
        """
        output_dict = {}
        for key, value in input_dict.items():
            if isinstance(value, list):
                output_dict[key] = np.array(value)
            elif isinstance(value, dict):
                output_dict[key] = DfRecorder.__convert_lists_to_arrays(value)
            else:
                output_dict[key] = value
        return output_dict

class HistRecorder(Recorder):     
    def save(self): 
        with open(self.IOPath, "a") as txt:
            txt.write("\n".join(self.record))
    
    def print(self, content): 
        self.append(content)
        print(content)