import pandas as pd
import numpy as np
import sys

class MyFrame(pd.DataFrame): 
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs)
        for col in self.columns:
            if self.dtypes[col] == "O":
                self[col] = pd.to_numeric(self[col], errors='ignore')
    @property
    def _constructor(self): 
        return type(self)

def get_frame(N): 
    return MyFrame(
        data=np.vstack(
            [np.where(np.random.rand(N) > 0.36, np.random.rand(N), np.nan) for _ in range(10)]
        ).T, 
        columns=[f"col{i}" for i in range(10)]
    )

#When N is smallish, no issue
#frame = get_frame(1)
#frame.dropna(subset=["col0", "col1"])
#print("1 passed")

#Accept a data size value 
if __name__ == '__main__':
    N = int(sys.argv[1])
    frame = get_frame(N)
    frame.dropna(subset=["col0", "col1"])
    print(f"{N} passed")

# When N is largeish, `dropna` recurses in the `__init__` through `self.dtypes[col]` access
#frame = get_frame(6000)
#frame.dropna(subset=["col0", "col1"])
#print("6000 passed")
