import sys
import os
import numpy as np
import pandas as pd
def main():
    def check_inputs(weights, impacts, num_of_cols):
        if not isinstance(impacts, str) or not isinstance(weights, str):
          raise ValueError("Impacts and weights need to be strings and should be separated by commas.")
        i_list = impacts.split(',')
        w_list = weights.split(',')

        if len(i_list) != len(w_list) or len(i_list) != num_of_cols-1:
          raise ValueError("The number of impacts and weights need to be the same.")
    
        for impact in i_list:
          if impact not in ['+', '-']:
            raise ValueError("Impacts can be either '+' or '-'.")
        try:
            w_list = [float(weight) for weight in w_list]
        except ValueError:
            raise ValueError("Weights need to be numeric values.")
    
        return w_list, i_list
       
        def topsis(input_file, weights, impacts, output_file):
            try:
                if not os.path.exists(input_file):
                  raise FileNotFoundError("Input File not found.")
            except FileNotFoundError as e:
                print(e)
                exit(1)
                df = pd.read_csv(input_file)
        
            if len(df.columns) < 3:
              raise ValueError("Input file contains atleast three columns.")
    
            weights, impacts = check_inputs(weights, impacts, len(df.columns))
    
            data = df.iloc[:, 1:].values
        
            d = np.sqrt(np.sum(data**2, axis=0))
            normalized = data/d
            weighted = normalized * weights
            best = np.zeros(weighted.shape[1])
            worst = np.zeros(weighted.shape[1])
    
            for i in range(weighted.shape[1]):
                if impacts[i]=="+":
                    best[i] = np.max(weighted[:, i])
                    worst[i] = np.min(weighted[:, i])
                else:
                    best[i] = np.min(weighted[:, i])
                    worst[i] = np.max(weighted[:, i])
            
            m1 = np.sqrt(np.sum((weighted - best) ** 2, axis=1))     
            m2 = np.sqrt(np.sum((weighted - worst) ** 2, axis=1))   
            total = m1 + m2
            performance = m2/total
            df['Topsis Score'] = performance
            df['Ranks'] = df['Topsis Score'].rank(ascending=False).astype(int)
            df.to_csv(output_file, index=False)
    
            print(f"Results saved to {output_file}")
            if name=="main":
                if len(sys.argv) != 5:
                    print("Usage: python <program.py> <input_file> <Weights>  <Impacts> <output_file>")
                    exit(1)
    
            input_file = sys.argv[1]
            weights = sys.argv[2]
            impacts = sys.argv[3]
            output_file = sys.argv[4]
        
            topsis(input_file, weights, impacts,output_file)