import sys
import numpy as np
import pandas as pd

def topsis(input_file, weights, impacts, output_file):
    data = pd.read_csv(input_file)
    decision_matrix = data.iloc[:, 1:].values
    weights = np.array([float(w) for w in weights.split(',')])
    impacts = impacts.split(',')

    # Normalize the decision matrix
    normalized_matrix = decision_matrix / np.sqrt((decision_matrix ** 2).sum(axis=0))

    # Calculate the weighted normalized matrix
    weighted_normalized_matrix = normalized_matrix * weights

    # Determine the ideal and negative ideal solutions
    ideal_solution = np.array([
        weighted_normalized_matrix[:, i].max() if impacts[i] == '+' else weighted_normalized_matrix[:, i].min()
        for i in range(len(weights))
    ])
    negative_ideal_solution = np.array([
        weighted_normalized_matrix[:, i].min() if impacts[i] == '+' else weighted_normalized_matrix[:, i].max()
        for i in range(len(weights))
    ])

    # Calculate the separation measures
    euclidean_distance_ideal = np.sqrt(((weighted_normalized_matrix - ideal_solution) ** 2).sum(axis=1))
    euclidean_distance_negative_ideal = np.sqrt(((weighted_normalized_matrix - negative_ideal_solution) ** 2).sum(axis=1))

    # Calculate the relative closeness to the ideal solution
    closeness_coefficients = euclidean_distance_negative_ideal / (euclidean_distance_ideal + euclidean_distance_negative_ideal)

    # Determine the best alternative
    best_alternative = np.argmax(closeness_coefficients)

    data['Closeness Coefficient'] = closeness_coefficients
    data['Rank'] = data['Closeness Coefficient'].rank(ascending=False)

    data.to_csv(output_file, index=False)
    print(f"Results in: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python topsis.py <weights> <impacts> <inputFileDataName> <outputFileDataName>")
        sys.exit(1)

    weights = sys.argv[1]
    impacts = sys.argv[2]
    input_file = sys.argv[3]
    output_file = sys.argv[4]

    topsis(input_file, weights, impacts, output_file)
