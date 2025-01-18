import numpy as np
#DECISION MATRIX
decision_matrix=np.array([
    [9,7,1,2],
    [11,3,5,7],
    [6,5,4,6],
    [8,9,3,7],
    [7,9,4,5],
])
#Criteria Weights
weights=np.array([0.4, 0.3, 0.1,0.2])
#Normalize the decision matrix
normalized_matrix=decision_matrix/np.sqrt((decision_matrix**2).sum(axis=0))
#Calculate the weighted normalized matrix
weighted_normalized_matrix=normalized_matrix *weights
#Determine the ideal and negative ideal solutions
ideal_solution=np.array([weighted_normalized_matrix[:,i].max() if weights[i] > 0 else weighted_normalized_matrix[:,i].min() for i in range(len(weights))])
negative_ideal_solution=np.array([weighted_normalized_matrix[:,i].min() if weights[i] > 0 else weighted_normalized_matrix[:,i].max() for i in range(len(weights))])
#Calculate the separation measures
euclidean_distance_ideal = np.sqrt(((weighted_normalized_matrix - ideal_solution)**2).sum(axis=1))
euclidean_distance_negative_ideal = np.sqrt(((weighted_normalized_matrix - negative_ideal_solution)**2).sum(axis=1))
#Calculate teh relative closeness to the ideal solution
closeness_coefficients=euclidean_distance_negative_ideal / (euclidean_distance_ideal + euclidean_distance_negative_ideal)
#Determine the best alternative 
best_alternative = np.argmax(closeness_coefficients)
print("Closeness Coefficients: ", closeness_coefficients)
print("Best Alternative: ", best_alternative +1)