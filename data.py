import numpy as np

def get_transition_matrix(data_path, n_grades=6, T=40):

    f = open(data_path,'r')

    grades = ["G" + str(i) for i in range(1,n_grades+1)]

    transition_matrix = np.zeros((n_grades*T+1,n_grades*T+1))

    for lines in f.readlines():
        lines = lines.replace('\n','').split(',')
        
        prevstate = 0

        for grade,t in zip(lines[1:],range(1,T+1)):
                
            if grade == '':
                break
            elif grade == 'VL':
                break
            elif grade == 'IVL':
                state = n_grades*T
            else:
                grade = grades.index(grade)
                state = grade * T + t
            
            transition_matrix[prevstate,state] += 1.0
            prevstate = state

    for row in range(n_grades*T):
        rowsum = np.sum(transition_matrix[row])
        if  rowsum != 0:
            transition_matrix[row] = transition_matrix[row]/rowsum
        
        elif row % T != 39:
            transition_matrix[row][-1] = 1.0

    
    return transition_matrix
