import math

def permutations_with_replecement(elements,n):
    return permutations_helper(elements,[0]*n,n-1)#this is generator

def permutations_helper(elements,result_list,d):
    if d<0:
        yield tuple(result_list)
    else:
        for i in elements:
            result_list[d]=i
            all_permutations = permutations_helper(elements,result_list,d-1)#this is generator
            for g in all_permutations:
                yield g

class solver():

    operators = ["+","-","/","*","**", "%", "&", "|", "^", "<<", ">>"]    
    
    def __init__(self, values, goal):
        self.values = values
        self.goal = goal
        self.solutions = set()
        
    
    def get_solutions(self):
        for operator in permutations_with_replecement(solver.operators,3):
            # ((n + m) + k) + p
            expression = "((" + str(self.values[0]) + operator[0] + \
                         str(self.values[1]) + ")" + operator[1] + \
                         str(self.values[2]) + ")" + operator[2] + \
                         str(self.values[3])
            answer = 0
            try:
                answer = eval(expression)
            except ZeroDivisionError:
                continue
            except ValueError:
                continue
            
            if answer == self.goal or answer == float(self.goal):
                print(expression, answer)
            # n + (m + (k + p))
            expression = str(self.values[0]) + operator[0] + \
                         "(" + str(self.values[1]) + operator[1] + \
                         "(" + str(self.values[2]) + operator[2] + \
                         str(self.values[3]) + "))"
            answer = 0
            try:
                answer = eval(expression)
            except ZeroDivisionError:
                continue
            except ValueError:
                continue
            
            if answer == self.goal or answer == float(self.goal):
                print(expression, answer)
                
            #(n + m) + (k + p)
            expression = "(" + str(self.values[0]) + operator[0] + \
                         str(self.values[1]) + ")" + operator[1] + \
                         "(" + str(self.values[2]) + operator[2] + \
                         str(self.values[3]) + ")"
            answer = 0
            try:
                answer = eval(expression)
            except ZeroDivisionError:
                continue
            except ValueError:
                continue
            
            if answer == self.goal or answer == float(self.goal):
                print(expression, answer)
                
            
s = solver([2,4,6,2], 24)
s.get_solutions()
