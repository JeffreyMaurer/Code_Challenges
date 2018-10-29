import itertools as it


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
                

def unique_permutations(elements):
    if len(elements) == 1:
        yield (elements[0],)
    else:
        unique_elements = set(elements)
        for first_element in unique_elements:
            remaining_elements = list(elements)
            remaining_elements.remove(first_element)
            for sub_permutation in unique_permutations(remaining_elements):
                yield (first_element,) + sub_permutation
    
p = list(range(5))
for cp in permutations_with_replecement(p,5):
    print(cp)
    
for cp in unique_permutations(p,5):
    print(cp)
