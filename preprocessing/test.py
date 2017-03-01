import numpy as np

def weight_func(x):
        # #x_ij is the occcurencc count, alpha is the scaling param, x_max is the cutoff
        # if x_ij < x_max:
        #     return (x_ij/x_max)^a
        return 2

cooccur_mat = np.random.rand(2 , 2)
print cooccur_mat

nonzero =np.nonzero(cooccur_mat > 0.2)

weight_func = np.vectorize(weight_func)

result = weight_func(cooccur_mat[nonzero])

cooccur_mat[nonzero] = result

print zip(nonzero[0], nonzero[1])