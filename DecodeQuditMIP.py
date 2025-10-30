# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 22:37:11 2025

Companion code for
Qudit low-density parity-check codes
Daniel J. Spencer, Andrew Tanggara, Tobias Haug, Derek Khu, Kishor Bharti
arXiv:2510.06495

Simulates qudit QEC codes for code capacity experiments
Also computes code distance for qudits

Generates codes used in paper

Generates X errors and applies Z syndromes + correction

Assumes equal errors for all qudit error types

Decoding via MIP package

Needs galois and mip package to be installed via pip

pip install galois
pip install mip

@author: Tobias Haug @TII

tobias.haug@u.nus.edu

"""

import numpy as np

import mip
import time
import scipy

import galois


##Tip: add random stabilizer to logicOp, this greatly enhances stability and convergence
def distance_test_qudit(stab,logicOp,qudit,max_time=mip.INF):
    
    assert qudit!=4 ##not properly impelemtned for non-prime qudit numbers...
    
    ##turn into np array since we do not want GF here
    logicOp_tmp=np.array(logicOp)
    
    stab_tmp=np.array(stab)
    
    start_time=time.time()
    # number of qubits
    n = stab_tmp.shape[1]
    # number of stabilizers
    m = stab_tmp.shape[0]

    # maximum stabilizer weight
    wstab = np.max([np.sum((stab_tmp[i,:]%qudit)!=0) for i in range(m)])
	# weight of the logical operator
    wlog = np.count_nonzero(logicOp_tmp)
	# how many slack variables are needed to express orthogonality constraints modulo qudit
    
    ##does this need to increase for qudits?
    ##factor (qudit-1)**2 comes from fact that largest possible value is (qudit-1)**2 
    ##this is from stabilizer check weight qudit-1, and qudit itsself has qudit-1 as mixmal value
    ##as we take product, we get maximally (qudit-1)**2 over at most wstab qubits
    
    #print(stab)
    num_anc_stab = int(np.ceil(np.log((qudit-1)**2*wstab)/np.log(qudit)))
    num_anc_logical = int(np.ceil(np.log((qudit-1)**2*wlog)/np.log(qudit)))
    #print(num_anc_stab,num_anc_logical)
	# total number of variables
    num_var = n + m*num_anc_stab + num_anc_logical

    model = mip.Model()
    model.verbose = 0
    ##variables, 
    ##restriction between 0 and qudit-1
    x = [model.add_var(var_type=mip.INTEGER,lb=0,ub=qudit-1) for i in range(num_var)]
    
    
    #x = [model.add_var(var_type=CONTINUOUS) for i in range(num_var)]
    ##minimze weight on non-slack variables
    ##we want to have the logical error with minimal support
    ##there is no difference between 1,2,..,qudit-1 values, so we need to add the !=0
    ##if x is 0, it is counted as False=0, else 1
    if(qudit==2):
        model.objective = mip.minimize(mip.xsum(x[i] for i in range(n)))
    
        
    else:
        # ##the constraint must be chosen such that when x[i]!=0, the minimziation yields 1, else 0
        # ##cannot use miniization over x[i] for qudits as it can take higher values!
        
        
        y=[[model.add_var(var_type=mip.INTEGER,lb=0,ub=1) for j in range(qudit-1) ] for i in range(n)]
        
        
        for i in range(n):
            ##ensures that ys are 0 when x=0, else one of the ys>0
            model+= x[i]-mip.xsum((j+1)*y[i][j] for j in range(qudit-1)) ==0
            
            ##this ensures that only one y is triggered at a time, so that we can use as a hamming weight
            model+= mip.xsum(y[i][j] for j in range(qudit-1)) <=1
            
        ##this is 0 when x=0, else 1
        ##this is exactly the hamming weight, i.e. indicating whether x[i] differs from 0
        model.objective = mip.minimize(mip.xsum(mip.xsum(y[i][j] for j in range(qudit-1)) for i in range(n) ))

        
        # ##the constraint must be chosen such that when x[i]!=0, the minimziation yields 1, else 0
        # ##cannot use miniization over x[i] for qudits as it can take higher values!
        
        


    weight_stab=[]
	# orthogonality to rows of stab constraints
    for row in range(m): ##go through stabilizers
        weight = np.zeros(num_var,dtype=int)#[0]*num_var
        # supp = np.nonzero(stab[row,:])[0] ##support of stabilizer
        # #print(supp)
        # for q in supp:
        #     weight[q] = stab[row,q]%qudit
            
        weight[:n]=stab_tmp[row,:]%qudit
        ##slack variables to account for modulo 2
        cnt = 1
        for q in range(num_anc_stab):
            ##slack variables which give modulo qudit
            weight[n + row*num_anc_stab +q] = -(qudit**cnt)#(1<<cnt)## -2**cnt
            cnt+=1
        ##should commute with stabilizer %qudit
        model+= mip.xsum(weight[i] * x[i] for i in range(num_var)) == 0
        weight_stab.append(list(weight))




    #print(logicOp_tmp)
	# non-zero overlap with logicOp constraint
    ##anti-commute with logical

    #weight = [0]*num_var
    weight = np.zeros(num_var,dtype=int)
    
    #    supp = np.nonzero(logicOp_tmp)[0]
    # for q in supp:
    #     weight[q] = logicOp_tmp[q]%qudit
        
        
    weight[:n]=logicOp_tmp
        
    ##slack variables to account for modulo 2
    cnt = 1
    for q in range(num_anc_logical):
        ##slack variables which give modulo qudit
        weight[n + m*num_anc_stab +q] = -(qudit**cnt)#-(1<<cnt)
        cnt+=1
    #print(weight)
    ##the anti-commutation condition, now must not be 0 %qudit
    model+= mip.xsum(weight[i] * x[i] for i in range(num_var)) >= 1
    model+= mip.xsum(weight[i] * x[i] for i in range(num_var)) <= qudit-1
        
    # for i in range(num_var):
    #     model += xsum([x[i]])>=0
    #     model += xsum([x[i]])<=1
    
    #max_time=0
    
    res=model.optimize(max_seconds=max_time)
    
    end_time=time.time()-start_time
    
    print(res,"time:",end_time)
    
    #print(res==res.NO_SOLUTION_FOUND)
    
    if(res==res.NO_SOLUTION_FOUND):
        return -1
    
    #print(weight)
    #print(weight_stab)

    #print([x[i].x for i in range(n)])

    ##we want to have the logical error with minimal support
    ##there is no difference between 1,2,..,qudit-1 values, so we need to add the !=0
    ##if x is 0, it is counted as False=0, else 1
    opt_val = int(sum([x[i].x!=0 for i in range(n)]))
    

    return opt_val


def qudit_css_code(qudit,HX,HZ,get_logicals=True):
    

    # Initialize Galois field array
    GF = galois.GF(qudit)
    
    # Convert HX and HZ to GF data structure
    HX_GF = GF(HX)
    #HXT_GF = GF(HX.T)
    HZ_GF = GF(HZ)
    #HZT_GF = GF(HZ.T)
    
    n=np.shape(HX)[1]
    
    

    
    ##number of logicals
    K=n-np.linalg.matrix_rank(HX_GF)-np.linalg.matrix_rank(HZ_GF)
    
    #print(HZ_GF@(HX_GF.T))
    
    
    
    ##get logicals
    def get_logicals_z(X_kernel,Z_row_space):
        ##adapted from https://pypi.org/project/bposd/
        ##extended to qudits using galois package
        
        # from ldpc import mod2
    
        
        # ker_hx=mod2.nullspace(HX) #compute the kernel basis of hx
        # im_hzT=mod2.row_basis(HZ) #compute the image basis of hz.T
        
        # #in the below we row reduce to find vectors in kx that are not in the image of hz.T.
        # log_stack=np.vstack([im_hzT,ker_hx])
        # pivots=mod2.row_echelon(log_stack.T)[3]
        # log_op_indices=[i for i in range(im_hzT.shape[0],log_stack.shape[0]) if i in pivots]
        # log_ops=log_stack[log_op_indices]
        
        log_stack=np.vstack([Z_row_space,X_kernel])
        
        ##get row echlon form
        row_reduced=(log_stack.T).row_reduce()
        
        ##list of pivots (i.e. first non-zero column in each row)
        pivots=np.argmax(row_reduced!=0,axis=1)
        pivots=pivots[:np.argmax(pivots)+1]
        
        log_op_indices=[i for i in range(Z_row_space.shape[0],log_stack.shape[0]) if i in pivots]
        log_ops=log_stack[log_op_indices]
        return log_ops


    if(get_logicals==True and K>0):
        
        
        ##check that HX and HZ commute
        assert not (HZ_GF@(HX_GF.T)).any()
        assert not (HX_GF@(HZ_GF.T)).any()
        
        
        # Calculate kernels and row spaces (mod q)
        X_kernel = HX_GF.null_space()
        X_row_space = (HX_GF.T).column_space()
        Z_kernel = HZ_GF.null_space()
        Z_row_space = (HZ_GF.T).column_space()
        
        # print(X_kernel)
        # print(Z_row_space)
        
    
        
        
        lz=get_logicals_z(X_kernel,Z_row_space)
        lx=get_logicals_z(Z_kernel,X_row_space)
        
        assert n==HZ.shape[1]==lz.shape[1]==lx.shape[1]
        assert K==lz.shape[0]==lx.shape[0]
        
        ##check that logicals in kernel
        assert not (HZ_GF@lx.T).any()
        assert not (HX_GF@lz.T).any()
        
        ##check that logical operators span K logical qubits
        assert np.linalg.matrix_rank((lx@lz.T))==K
        
        
        # from bposd.css import css_code
        # qcode=css_code(HX,HZ)
        # qcode.test()
        # # print(qcode.lx)
        # # print(qcode.lz)
        
        # #print((HZ_GF@lx.T).any())
        
        # ##assert np.linalg.matrix_rank((GF(qcode.lx)@lz.T))==K
        
    else:
        lz=[]
        lx=[]
        
    return n,K,HZ,HX,lz,lx

#def get_bicycle_code(ell,m,poly_A,poly_B,qudit=2,factor_A_in=[],factor_B_in=[]):
def get_bicycle_code(code_dict):
    #bicycle_parameters={"ell":ell,"m":m,"A":A,"B":B,"qudit":qudit,"factorA":factorA,"factorB":factorB,"gamma1":gamma1,"gamma2":gamma2,"delta1":delta1,"delta2":delta2}
    ell=code_dict["ell"]
    m=code_dict["m"]
    poly_A=code_dict["A"]
    poly_B=code_dict["B"]
    if("qudit" in code_dict.keys()):
        qudit=code_dict["qudit"]
        assert qudit!=4 ##not defined for prime power!
    else:
        qudit=2
        
    if("g" in code_dict.keys()):
        g=code_dict["g"]
    else:
        g=1
        
        
    if("factorA" in code_dict.keys() and len(code_dict["factorA"])>0):

        factor_A=code_dict["factorA"]
        factor_B=code_dict["factorB"]
        assert len(poly_A)==len(factor_A)
        assert len(poly_B)==len(factor_B)
    else:
        factor_A=[1]*len(poly_A)
        factor_B=[1]*len(poly_A)
        
    ####factor in front of HX=gamma1*A | gamma2*B, HZ=delta1*B^T | delta2*A^T
    if("gamma1" in code_dict.keys()):
        gamma1=code_dict["gamma1"]
        gamma2=code_dict["gamma2"] 
        delta1=code_dict["delta1"]
        delta2=code_dict["delta2"]
        ##from CSS
        assert (gamma1 * delta1 + gamma2 * delta2)%qudit == 0
    else:
        gamma1=1
        gamma2=1 
        delta1=1
        delta2=-1 ##default -1 for qudit>2, can be set equivalently to 1 for qudit=2
        assert (gamma1 * delta1 + gamma2 * delta2)%qudit == 0
        
    # #print(qudit)
    # if(len(factor_A_in)>0):
    #     assert len(poly_A)==len(factor_A_in)
    #     factor_A=factor_A_in
    # else:
    #     factor_A=[1]*len(poly_A)
        
    # if(len(factor_B_in)>0):
    #     assert len(poly_B)==len(factor_B_in)
    #     factor_B=factor_B_in
    # else:
    #     factor_B=[1]*len(poly_A)
    
    # Number of physical qudits
    n = 2*ell*m
    

    # define cyclic shift matrices 
    I_ell = np.identity(ell,dtype=np.int8)
    I_m = np.identity(m,dtype=np.int8)
    
    if(g>1):
        I_g = np.identity(g,dtype=np.int8)
    
    if(type(code_dict["A"][0])==str):
        ##old format as ("x2","y3",...)
        A_terms=[]
        for i in range(len(code_dict["A"])):
            if(code_dict["A"][i][0]=="x"):
                A_terms.append((int(code_dict["A"][i][1:]),0))
            elif(code_dict["A"][i][0]=="y"):
                A_terms.append((0,int(code_dict["A"][i][1:])))
            else:
                raise NameError("not defined")
                
        B_terms=[]
        for i in range(len(code_dict["B"])):
            if(code_dict["B"][i][0]=="x"):
                B_terms.append((int(code_dict["B"][i][1:]),0))
            elif(code_dict["B"][i][0]=="y"):
                B_terms.append((0,int(code_dict["B"][i][1:])))
            else:
                raise NameError("not defined")
                
    else:
        ##new format as list of [(x,y),(x,y),...]
        A_terms=code_dict["A"]
        B_terms=code_dict["B"]

    A=np.zeros([ell*m*g,ell*m*g],dtype=np.int8)
    for i in range(len(A_terms)):
      xp=np.kron(np.roll(I_ell, A_terms[i][0], axis=1), I_m)
      yp=np.kron(I_ell, np.roll(I_m, A_terms[i][1], axis=1))
      A+=(factor_A[i]*np.dot(xp,yp))%qudit

    B=np.zeros([ell*m*g,ell*m*g],dtype=np.int8)
    for i in range(len(B_terms)):
      xp=np.kron(np.roll(I_ell, B_terms[i][0], axis=1), I_m)
      yp=np.kron(I_ell, np.roll(I_m, B_terms[i][1], axis=1))
      B+=((factor_B[i])*np.dot(xp,yp))%qudit
    
    
    HX = np.hstack((gamma1*A, gamma2*B)).astype(np.int8) % qudit
    HZ = np.hstack((delta1*np.transpose(B), delta2*np.transpose(A))).astype(np.int8) % qudit
    
    return HX,HZ

class MIP_decoder():

    
    def __init__(self,
                 parity_check_matrix,
                 probabilities,
                 qudit=2,
                 max_time=0,
                 retries=2,
                 ):


        #import mip
        

        self.qudit=qudit
        


        self.parity_check_matrix=parity_check_matrix
        
        
        def log_weights_2(x,y, eps=1e-20):
            weights = np.log(
                (x + eps) / (y + eps)
            )
            return weights
        
        
        ##probabilities is the prob per possible error
        ##for qudits, we have qudits-1 possible errors, so total error rate is (qudit-1)*probabilities
        
        self.weight_error=-log_weights_2(probabilities,1-(self.qudit-1)*probabilities)
        ##weights are positive as we are minimizing (i.e. 0 error is minimal solution)
        
        #print(self.weight_error)
        
        
        self.retries=retries
        
        
        if(max_time==0):
            self.max_time=mip.INF
        else:
            self.max_time=max_time
        




    def decode(self, syndrome: np.ndarray, **kwargs) -> np.ndarray:
        """Get X and Z corrections given code and measured syndrome."""
        


        syndrome = np.array(syndrome, dtype=int)
        
        

        decoder_time=time.perf_counter()

        weight_error=self.weight_error
        qudit=self.qudit
        stab=self.parity_check_matrix#.toarray()
        
        #stab=scipy.sparse.bmat(stab.toarray(),format='csr',dtype=int)
        
        #print(stab)
        

        
        #print(stab.toarray())
    
        #start_time=time.time()
        # number of qubits
        n = stab.shape[1]
        # number of stabilizers
        m = stab.shape[0]
        
        #print(stab)
        
        assert len(syndrome)==m
        

        if(len(weight_error)>0):
            
            assert len(weight_error)==n
            weight_tmp=weight_error
        else:
            weight_tmp=np.ones(n)
            

        
        #print(weight_err[0])
            
        #print(weight_err)
        # maximum stabilizer weight
        wstab = np.max([np.sum((stab[i,:])!=0) for i in range(m)])
    	# weight of the logical operator
        #print(wstab)
    	# how many slack variables are needed to express orthogonality constraints modulo qudit
        num_anc_stab = int(np.ceil(np.log((qudit-1)**2*wstab+qudit+1)/np.log(qudit)))
        #print(num_anc_stab)
        #print(stab.toarray())
        #print(num_anc_stab,num_anc_logical)
    	# total number of variables
        num_var = n + m*num_anc_stab 
        
        

                    
        for rep in range(self.retries):

            weight_err=weight_tmp#*(0.1+np.random.rand()*0.9)

        
            model = mip.Model()
            model.verbose = 0
            ##variables, 
            ##restriction between 0 and qudit-1
            x = [model.add_var(var_type=mip.INTEGER,lb=0,ub=qudit-1) for i in range(num_var)]
            
    
            
            #x = [model.add_var(var_type=CONTINUOUS) for i in range(num_var)]
            ##minimze weight on non-slack variables
            ##we want to have the logical error with minimal support
            ##there is no difference between 1,2,..,qudit-1 values, so we need to add the !=0
            ##if x is 0, it is counted as False=0, else 1
            if(qudit==2):
                model.objective = mip.minimize(mip.xsum(weight_err[i]*x[i] for i in range(n)))

            else:
                
                # ##the constraint must be chosen such that when x[i]!=0, the minimziation yields 1, else 0
                # ##cannot use miniization over x[i] for qudits as it can take higher values!
                
                y=[[model.add_var(var_type=mip.INTEGER,lb=0,ub=1) for j in range(qudit-1) ] for i in range(n)]
                
                for i in range(n):
                    ##ensures that ys are 0 when x=0, else one of the ys>0
                    model+= x[i]-mip.xsum((j+1)*y[i][j] for j in range(qudit-1)) ==0
                    
                    ##this ensures that only one y is triggered at a time, so that we can use as a hamming weight
                    model+= mip.xsum(y[i][j] for j in range(qudit-1)) <=1
                    
                ##this is 0 when x=0, else 1
                ##this is exactly the hamming weight, i.e. indicating whether x[i] differs from 0
                ##here we assume that each possible error on the qudits (quidt-1 in total) is equally likely
                model.objective = mip.minimize(mip.xsum(weight_err[i]*mip.xsum(y[i][j] for j in range(qudit-1)) for i in range(n) ))

                # raise NameError("Not implemented for qudits",qudit)
        
            #print(syndrome)
            weight_stab=[]
        	# orthogonality to rows of stab constraints
            for row in range(m): ##go through stabilizers
                #weight = [0]*num_var
                weight = np.zeros(num_var,dtype=int)
                
                # if(scipy.sparse.issparse(stab)):
                #     supp = stab[row,:].nonzero()[1] ##support of stabilizer
                # else:
                #     supp = np.nonzero(stab[row,:])[0] ##support of stabilizer
                #print(supp)
                # for q in supp:
                #     weight[q] = stab[row,q]%qudit
                    
            
                if(scipy.sparse.issparse(stab)):
                    tmp=stab[row,:].toarray()
                else:
                    tmp=stab[row,:]
                
                weight[:n]=tmp#%qudit
    
                ##slack variables to account for modulo qudit
                cnt = 1
                for q in range(num_anc_stab):
                    ##slack variables which give modulo qudit
                    weight[n + row*num_anc_stab +q] = -(qudit**cnt)#(1<<cnt)## -2**cnt
                    cnt+=1
                ##match syndrome
                model+= mip.xsum(weight[i] * x[i] for i in range(num_var)) == syndrome[row]
                weight_stab.append(list(weight))
        
        
            
        
            #max_time=0
            
            res=model.optimize(max_seconds=self.max_time)
            
            #end_time=time.time()-start_time
            
            #print(res)
            
            #print(res==res.NO_SOLUTION_FOUND)
            
            if(res==res.NO_SOLUTION_FOUND):
                correction_opt=[]
                print("No solution found (TIMEOUT)",syndrome)
            
            elif(res==res.INFEASIBLE):
                print("Infeasible",syndrome)
                correction_opt=[]
            elif(res==res.OPTIMAL or res==res.FEASIBLE):
            
                #print(weight)
                #print(weight_stab)
            
                #print([x[i].x for i in range(n)])
            
                ##we want to have the logical error with minimal support
                ##there is no difference between 1,2,..,qudit-1 values, so we need to add the !=0
                ##if x is 0, it is counted as False=0, else 1
                correction_opt = np.array([x[i].x for i in range(n)])
            else:
                print("Got other error",res)
                correction_opt=[]
            #print(correction_opt)
            
            #print([x[i].x for i in range(num_var)])
            
            # print([(weight[i] * x[i]).x for i in range(num_var)])
        
    
            sub_optimal_errors=[]
            
            if(len(correction_opt)>0):
                ##suceeded
                break
            else:
                print("MIP failed, retry",rep+1)
            if(rep==self.retries-1):
                ##did not suceed
                print("Decoding with MIP failed",rep+1,"times, assume zero correction")
                correction_opt=np.zeros(n,dtype=np.int8)

            

        decoder_time=time.perf_counter()-decoder_time
        



        return correction_opt,decoder_time




def super_fast_choice_cum(cumulative_probs, rng):
    """
    probs should be probability distribution along columns, and along row different instances
    sample from this distribution
    we are using later on cumulative probs because its faster, it can be written in one line
    """
    
    size=np.shape(cumulative_probs)[1]
    #get_time=time.perf_counter()
    #cumulative_probs=np.cumsum(probs,axis=0)
    #print(time.perf_counter()-get_time)

    x=rng.random(size)

    ##pick according to cumulative_probs
    ##cumulative probs needs to add to 1, such that last element is never true
    ##else argmin will return 0 which is inccorect
    res=np.argmin(cumulative_probs<x,axis=0)
    

    return res





def generate( n, error_rate, rng,qudit):
    #rng = np.random.default_rng() if rng is None else rng
    
    probs=np.zeros([qudit,n])
    ##identity
    probs[0,:]=1-error_rate

    ##errors, for qudit there are qudit-1 errors for X error ladder operator 
    probs[1:,:]=error_rate/(qudit-1)

        
    cumulative_probs=np.cumsum(probs,axis=0)



    ##0: Identity, 1: X, 2: Z, 3:Y
    #get_time=time.perf_counter()
    #error_pauli= np.array(super_fast_choice([p_i,p_x,p_z,p_y],rng=rng),dtype=np.int8)
    
    ##directly feed in cumulative prob to save time
    error= np.array(super_fast_choice_cum(cumulative_probs,rng=rng),dtype=np.int8)
    #print(time.perf_counter()-get_time)


    return error

def to_array(matrix):
    if isinstance(matrix, np.ndarray):
        return matrix
    else:
        return matrix.toarray()


rng=np.random.default_rng()

error_rate=0.1 ##physical error rate for qudits


###default gamma, delta factors, leave unchanged here
gamma1, gamma2, delta1, delta2=(1, 1, 1, -1)



##various code definitions, comment/uncomment to use particular code

##coprime

## [[24,4,4]]_3 code weight 5
qudit=3
ell, m = 4, 3
#A, B = ['x1', 'x2'], ['x3', 'y1', 'y2']
A, B =[(1,0), (2,0)],[(3,0), (0,1),(0,2)]
factorA, factorB = [1,1], [1,2,2] ##factor in front of polyonomials
experiment_name="New24_4_4_5_3"


# # ##code proposed by https://scirate.com/arxiv/2503.22071#2408, new way to input code
# #[[30,4,5]]_3
# qudit=3
# ell=5
# m=3
# ##A, B =['x0', 'x1'],['y0', 'y1', 'u2']
# A, B =[(0,0), (0,1), (2,2)],[(0,0), (1,0)]
# experiment_name="D30_4_5_5_3"
# factorA=[2,2,2]
# factorB=[2,1]
# gamma1, gamma2, delta1, delta2= (2, 1, 1, 1)



# ##[1+x|1+y+x3y2],l=8,m=3
# ##code  [[48,4,7]]_3 by https://arxiv.org/abs/2503.22071
# ell,m=8,3
# qudit=3
# A=[(0,0),(1,0)] ##[(x,y),(x,y),...]                   
# B=[(0,0),(0,1),(3,2)]##[(x,y),(x,y),...]   
# experiment_name="DB48_4_7_5_3"
# factorA=[2,1]
# factorB=[1,1,1]
# gamma1, gamma2, delta1, delta2= (1, 2, 1, 1)


# ##3,5,"[[88, 8, 5]]_3",4,11,2y3 + 1y8,1y1 + 1y0 + 1y9,88,8,5,5,5
# ell,m=4,11
# qudit=3
# A=[(0,3),(0,8)]##[(x,y),(x,y),...]           
# B=[(0,1),(0,0),(0,9)]##[(x,y),(x,y),...]   
# factorA=[2,1]
# factorB=[1,1,1]
# experiment_name="DB88_8_5_5_3"



# ##weight 4
# ##coprime#5,6,"[[84, 6, 5]]_5",7,6,4y3 + 5y4 + 4x1,3x2 + 5x4 + 2x6,84,6,5,5,5
# ell,m=7,6
# qudit=5
# A=[(0,3),(0,4),(1,0)]##[(x,y),(x,y),...]           
# B=[(2,0),(4,0),(6,0)]##[(x,y),(x,y),...]   
# factorA=[4,5,4]
# factorB=[3,5,2]
# experiment_name="DB84_6_5_4_5"



# # ##code proposed by https://scirate.com/arxiv/2503.22071#2408, new way to input code
# #[[30,4,5]]_5
# # test_code2={'n': 30, 'k': 4, 'd': 5, 'k/n': 1/7, 'l': 5, 'm': 3, 'pseudo_threshold': 0.0315, 'A': 'x^0 + x^2', 'B': 'y^0 + y^1+z^2'}
# qudit=5
# ell=5
# m=3
# ##A, B =['x0', 'x1'],['y0', 'y1', 'u2']
# A, B =[(0,0), (0,1), (2,2)],[(0,0), (1,0)] ##[(x,y),(x,y),...]   
# experiment_name="D30_4_5_5_5"
# factorA=[3,3,3]
# factorB=[2,3]
# gamma1, gamma2, delta1, delta2= (4, 4, 4, 1)



# ##[1+x|1+y+x3y2],l=8,m=3
# ##code  [[48,4,7]]_5 by https://arxiv.org/abs/2503.22071
# ell,m=8,3
# qudit=5
# A=[(0,0),(1,0)]##[(x,y),(x,y),...]           
# B=[(0,0),(0,1),(3,2)]##[(x,y),(x,y),...]   
# factorA=[3,1]
# factorB=[3,3,1]
# gamma1, gamma2, delta1, delta2= (4, 1, 3, 3)
# experiment_name="DB48_4_7_5_5"
# size_code=7



# ##[[54,6,6]]_5" weight 6,3,9,2y2 + 4y7 + 4y4,3y6 + 3y3 + 4y8,"[2, 4, 4]","[3, 3, 4]","['y2', 'y7', 'y4']","['y6', 'y3', 'y8']","(-3, -1, -1, 3)",54,6,6,6,6,True
# qudit=5
# ell=3
# m=9
# #A, B =['y2', 'y7', 'y4'],['y6', 'y3', 'y8']
# A=[(0,2),(0,7),(0,4)]##[(x,y),(x,y),...]           
# B=[(0,6),(0,3),(0,8)]##[(x,y),(x,y),...]   
# factorA, factorB =[2, 4, 4],[3, 3, 4]
# gamma1, gamma2, delta1, delta2=(-3, -1, -1, 3)
# experiment_name="Q54_6_6_6_5"


# ##[[64,8,5]]_5" weight 6,8,4,2x3 + 1x5 + 31,3x1 + 3x6 + 2x7,"[2, 1, 3]","[3, 3, 2]","['x3', 'x5', '1']","['x1', 'x6', 'x7']","(-3, -3, 2, 3)",64,8,5,5,5,True
# qudit=5
# ell=8
# m=4
# #A, B =['x3', 'x5', 'x0'],['x1', 'x6', 'x7']
# A=[(3,0),(5,0),(0,0)]##[(x,y),(x,y),...]           
# B=[(1,0),(6,0),(7,0)]##[(x,y),(x,y),...]   
# factorA, factorB =[2, 1, 3],[3, 3, 2]
# gamma1, gamma2, delta1, delta2=(-3, -3, 2, 3)
# experiment_name="Q64_8_5_6_5"




# #[[28,4,5]]_5",7,2,3x6 + 3x2 + 4x3,3x5 + 1x1 + 11,"[3, 3, 4]","[3, 1, 1]","['x6', 'x2', 'x3']","['x5', 'x1', '1']",28,4,5,5,5,True
# ##[[28,4,5]]_5" weight 6,7,2,3x6 + 3x2 + 4x3,3x5 + 1x1 + 11,"[3, 3, 4]","[3, 1, 1]","['x6', 'x2', 'x3']","['x5', 'x1', '1']",28,4,5,5,5,True
# qudit=5
# ell=7
# m=2
# #A, B =['x6', 'x2', 'x3'],['x5', 'x1', 'x0']
# A=[(6,0),(2,0),(3,0)]##[(x,y),(x,y),...]           
# B=[(5,0),(1,0),(0,0)]##[(x,y),(x,y),...]   
# factorA, factorB =[3, 3, 4],[3, 1, 1]
# gamma1, gamma2, delta1, delta2=(1, 1, 1, -1)
# # experiment_name="New28_4_4_6_3"




# ###0,7,6,"[[30, 4, 5]]_7",3,5,4y1 + 4y3 + 6y2,1x1 + 2x2 + 4y4,30,4,5,5,5
# ell,m=3,5
# qudit=7
# A=[(0,1),(0,3),(0,2)]##[(x,y),(x,y),...]           
# B=[(1,0),(2,0),(0,4)]##[(x,y),(x,y),...]   
# factorA=[4,4,6]
# factorB=[1,2,4]
# experiment_name="DB30_4_5_6_7"



##Lacross codes

# ##lacross codes by dan
# ##Code parameters	q	n	k	d	dx	dz	a0	a1	a2
# ##[[89, 9, 5]]_3	3	8	3	5	5	5	2	1	1
# n_lc=8
# k_lc=3
# alphas_lc=[2,1,1]
# qudit=3
# code_type=1 ##lacross
# experiment_name="LLC89_9_5_6_3"

# ##[[34, 4, 4]]_7	7	5	2	4	4	4	6	5	1
# n_lc=5
# k_lc=2
# alphas_lc=[6,5,1]
# qudit=7
# code_type=1 ##lacross
# experiment_name="LLC34_4_4_6_7"

# ##[[52, 4, 5]]_5",5,6,2,5,5,5,4,4,3
# n_lc=6
# k_lc=2
# alphas_lc=[4,4,3]
# qudit=5
# code_type=1 ##lacross
# experiment_name="LLC52_4_5_6_5"

code_dict={"ell":ell,"m":m,"A":A,"B":B,"qudit":qudit,"factorA":factorA,"factorB":factorB,"gamma1":gamma1,"gamma2":gamma2,"delta1":delta1,"delta2":delta2}

##get parity check matrix
Hx,Hz=get_bicycle_code(code_dict)


##get logicals
n,K,HZ,HX,lz,lx=qudit_css_code(qudit,to_array(Hx),to_array(Hz))



print("[[",n,",",K,"]]_"+str(qudit))


assert K>0

lx=np.array(lx,dtype=np.int8)
lz=np.array(lz,dtype=np.int8)

##get distance
distance_list_x=[]
for i in range(K):
    w =distance_test_qudit(HZ,lz[i,:],qudit)
    print('Logical qudit=',i,'Distance=',w)
    distance_list_x.append(w)
    
        
distance_x=np.amin(distance_list_x)##z type error distance

print("Found distance x",distance_x)

print("[[",n,",",K,",",distance_x,"]]_"+str(qudit))


print("Perform X error correction")

##do X error, so we need Z parity checks
parity_check_matrix=Hz

##error rate to be used as prior for decoder

probabilities_decoder=np.zeros([n])
probabilities_decoder[:]=error_rate/(qudit-1)


# probabilities=np.zeros([qudit,n])
# probabilities[0,:]=1-error_rate

# probabilities[1:qudit,:]=error_rate/(qudit-1)

# probabilities=np.mean([np.sum(probabilities[np.arange(qudit*i,(i+1)*qudit),:],axis=0) for i in range(1,qudit)],axis=0)

##intialise decoder
decoder= MIP_decoder( parity_check_matrix,
                 probabilities_decoder,
                 qudit=qudit,
                 max_time=0,
                 retries=2)



print("Generating X error")

##get errors
error = generate(n, error_rate=error_rate, rng=rng,qudit=qudit)%qudit

print("Physical error",error)

syndrome=parity_check_matrix.dot(error) % qudit

##run decoder
decoded_error,time_taken = decoder.decode(syndrome)

print("Decoded error",decoded_error)


error_after_correction=(error-decoded_error)%qudit


print("Remaining error",error_after_correction)


syndrome_after_correction=parity_check_matrix.dot(error_after_correction) % qudit

out_of_codespace=np.any(syndrome_after_correction!=0)

print("Is out of codespace:",out_of_codespace)


logical = lz.dot(error_after_correction) %qudit

logical_error=np.any(logical!=0)

print("Logical error occurred:",logical_error)

#print(time_taken)
         
    


