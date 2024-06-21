import numpy as np
import DPMed as DPM

min_point = -.5 # arbitrary bounds for p25 and p75 estimates
max_point = 1.5
min_slope = -50 # arbitrary bounds for slopes
max_slope = 50
default_beta = 9.0
default_beta_prop = 0.5
default_d = 3
default_delta = 10.0**(-6.0)
default_k = 3 # k=-1 results in the full TS

def computeMatchingHalf(x, y, n, xnew, min_est=min_point, max_est=max_point): # Nghĩa là tạo ra số cặp bằng 1/2 số phần tử 
    xnew_ests = [[] for i in range(len(xnew))]

    z = np.arange(n)
    z = np.random.permutation(z)

    for i in range(0, n-1, 2):
        p = z[i]
        q = z[i+1]
        
        if x[q] != x[p]:
            slope = float(y[q]-y[p])/ float(x[q] - x[p]) 
            xmean = (x[q]+x[p])/2
            ymean = (y[q]+y[p])/2
            for i in range(len(xnew)):
                xnew_ests[i].append(slope*np.array(xnew[i])+(ymean-slope*xmean))

    return xnew_ests # Số phần từ trong xnew_est là n / 2

def computeMatchingAll(x, y, n, xnew, min_est=min_point, max_est=max_point): # Nghĩa là tạo ra số cặp bằng số phần tử. 
    xnew_ests = [[] for i in range(len(xnew))]

    for p in range(n):
        for q in range(p+1, n):
            x_delta = float(x[q]-x[p])
            if x_delta != 0:
                slope = float(y[q]-y[p]) / float(x_delta) # compute slope between two points
                xmean = (x[q]+x[p])/2 
                ymean = (y[q]+y[p])/2
                for i in range(len(xnew)):
                    xnew_ests[i].append(slope*np.array(xnew[i])+(ymean-slope*xmean))

    return xnew_ests # Số phần tử trong xnew_ests là n * (n - 1) / 2

def computeMatchingkpair(x, y, n, xnew, k, min_est=min_point, max_est=max_point):
    xnew_ests = [[] for i in range(len(xnew))]

    z = np.arange(n)
    z = np.random.permutation(z)
    a = np.random.choice(n-1, k, replace=False)
    
    for j in a:
        p = z[j]
        q = z[n-1]
                
        for i in range(1,int((n-1) / 2 + 1)):
            p = z[(j-i) % (n-1)]
            q = z[(j+i) % (n-1)]

            if float(x[q] - x[p]) != 0: # instead of setting x_delta to 0.001, just don't compute slope if x_delta is 0
                slope = float(y[q]-y[p])/ float(x[q] - x[p]) # compute slope between two points
                xmean = (x[q]+x[p])/2
                ymean = (y[q]+y[p])/2
                for m in range(len(xnew)):
                    xnew_ests[m].append(slope*np.array(xnew[m])+(ymean-slope*xmean))

    return xnew_ests


def clipEsts(xnew_ests, min_est=min_point, max_est=max_point):
    for i in range(len(xnew_ests)):
        xnew_ests[i] = list(np.clip(np.array(xnew_ests[i]), min_est, max_est)) # gọt lại các giá trị đã ước lượng. 

    return xnew_ests


def prepForDPMedian(x, y, n, eps, xnew, half=False):  # Hma2 này cắt gọt các giá trị ước lượng cho phù hợp
    if half:
        xnew_ests = computeMatchingHalf(x, y, n, xnew)
        new_eps = eps
    else:
        xnew_ests = computeMatchingAll(x, y, n, xnew)
        new_eps = float(eps) / float(n-1)

    xnew_ests = clipEsts(xnew_ests, min_est=min_point, max_est=max_point)
    return xnew_ests, new_eps


def DPTheilSenExp(x, y, xm, ym, n, eps, xnew, half=False):  
    xnew_ests, new_eps = prepForDPMedian(x, y, n, eps, xnew, half)
    xnew_dp_ests = []
    for i in range(len(xnew)):
        xnew_dp_ests.append(DPM.dpMedian(xnew_ests[i], lower_bound=min_point, upper_bound=max_point, epsilon=new_eps))
    return xnew_dp_ests


def DPTheilSenWide(x, y, xm, ym, n, eps, xnew, width = 0.01, half=False):
    xnew_ests, new_eps = prepForDPMedian(x, y, n, eps, xnew, half)

    xnew_dp_ests = []
    
    for i in range(len(xnew)):
        # xnew_dp_ests.append(dpMedianExpWide(xnew_ests[i], lower_bound=min_point, upper_bound=max_point, epsilon=new_eps, width=width))       
        xnew_dp_ests.append(DPM.DPMedExpWide(xnew_ests[i], lower_bound=min_point, upper_bound=max_point, epsilon=new_eps, width=width))
    return xnew_dp_ests


def DPTheilSenSS(x, y, xm, ym, n, eps, xnew, beta_prop=default_beta_prop, d=default_d, half=False, k=default_k):
    if k < 0:
        k = len(x) - 1
    else:
        k = k
    
    if beta_prop > 1.0:
        return 
    
    xnew_est = computeMatchingkpair(x, y, n, xnew, k, min_point, max_point)
    xnew_est = clipEsts(xnew_est, min_point, max_point)
    beta = float(eps) * beta_prop / float(d + 1)

    xnew_dp_ests = []
    # def DPMedThielSen_Student(x, epsilon, beta, smooth_sens, d):
    for i in range(len(xnew)):
        smooth_sen = DPM.smooth_sens_coef(n, xnew_est[i], beta, min_point, max_point, k)
        xnew_dp_ests.append(DPM.DPMedThielSen_Student(xnew_est[i], epsilon=eps, beta=beta, smooth_sens=smooth_sen, d = d))
    
    return xnew_dp_ests



