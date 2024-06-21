from scipy.linalg import lstsq
import numpy as np
import math
from scipy.stats import t

eps = 0.5
min_point = -.5 # Tại sao lại là -0.5?
max_point = 1.5 # Tai sao lai la 1.5?

#################################################################
def dpMedian(x, lower_bound, upper_bound, epsilon = eps, granularity = 0.01):
    """
    :param x: Mảng các thành phần x
    :param lower_bound: Giá trị nhỏ nhất trong x, mặc định là 0 
    :param upper_bound: Giá trị lớn nhất trong x, mặc định là 1
    :param epsilon: Hệ số DP
    :param granularity: Hệ số này giúp phân bổ lại giá trị của x, giúp thuật toán chính xác hơn. 
    :return: eps-DP approximation to median
    """

    x_median = np.median(x)

    if len(x) % 2 == 0:
        z = np.concatenate([x, [x_median, x_median, lower_bound, upper_bound]])
    if len(x) % 2 == 1:
        z = np.concatenate([x, [x_median, lower_bound, upper_bound]])
    
    z.sort()
    n = len(z) 

    # Sắp xếp lại các phần tử để các phần tử tránh bị tập trung quá dày đặc. 
    for i in range(math.floor(n/2)+1):
        z[i] = max(lower_bound , z[i] - granularity)
        z[n-i-1] = min(z[n-i-1] + granularity, upper_bound)

    currentMax = -np.inf
    currentInt = -1

    # Đoạn code này duyệt qua từng phần tử liền kề của x, tính toán giá trị khoảng cách bằng logarit và tính điểm theo logarit + khoảng cách tới điểm trung vị. 
    for i in range(1, n):
        start = z[i-1]
        end = z[i]

        length = end-start
        if (length <= 0):
            loglength = -np.inf
        else:
            loglength = math.log(length)

        rungheight =  abs(i - n/2)

        score = loglength - eps/2 * rungheight       

        noisyscore = score + np.random.gumbel(loc=0.0, scale=1.0)
        if (noisyscore > currentMax):
            currentInt = i
            currentMax = noisyscore

    return np.random.uniform(low=z[currentInt-1], high=z[currentInt])

    """
    Chọn và trả về giá trị nào đó ngẫu nhiên nằm giữa giá trị mà có noisy score cao nhất với giá trị trước nó. 
    Điều này đảm bảo được về mặt bảo mật cũng như về mặt giá trị này gần với trung vị nhất. 
    Tóm lại, đoạn hàm này tính toán trung vị của một mảng đầu vào một cách bảo mật. 
    """
#################################################################




#################################################################
def DPMedExpWide(x, lower_bound=min_point, upper_bound=max_point, epsilon=1.0, width=0.01):
    """
    :param x: mảng gồm các số đầu vào
    :param lower_bound: Cận dưới của x
    :param upper_bound: Cận trên của x
    :param epsilon: hệ số bảo mật
    """

    # First, sort the values
    z = x.copy() #making a working copy of the data. 
    z.sort()
    n = len(z)

    if n % 2 == 0:
        true_median = np.median(z)
        z.insert(n//2, true_median)
        n = n + 1
    
    for i in range(n//2):
        z[i] = max(lower_bound, z[i] - width)
        z[n-i-1] = min(z[n-i-1] + width, upper_bound)
    
    z.insert(0, lower_bound)
    z.append(upper_bound)
    n = n + 2

    
    maxNoisyScore = -np.inf
    argMaxNoisyScore = -1

    for i in range (1, n):
        if (z[i] - z[i - 1]) > 0:
            lgLength = np.log(np.abs(z[i] - z[i-1]))
        else:
            lgLength = -np.inf
            
        disFromMed = math.ceil(abs((i-1/2)-(n+1)/2))

        score = -(epsilon/2) * disFromMed + lgLength
        noisyScore = score + np.random.gumbel(loc=0.0, scale=1.0)

        if noisyScore > maxNoisyScore:
            maxNoisyScore = noisyScore
            argMaxNoisyScore = i

    left = z[argMaxNoisyScore - 1]
    right = z[argMaxNoisyScore]

    return np.random.uniform(low=left, high=right)
#################################################################




#################################################################


def smooth_sens_coef(n, x, beta, lower_bound, upper_bound, k):
    """
    n: số lượng phần tử trong mảng x
    x: mảng các phần tử
    beta: tham số làm mượt
    lower_bound: cận dưới của x
    upper_bound: cận trên của x
    """

    x = np.clip(x, lower_bound, upper_bound)
    x = np.concatenate(([lower_bound], x, [upper_bound]), axis = 0)
    x.sort()
    n = len(x) - 1
    m = n // 2 
    max_value = -np.inf


    j = min(m, n)
    i = max(m - k, 0)

    curr_value = (x[j] - x[i]) 

    # if curr_value > max_value:
    #     max_value = curr_value
    #     best_pair = [i, j]

    for o in range(1, n + 1):
        for p in range(o * k + k):
            j = min(m + p, n)
            i = max(m - (o * k + k) + p, 0)
            curr_value = (x[j] - x[i]) * np.exp(-beta * o)
            if curr_value > max_value:
                max_value = curr_value
    
    return max_value


def DPMedThielSen_Student(x, epsilon, beta, smooth_sens, d):
    Z = t.rvs(df = d)
    true_median = np.median(x)
    s = 2 * np.sqrt(d) * (epsilon - abs(beta) * (d+1)) / (d+1)
    
    return true_median + (1/s) * smooth_sens * Z 

#################################################################





    

    










