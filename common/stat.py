def product(iterable):
    prod = 1
    for n in iterable:
        prod *= n
    return prod

def npr(n, r):
    """
    Calculate the number of ordered permutations of r items taken from a
    population of size n.

    >>> npr(3, 2)
    6
    >>> npr(100, 20)
    1303995018204712451095685346159820800000
    """
    assert(0 <= r <= n)
    return product(range(n - r + 1, n + 1))

def ncr(n, r):
    from math import factorial
    """
    Calculate the number of unordered combinations of r items taken from a
    population of size n.

    >>> ncr(3, 2)
    3
    >>> ncr(100, 20)
    535983370403809682970
    >>> ncr(100000, 1000) == ncr(100000, 99000)
    True
    """
    assert (0 <= r <= n)
    if r > n // 2:
        r = n - r
    return npr(n, r) // factorial(r)

def converge(a, r):
    ''' a * r ** 0 + a * r ** 1 + a * r ** 2 ... '''
    assert(-1 < r < 1)
    return a / (1 - r)

def mean(xs, fs=None):
    '''xs is list of value or list of fd.
       fd is tuple(class, frequency).
       fs is list of frequency.
       if xs is list of fd, fs should be None.'''

    def mean_of_list(xs):
        # xs is list
        return sum(xs) / len(xs)

    def mean_of_xs_and_fs(xs,fs):
        # fd is list of tuple(class, frequence)
        sv = 0
        sf = 0
        for v, f in zip(xs, fs):
            sv = sv + v * f
            sf = sf + f
        return sv / sf

    if fs == None:
        if isinstance(xs[0], tuple):
            xs,fs = list(zip(*xs))
            return mean(xs,fs)
        else:
            return mean_of_list(xs)
    else:
        assert(len(xs) == len(fs))
        return mean_of_xs_and_fs(xs, fs)

def variance_with_frequency(xs, fs = None, parent = False):
    '''xs is list of value or list of fd.
       fd is tuple(class, frequency).
       fs is list of frequency.
       if xs is list of fd, fs should be None.'''

    def variance_with_frequency_inner(xs, fs, parent):
        m = mean(xs, fs)
        n = sum(fs)
        freedom = n if parent else n - 1

        sv = 0
        for x, f in zip(xs, fs):
            sv = sv + (x - m) ** 2 * f
        return sv / freedom;

    if fs == None:
        if isinstance(xs[0], tuple):
            xs, fs = list(zip(*xs))
            return variance_with_frequency_inner(xs, fs, parent)
        else:
            fs = [1 for _ in xs]
            return variance_with_frequency_inner(xs, fs, parent)
    else:
        assert(len(xs) == len(fs))
        return variance_with_frequency_inner(xs, fs, parent)

def weighted_mean(xs):
    '''xs is list of tuple(frequence of group, mean of group).'''
    # xs is list of tuple(frequence of group, mean of group)
    n = sum(map(lambda t: t[0], xs))
    return sum(map(lambda t: t[0]*t[1], xs)) / n

def median(xs):
    '''xs is list of value.'''
    n = len(xs)
    sx = sorted(xs)
    if n % 2 == 1: # 홀수
        return sx[(n + 1) // 2 - 1]
    else:
        lo = (n + 1) // 2 - 1
        hi = lo + 1
        return (sx[lo] + sx[hi]) / 2;

def standard_deviation(xs, parent = False):
    '''xs is list of value or list of fd.
       fd is tuple(class, frequence).'''
    import math
    def standard_deviation_of_list(xs, parent = False):
        return math.sqrt(variance_with_frequency(xs, parent=parent))

    def standard_deviation_of_fd(fd, parent = False):
        return math.sqrt(variance_of_fd(fd, parent=parent))

    if isinstance(xs[0], tuple):
        return standard_deviation_of_fd(xs, parent)
    else:
        return standard_deviation_of_list(xs, parent)

def variance(xs, ps):
    '''xs is list of value.
       ps is list of probability.'''
    assert(len(xs) != 0 and len(xs) == len(ps))

    m = mean(xs, ps)
    return mean([(x - m) ** 2 for x in xs], ps)

def covariance(xs, ys, xypm):
    '''
    '''
    assert(len(xs) != 0 and len(ys) != 0 and len(xypm) != 0)
    assert(len(xs) == len(xypm))
    assert(len(ys) == len(xypm[0]))

    xp = [sum(yps) for yps in xypm]
    yxps = [xp for col, _ in enumerate(xypm[0]) for yps in xypm for xp in [yps[col]]]
    yp = [sum(yxps[col * len(xypm):(col + 1) * len(xypm)]) for col, _ in enumerate(xypm[0])]

    xm = mean(xs, xp)
    ym = mean(ys, yp)
    dxs = [x - xm for x in xs]
    dys = [y - ym for y in ys]
    return sum([v
                for row, yps in enumerate(xypm)
                    for col, xyp in enumerate(yps)
                        for v in [xyp * dxs[row] * dys[col]]])

def correlation(xs, ys, xypm):
    '''
    '''

    import math

    cov = covariance(xs, ys, xypm)

    xps = [sum(yps) for yps in xypm]
    yxps = [xp for col, _ in enumerate(xypm[0]) for yps in xypm for xp in [yps[col]]]
    yps = [sum(yxps[col * len(xypm):(col + 1) * len(xypm)]) for col, _ in enumerate(xypm[0])]

    xv = variance(xs, xps)
    yv = variance(ys, yps)

    return cov / (math.sqrt(xv * yv))

def bernoulli_trial(p):
    import random
    return 1 if random.random() < p else 0

def binominal_pmf(p, n, y):
    return ncr(n, y) * p ** y * (1 - p) ** (n - y)

def binominal_cmf(p, n, y):
    return sum(binominal_pmf(p, n, x) for x in range(y + 1))

def binominal_trial(p, n):
    return sum(bernoulli_trial(p) for _ in range(n))

def geometric_pmf(p, n):
    return (1 - p) ** (n - 1) * p

def geometric_cmf(p, y):
    return sum(geometric_pmf(p, x) for x in range(1, y + 1))

def geometric_trial(p):
    count = 1
    while bernoulli_trial(p) == 0:
        count += 1
    return count

def poisson_pmf(rate, x, time = 1):
    import math
    return ((rate * time) ** x * math.e ** (-rate * time)) / math.factorial(x)

def poisson_cmf(rate, n, time = 1):
    return sum(poisson_pmf(rate, x, time) for x in range(n + 1))

def hypergeometric_pmf(g1, g2, n1, n2):
    return ncr(g1, n1) * ncr(g2, n2) / ncr(g1 + g2, n1 + n2)

def hypergeometric_trial(g1, g2, n):
    from random import randrange

    ball = [1 for _ in range(g1)] + [0 for _ in range(g2)]

    chosen = []
    for i in range(n):
        chosen.append(randrange(g1 + g2 - i))

    n1 = 0
    for c in chosen:
        n1 = n1 + ball[c]
        del ball[c]

    return n1

def make_hist(xs, class_ticks):
    # list of frequceny of class
    cfs = [0 for _ in range(len(class_ticks) - 1)]

    for x in xs:
        for ci, c in enumerate(class_ticks):
            if x >= c and x < class_ticks[ci + 1]:
                cfs[ci] = cfs[ci] + 1
                break

    # list of class range
    crs = []
    # shorted class range
    scr = class_ticks[1] - class_ticks[0]
    for ci, c in enumerate(class_ticks):
        if ci + 1 == len(class_ticks):
            break
        cr = class_ticks[ci + 1] - class_ticks[ci]
        if cr < scr:
            scr = cr
        crs.append(cr)

    # total frequency
    tf = sum(cfs)
    hist = []
    for fi, f in enumerate(cfs):
            hist.append(f / tf * (scr / crs[fi]))

    del crs[len(crs) - 1]
    return crs, hist

def make_hist_with_mono_class_range(xs, class_tick_count, class_bottom = None, class_top = None):
    if class_bottom == None:
        class_bottom = min(xs)
    if class_top == None:
        class_top = max(xs)

    # list of frequceny of class
    cfs = [0 for _ in range(class_tick_count)]

    # class range unit
    cru = (class_top - class_bottom) / class_tick_count
    crs = [class_bottom + i * cru for i in range(class_tick_count + 1)]

    last_cr_index = len(crs) - 1
    for x in xs:
        if x >= crs[last_cr_index]:
            cfs[last_cr_index - 1] = cfs[last_cr_index - 1] + 1
        else:
            for ci, _ in enumerate(crs):
                if crs[ci] <= x < crs[ci + 1]:
                    cfs[ci] = cfs[ci] + 1
                    break

    # total frequency
    tf = sum(cfs)
    hist = []
    for fi, f in enumerate(cfs):
            hist.append(f / tf)

    del crs[len(crs) - 1]
    return crs, hist 

def normal_pdf(x, mu=0, sigma=1):
    import math
    sqrt_two_pi = math.sqrt(2 * math.pi)
    return (math.exp(-(x-mu) ** 2 / 2 / sigma ** 2) / (sqrt_two_pi * sigma))

def normal_cdf(x, mu=0,sigma=1):
    import math
    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2

def inverse_normal_cdf(p, mu=0, sigma=1, tolerance=0.00001):
    """find approximate inverse using binary search"""

    # if not standard, compute standard and rescale
    if mu != 0 or sigma != 1:
        return mu + sigma * inverse_normal_cdf(p, tolerance=tolerance)

    low_z, low_p = -10.0, 0            # normal_cdf(-10) is (very close to) 0
    hi_z,  hi_p  =  10.0, 1            # normal_cdf(10)  is (very close to) 1
    while hi_z - low_z > tolerance:
        mid_z = (low_z + hi_z) / 2     # consider the midpoint
        mid_p = normal_cdf(mid_z)      # and the cdf's value there
        if mid_p < p:
            # midpoint is still too low, search above it
            low_z, low_p = mid_z, mid_p
        elif mid_p > p:
            # midpoint is still too high, search below it
            hi_z, hi_p = mid_z, mid_p
        else:
            break

    return mid_z

def exponential_pdf(x, rate=1.0):
    assert(x >= 0)
    import math
    return rate * math.exp(-rate * x)

def exponential_cdf(x, rate=1.0):
    assert(x >= 0)
    import math
    return (1 - math.exp(-rate * x))

def normal_trial(mu=0, sigma=1):
    import random
    return random.gauss(mu, sigma)

def t_pdf(x, df):
    from scipy.stats import t
    return t.pdf(x, df)

def t_cdf(x, df):
    from scipy.stats import t
    return t.cdf(x, df)

def inverse_t_cdf(p, df):
    from scipy.stats import t
    alpha = 0.0
    if p > 0.5:
        alpha = 1 - (1 - p) * 2
        _, hi = t.interval(alpha, df)
        return hi
    else:
        return -inverse_t_cdf(1 - p, df)

normal_probability_below = normal_cdf

def normal_probability_above(lo, mu=0, sigma=1):
    return 1 - normal_cdf(lo, mu, sigma)

def normal_probability_between(lo, hi, mu=0, sigma=1):
    return normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma)

def normal_probability_outside(lo, hi, mu=0, sigma=1):
    return 1 - normal_probability_between(lo, hi, mu, sigma)

normal_upper_bound = inverse_normal_cdf

def normal_lower_bound(probability, mu=0, sigma=1):
    return inverse_normal_cdf(1 - probability, mu, sigma)

def normal_two_sided_bounds(probability, mu=0, sigma=1):
    tail_probability = (1 - probability) / 2

    upper_bound = normal_lower_bound(tail_probability, mu, sigma)
    lower_bound = normal_upper_bound(tail_probability, mu, sigma)

    return lower_bound, upper_bound

t_probability_below = t_cdf

def t_probability_above(lo, df):
    return 1 - t_cdf(lo, df)

def t_probability_between(lo, hi, df):
    return t_cdf(hi, df) - t_cdf(lo, df)

def t_probability_outside(lo, hi, df):
    return 1 - t_probability_between(lo, hi, df)

t_upper_bound = inverse_t_cdf

def t_lower_bound(probability, df):
    return inverse_t_cdf(1 - probability, df)

def t_two_sided_bounds(probability, df):
    tail_probability = (1 - probability) / 2

    upper_bound = t_lower_bound(tail_probability, df)
    lower_bound = t_upper_bound(tail_probability, df)

    return lower_bound, upper_bound

NORMAL_SAMPLE_SIZE = 30

TEST_IS_BIGGER = 0
TEST_IS_SMALLER = 1
TEST_IS_NOT_SAME = 2

def estimate_mean(**kwargs):
    '''sigma0 : parent standard deviation (sigma)
       mu : sample mean (mu)
       sigma : sample standard deviation (sigma)
       n : sample size (n)
       alpha : level of significance'''

    import math

    sigma0 = kwargs.get("sigma0")
    mu = kwargs.get("mu")
    sigma = kwargs.get("sigma")
    n = kwargs.get("n")
    alpha = kwargs.get("alpha", 0.05)

    assert(n != None)
    assert(mu != None)
    assert(sigma0 != None or sigma != None)

    tail_probability = alpha / 2

    if n > NORMAL_SAMPLE_SIZE or sigma0 != None:
        sigma = (sigma0 or sigma) / math.sqrt(n)

        lower_bound = mu + inverse_normal_cdf(tail_probability) * sigma
        upper_bound = mu + inverse_normal_cdf(1 - tail_probability) * sigma

        return lower_bound, upper_bound
    else:
        sigma = sigma / math.sqrt(n)

        lower_bound = mu + inverse_t_cdf(tail_probability, n - 1) * sigma
        upper_bound = mu + inverse_t_cdf(1 - tail_probability, n - 1) * sigma

        return lower_bound, upper_bound

def sample_size_to_estimate_mean_with_tolerance(**kwargs):
    ''' tolerance :
        sigma :
        alpha : '''

    tolerance = kwargs.get("tolerance")
    sigma = kwargs.get("sigma")
    alpha = kwargs.get("alpha", 0.05)

    assert(tolerance != None)
    assert(sigma != None)

    return (inverse_normal_cdf(1 - alpha / 2) * sigma / tolerance) ** 2

def sample_size_to_estimate_mean_with_error(**kwargs):
    '''mu0 :
       mua :
       sigma :
       alpha :
       beta : '''

    import math

    mu0 = kwargs.get("mu0")
    mua = kwargs.get("mua")
    sigma = kwargs.get("sigma")
    alpha = kwargs.get("alpha", 0.05)
    beta = kwargs.get("beta", 0.1)

    assert(mu0 != None)
    assert(mua != None)
    assert(sigma != None)

    return ((inverse_normal_cdf(1 - alpha) + inverse_normal_cdf(1 - beta)) / (mua - mu0) * sigma) ** 2

def test_mean(**kwargs):
    '''mu0 :
       sigma0 : parent standard_deviation (sigma)
       mu : sample mean (mu)
       sigma : sample standard deviation (sigma)
       n : sample size (n)
       alpha : level of significance
       test : one of TEST_IS_BIGGER, TEST_IS_SMALLER, TEST_IS_NOT_SAME'''

    import math

    mu0 = kwargs.get("mu0")
    sigma0 = kwargs.get("sigma0")
    mu = kwargs.get("mu")
    sigma = kwargs.get("sigma")
    n = kwargs.get("n")
    alpha = kwargs.get("alpha", 0.05)
    test = kwargs.get("test", TEST_IS_NOT_SAME)

    assert(mu0 != None)
    assert(n != None)
    assert(mu != None)
    assert(sigma0 != None or sigma != None)
    assert(test == TEST_IS_BIGGER or test == TEST_IS_SMALLER or TEST_IS_NOT_SAME)

    if test == TEST_IS_NOT_SAME:
        tail_probability = alpha / 2

        if n > NORMAL_SAMPLE_SIZE or sigma0 != None:
            sigma = (sigma0 or sigma) / math.sqrt(n)

            KL = mu0 + inverse_normal_cdf(tail_probability) * sigma
            KU = mu0 + inverse_normal_cdf(1 - tail_probability) * sigma
        else:
            sigma = sigma / math.sqrt(n)

            KL = mu0 + inverse_t_cdf(tail_probability, n - 1) * sigma
            KU = mu0 + inverse_t_cdf(1 - tail_probability, n - 1) * sigma

        print('KL = ', KL, ', KU =', KU)
        if KL <= mu <= KU:
            return False
        else:
            return True

    elif test == TEST_IS_BIGGER:
        if n > NORMAL_SAMPLE_SIZE or sigma0 != None:
            sigma = (sigma0 or sigma) / math.sqrt(n)

            KU = mu0 + inverse_normal_cdf(1 - alpha) * sigma
        else:
            sigma = sigma / math.sqrt(n)

            KU = mu0 + inverse_t_cdf(1 - alpha, n - 1) * sigma

        print('KU = ', KU)
        if mu <= KU:
            return False
        else:
            return True

    elif test == TEST_IS_SMALLER:
        if n > NORMAL_SAMPLE_SIZE or sigma0 != None:
            sigma = (sigma0 or sigma) / math.sqrt(n)

            KL = mu0 + inverse_normal_cdf(alpha) * sigma
        else:
            sigma = sigma / math.sqrt(n)

            KL = mu0 + inverse_t_cdf(alpha, n - 1) * sigma

        print('KL = ', KL)
        if KL <= mu:
            return False
        else:
            return True

def estimate_proportion(**kwargs):
    '''p : sample proportion (p)
       n : sample size (n)
       alpha : level of significance'''

    import math

    p = kwargs.get("p")
    n = kwargs.get("n")
    alpha = kwargs.get("alpha", 0.05)

    assert(p != None)
    assert(n != None)
    assert(n >= NORMAL_SAMPLE_SIZE)
    assert(n * p >= 5)
    assert(n * (1 - p) >= 5)

    tail_probability = alpha / 2
    sigma = math.sqrt(p * (1 - p) / n)

    lower_bound = p + inverse_normal_cdf(tail_probability) * sigma
    upper_bound = p + inverse_normal_cdf(1 - tail_probability) * sigma

    return lower_bound, upper_bound

def sample_size_to_estimate_proportion_with_tolerance(**kwargs):
    ''' tolerance :
        p :
        alpha : '''

    tolerance = kwargs.get("tolerance")
    p = kwargs.get("p")
    alpha = kwargs.get("alpha", 0.05)

    assert(tolerance != None)
    assert(p != None)

    return (inverse_normal_cdf(1 - alpha / 2) / tolerance) ** 2 * p * (1 - p)

def sample_size_to_estimate_proportion_with_error(**kwargs):
    ''' p0 :
        alpha :
        pa :
        beta : '''

    import math

    p0 = kwargs.get("p0")
    alpha = kwargs.get("alpha", 0.05)
    pa = kwargs.get("pa")
    beta = kwargs.get("beta", 0.1)

    assert(p0 != None)
    assert(pa != None)

    return ((inverse_normal_cdf(1 - alpha) * math.sqrt(p0*(1-p0)) +
                inverse_normal_cdf(1 - beta) * math.sqrt(pa*(1-pa))) /
            (pa - p0)) ** 2

def test_proportion(**kwargs):
    '''p0 :
       p : sample proportion (p)
       n : sample size (n)
       alpha : level of significance
       test : one of TEST_IS_BIGGER, TEST_IS_SMALLER, TEST_IS_NOT_SAME'''

    import math

    p0 = kwargs.get("p0")
    p = kwargs.get("p")
    n = kwargs.get("n")
    alpha = kwargs.get("alpha", 0.05)
    test = kwargs.get("test", TEST_IS_NOT_SAME)

    assert(p0 != None)
    assert(p != None)
    assert(n != None)
    assert(n >= NORMAL_SAMPLE_SIZE)
    assert(n * p >= 5)
    assert(n * (1 - p) >= 5)
    assert(test == TEST_IS_BIGGER or test == TEST_IS_SMALLER or TEST_IS_NOT_SAME)

    sigma = math.sqrt(p0 * (1 - p0) / n)

    if test == TEST_IS_NOT_SAME:
        tail_probability = alpha / 2

        KL = p0 + inverse_normal_cdf(tail_probability) * sigma
        KU = p0 + inverse_normal_cdf(1 - tail_probability) * sigma

        print('KL=', KL, ', KU=', KU)
        if KL <= p <= KU:
            return False
        else:
            return True

    elif test == TEST_IS_BIGGER:
        tail_probability = alpha

        KU = p0 + inverse_normal_cdf(1 - tail_probability) * sigma

        print('KU=', KU)
        if p <= KU:
            return False
        else:
            return True
    elif test == TEST_IS_SMALLER:
        tail_probability = alpha

        KL = p0 + inverse_normal_cdf(tail_probability) * sigma

        print('KL=', KL)
        if KL <= p:
            return False
        else:
            return True

def estimate_diff_between_means(**kwargs):
    '''mu1 : sample 1 mean (mu)
       sigma1 : sample 1 standard deviation (sigma)
       n1 : sample 1 size
       mu2 : sample 1 mean (mu)
       sigma2 : sample 1 standard deviation (sigma)
       n2 : sample 1 size
       are_parent_sigmas_same :
       alpha : level of significance'''

    import math

    mu1 = kwargs.get("mu1")
    sigma1 = kwargs.get("sigma1")
    n1 = kwargs.get("n1")
    mu2 = kwargs.get("mu2")
    sigma2 = kwargs.get("sigma2")
    n2 = kwargs.get("n2")
    are_parent_sigmas_same = kwargs.get("are_parent_sigmas_same", True)
    alpha = kwargs.get("alpha", 0.05)

    assert(mu1 != None)
    assert(sigma1 != None)
    assert(n1 != None)
    assert(mu2 != None)
    assert(sigma2 != None)
    assert(n2 != None)

    diff = mu1 - mu2
    tail_probability = alpha / 2
    sn1 = sigma1 ** 2 / n1
    sn2 = sigma2 ** 2 / n2

    if n1 >= NORMAL_SAMPLE_SIZE and n2 >= NORMAL_SAMPLE_SIZE:
        sigma = math.sqrt(sn1 + sn2)

        lower_bound = diff + inverse_normal_cdf(tail_probability) * sigma
        upper_bound = diff + inverse_normal_cdf(1 - tail_probability) * sigma

        return lower_bound, upper_bound
    else:
        if are_parent_sigmas_same:
            df = n1 + n2 - 2
            sp2 = ((n1 - 1) * sigma1 ** 2 + (n2 - 1) * sigma2 ** 2) / df
            sigma = math.sqrt(sp2 * (1 / n1 + 1 / n2))
        else:
            sigma = math.sqrt(sn1 + sn2)
            df = int(((sn1 + sn2) ** 2 / (sn1 ** 2 / (n1 - 1) + sn2 ** 2 / (n2 - 1))) + 0.5)

        lower_bound = diff + inverse_t_cdf(tail_probability, df) * sigma
        upper_bound = diff + inverse_t_cdf(1 - tail_probability, df) * sigma

        return lower_bound, upper_bound

def test_diff_between_means(**kwargs):
    '''mu1 : sample 1 mean (mu)
       sigma1 : sample 1 standard deviation (sigma)
       n1 : sample 1 size
       mu2 : sample 1 mean (mu)
       sigma2 : sample 1 standard deviation (sigma)
       n2 : sample 1 size
       are_parent_sigmas_same :
       alpha : level of significance
       test : one of TEST_IS_BIGGER, TEST_IS_SMALLER, TEST_IS_NOT_SAME'''

    import math

    mu1 = kwargs.get("mu1")
    sigma1 = kwargs.get("sigma1")
    n1 = kwargs.get("n1")
    mu2 = kwargs.get("mu2")
    sigma2 = kwargs.get("sigma2")
    n2 = kwargs.get("n2")
    are_parent_sigmas_same = kwargs.get("are_parent_sigmas_same", True)
    alpha = kwargs.get("alpha", 0.05)
    test = kwargs.get("test", TEST_IS_NOT_SAME)

    assert(mu1 != None)
    assert(sigma1 != None)
    assert(n1 != None)
    assert(mu2 != None)
    assert(sigma2 != None)
    assert(n2 != None)
    assert(test == TEST_IS_BIGGER or test == TEST_IS_SMALLER or TEST_IS_NOT_SAME)

    diff = mu1 - mu2
    sn1 = sigma1 ** 2 / n1
    sn2 = sigma2 ** 2 / n2

    if test == TEST_IS_NOT_SAME:
        tail_probability = alpha / 2
        if n1 >= NORMAL_SAMPLE_SIZE and n2 >= NORMAL_SAMPLE_SIZE:
            sigma = math.sqrt(sn1 + sn2)

            KL = inverse_normal_cdf(tail_probability) * sigma
            KU = inverse_normal_cdf(1 - tail_probability) * sigma

        else:
            if are_parent_sigmas_same:
                df = n1 + n2 - 2
                sp2 = ((n1 - 1) * sigma1 ** 2 + (n2 - 1) * sigma2 ** 2) / df
                sigma = math.sqrt(sp2 * (1 / n1 + 1 / n2))
            else:
                sigma = math.sqrt(sn1 + sn2)
                df = int(((sn1 + sn2) ** 2 / (sn1 ** 2 / (n1 - 1) + sn2 ** 2 / (n2 - 1))) + 0.5)

            KL = inverse_t_cdf(tail_probability, df) * sigma
            KU = inverse_t_cdf(1 - tail_probability, df) * sigma

        print('KL=', KL, ', KU=', KU)
        if KL <= diff <= KU:
            return False
        else:
            return True

    elif test == TEST_IS_BIGGER:
        tail_probability = alpha
        if n1 >= NORMAL_SAMPLE_SIZE and n2 >= NORMAL_SAMPLE_SIZE:
            sigma = math.sqrt(sn1 + sn2)

            KU = inverse_normal_cdf(1 - tail_probability) * sigma

        else:
            if are_parent_sigmas_same:
                df = n1 + n2 - 2
                sp2 = ((n1 - 1) * sigma1 ** 2 + (n2 - 1) * sigma2 ** 2) / df
                sigma = math.sqrt(sp2 * (1 / n1 + 1 / n2))
            else:
                sigma = math.sqrt(sn1 + sn2)
                df = int(((sn1 + sn2) ** 2 / (sn1 ** 2 / (n1 - 1) + sn2 ** 2 / (n2 - 1))) + 0.5)

            KU = inverse_t_cdf(1 - tail_probability, df) * sigma

        print('KU=', KU)
        if diff <= KU:
            return False
        else:
            return True

    elif test == TEST_IS_SMALLER:
        tail_probability = alpha
        if n1 >= NORMAL_SAMPLE_SIZE and n2 >= NORMAL_SAMPLE_SIZE:
            sigma = math.sqrt(sn1 + sn2)

            KL = inverse_t_cdf(tail_probability, df) * sigma

        else:
            if are_parent_sigmas_same:
                df = n1 + n2 - 2
                sp2 = ((n1 - 1) * sigma1 ** 2 + (n2 - 1) * sigma2 ** 2) / df
                sigma = math.sqrt(sp2 * (1 / n1 + 1 / n2))
            else:
                sigma = math.sqrt(sn1 + sn2)
                df = int(((sn1 + sn2) ** 2 / (sn1 ** 2 / (n1 - 1) + sn2 ** 2 / (n2 - 1))) + 0.5)

            KL = inverse_t_cdf(tail_probability, df) * sigma

        print('KL=', KL)
        if KL <= diff:
            return False
        else:
            return True

def estimate_diff_between_proportions(**kwargs):
    '''p1 : sample 1 proportion (p)
       n1 : sample 1 size
       p2 : sample 2 proportion (p)
       n2 : sample 2 size
       alpha : level of significance'''

    import math

    p1 = kwargs.get("p1")
    n1 = kwargs.get("n1")
    p2 = kwargs.get("p2")
    n2 = kwargs.get("n2")
    alpha = kwargs.get("alpha", 0.05)

    assert(p1 != None)
    assert(n1 != None)
    assert(p2 != None)
    assert(n2 != None)
    assert(n1 >= NORMAL_SAMPLE_SIZE and n2 >= NORMAL_SAMPLE_SIZE)
    assert(n1 * p1 >= 5)
    assert(n1 * (1 - p1) >= 5)
    assert(n2 * p2 >= 5)
    assert(n2 * (1 - p2) >= 5)

    diff = p1 - p2
    tail_probability = alpha / 2
    sigma = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)

    lower_bound = diff + inverse_normal_cdf(tail_probability) * sigma
    upper_bound = diff + inverse_normal_cdf(1 - tail_probability) * sigma

    return lower_bound, upper_bound


def test_diff_between_proportions(**kwargs):
    '''p1 : sample 1 proportion (p)
       n1 : sample 1 size
       p2 : sample 2 proportion (p)
       n2 : sample 2 size
       alpha : level of significance
       test : one of TEST_IS_BIGGER, TEST_IS_SMALLER, TEST_IS_NOT_SAME'''

    import math

    p1 = kwargs.get("p1")
    n1 = kwargs.get("n1")
    p2 = kwargs.get("p2")
    n2 = kwargs.get("n2")
    alpha = kwargs.get("alpha", 0.05)
    test = kwargs.get("test", TEST_IS_NOT_SAME)

    assert(p1 != None)
    assert(n1 != None)
    assert(p2 != None)
    assert(n2 != None)
    assert(n1 >= NORMAL_SAMPLE_SIZE and n2 >= NORMAL_SAMPLE_SIZE)
    assert(n1 * p1 >= 5)
    assert(n1 * (1 - p1) >= 5)
    assert(n2 * p2 >= 5)
    assert(n2 * (1 - p2) >= 5)
    assert(test == TEST_IS_BIGGER or test == TEST_IS_SMALLER or TEST_IS_NOT_SAME)

    diff = p1 - p2

    p = (n1 * p1 + n2 * p2) / (n1 + n2)
    sigma = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))

    if test == TEST_IS_NOT_SAME:
        tail_probability = alpha / 2

        KL = inverse_normal_cdf(tail_probability) * sigma
        KU = inverse_normal_cdf(1 - tail_probability) * sigma

        print('KL=', KL, ', KU=', KU)
        if KL <= diff <= KU:
            return False
        else:
            return True

    elif test == TEST_IS_BIGGER:
        tail_probability = alpha

        KU = inverse_normal_cdf(1 - tail_probability) * sigma

        print('KU=', KU)
        if diff <= KU:
            return False
        else:
            return True
    elif test == TEST_IS_SMALLER:
        tail_probability = alpha

        KL = inverse_normal_cdf(tail_probability) * sigma

        print('KL=', KL)
        if KL <= diff:
            return False
        else:
            return True

def estimate_diff_between_proportions_of_paired(**kwargs):
    '''d_bar : mean of diff (d bar)
       sd : standard deviation of diff (sd)
       n : sample size
       alpha : level of significance'''

    import math

    d_bar = kwargs.get("d_bar")
    sd = kwargs.get("sd")
    n = kwargs.get("n")
    alpha = kwargs.get("alpha", 0.05)

    return estimate_mean(mu=d_bar, sigma=sd,n=n,alpha=alpha)

def test_diff_between_proportions_of_paired(**kwargs):
    '''d_bar : mean of diff (d bar)
       sd : standard deviation of diff (sd)
       n : sample size
       alpha : level of significance
       test : one of TEST_IS_BIGGER, TEST_IS_SMALLER, TEST_IS_NOT_SAME'''

    d_bar = kwargs.get("d_bar")
    sd = kwargs.get("sd")
    n = kwargs.get("n")
    alpha = kwargs.get("alpha", 0.05)
    test = kwargs.get("test", TEST_IS_NOT_SAME)

    return test_mean(mu0=0, mu=d_bar, sigma=sd,n=n,alpha=alpha,test=test)

def estimate_sigma(**kwargs):
    '''sigma : sample standard deviation (sigma)
       n : sample size (n)
       alpha : level of significance'''

    sigma = kwargs.get("sigma")
    n = kwargs.get("n")
    alpha = kwargs.get("alpha", 0.05)

    assert(sigma != None)
    assert(n != None)
    assert(n >= NORMAL_SAMPLE_SIZE)

    from scipy.stats import chi2

    probability = 1 - alpha
    chi2_lower, chi2_upper = chi2.interval(probability, n - 1)

    lower_bound = (n - 1) * sigma ** 2 / chi2_upper
    upper_bound = (n - 1) * sigma ** 2 / chi2_lower

    return math.sqrt(lower_bound), math.sqrt(upper_bound)

def test_sigma(**kwargs):
    '''sigma0 :
       sigma : sample standard deviation (sigma)
       n : sample_size (n)
       alpha : level of significance
       test : one of TEST_IS_BIGGER, TEST_IS_SMALLER, TEST_IS_NOT_SAME'''

    sigma0 = kwargs.get("sigma0")
    sigma = kwargs.get("sigma")
    n = kwargs.get("n")
    alpha = kwargs.get("alpha", 0.05)
    test = kwargs.get("test", TEST_IS_NOT_SAME)

    assert(sigma0 != None)
    assert(sigma != None)
    assert(n != None)
    # assert(n >= NORMAL_SAMPLE_SIZE)
    assert(test == TEST_IS_BIGGER or test == TEST_IS_SMALLER or TEST_IS_NOT_SAME)

    from scipy.stats import chi2

    theta = (n - 1) * sigma ** 2 / sigma0 ** 2
    print('theta =', theta)

    if test == TEST_IS_NOT_SAME:
        probability = 1 - alpha

        KL, KU = chi2.interval(probability, n - 1)

        print('KL =', KL, ', KU =', KU)
        if KL <= theta <= KU:
            return False
        else:
            return True

    elif test == TEST_IS_BIGGER:
        probability = 1 - alpha * 2

        _, KU = chi2.interval(probability, n - 1)

        print('KU =', KU)
        if theta <= KU:
            return False
        else:
            return True

    elif test == TEST_IS_SMALLER:
        probability = 1 - alpha * 2

        KL, _ = chi2.interval(probability, n - 1)

        print('KL =', KL)
        if KL <= theta:
            return False
        else:
            return True

def estimate_ratio_of_sigmas(**kwargs):
    '''sigma1 : sample standard deviation (sigma)
       n1 : sample size (n)
       sigma2 : sample standard deviation (sigma)
       n2 : sample size (n)
       alpha : level of significance'''

    sigma1 = kwargs.get("sigma1")
    n1 = kwargs.get("n1")
    sigma2 = kwargs.get("sigma2")
    n2 = kwargs.get("n2")
    alpha = kwargs.get("alpha", 0.05)

    assert(sigma1 != None)
    assert(n1 != None)
    # assert(n1 >= NORMAL_SAMPLE_SIZE)
    assert(sigma2 != None)
    assert(n2 != None)
    # assert(n2 >= NORMAL_SAMPLE_SIZE)

    import math
    from scipy.stats import f

    f_lower, f_upper = f.interval(1 - alpha, n2 - 1, n1 - 1)
    sigma_ratio = sigma1 ** 2 / sigma2 ** 2

    lower_bound = f_lower * sigma_ratio
    upper_bound = f_upper * sigma_ratio

    return math.sqrt(lower_bound), math.sqrt(upper_bound)

def test_ratio_of_sigmas(**kwargs):
    '''sigma1 : sample standard deviation (sigma)
       n1 : sample size (n)
       sigma2 : sample standard deviation (sigma)
       n2 : sample size (n)
       alpha : level of significance
       test : one of TEST_IS_BIGGER, TEST_IS_SMALLER, TEST_IS_NOT_SAME'''

    sigma1 = kwargs.get("sigma1")
    n1 = kwargs.get("n1")
    sigma2 = kwargs.get("sigma2")
    n2 = kwargs.get("n2")
    alpha = kwargs.get("alpha", 0.05)
    test = kwargs.get("test", TEST_IS_NOT_SAME)

    assert(sigma1 != None)
    assert(n1 != None)
    # assert(n1 >= NORMAL_SAMPLE_SIZE)
    assert(sigma2 != None)
    assert(n2 != None)
    # assert(n2 >= NORMAL_SAMPLE_SIZE)
    assert(test == TEST_IS_BIGGER or test == TEST_IS_SMALLER or TEST_IS_NOT_SAME)

    theta = sigma1 ** 2 / sigma2 ** 2
    print("theta =", theta)

    from scipy.stats import f

    if test == TEST_IS_NOT_SAME:
        probability = 1 - alpha

        KL, KU = f.interval(probability, n1 - 1, n2 - 1)

        print("KL =", KL, ", KU =", KU)
        if KL <= theta <= KU:
            return False
        else:
            return True

    elif test == TEST_IS_BIGGER:
        probability = 1 - alpha * 2

        _, KU = f.interval(probability, n1 - 1, n2 - 1)

        print("KU =", KU)
        if theta <= KU:
            return False
        else:
            return True

    else:
        return test_ratio_of_sigmas(n1=n2,sigma1=sigma2,n2=n1,sigma2=sigma1,alpha=alpha,test=TEST_IS_BIGGER)

def simple_regression_equation_parameters_with_list(xs, ys):
    ''' yi = beta0 + beta1 * xi
    '''
    assert(len(xs) == len(ys))

    mux = mean(xs)
    muy = mean(ys)

    cov = sum((x - mux) * (ys[i] - muy) for i, x in enumerate(xs))
    x_var = sum((x - mux) ** 2 for x in xs)

    beta1 = cov / x_var
    beta0 = muy - beta1 * mux

    return beta0, beta1

def correlation_with_list(xs, ys):
    ''' -1 <= gamma <=1
    '''

    assert(len(xs) == len(ys))

    import math

    mux = mean(xs)
    muy = mean(ys)

    cov = sum((xs[i] - mux) * (y - muy) for i, y in enumerate(ys))
    sigma2x = sum((x - mux) ** 2 for x in xs)
    sigma2y = sum((y - muy) ** 2 for y in ys)

    return cov / math.sqrt(sigma2x * sigma2y)

def coefficient_with_list(xs, ys):
    ''' 0 <= gamma ** 2 <=1
    '''

    assert(len(xs) == len(ys))

    # beta0, beta1 = simple_regression_equation_parameters(xs, ys)

    # mux = mean(xs)
    # muy = mean(ys)

    # sse = sum((y - (beta0 + beta1 * xs[i])) ** 2 for i, y in enumerate(ys))
    # ssto = sum((y - muy) ** 2 for y in ys)

    # return 1 - sse / ssto

    return correlation_with_list(xs, ys) ** 2

def test_not_same_by_one_factor_anova_with_list(ls, alpha=0.05):
    from scipy.stats import f

    total_sum = 0
    total_count = 0
    SSE = 0
    for i, s in enumerate(ls):
        total_sum = total_sum + sum(s)
        total_count = total_count + len(s)
        yi_bar = sum(s) / len(s)
        for v in s:
            SSE += (v - yi_bar) ** 2
    y_bar = total_sum / total_count
    # print('SSE=', SSE)
    # print('y_bar=', y_bar)

    TSS = 0
    SST = 0
    for i, s in enumerate(ls):
        yi_bar = sum(s) / len(s)
        # print('yi_bar[',i,']=',yi_bar)
        for v in s:
            TSS += (v - y_bar) ** 2
            SST += (yi_bar - y_bar) ** 2

    # print('TSS=', TSS)
    # print('SST=', SST)

    nu1 = len(ls) - 1
    nu2 = total_count - len(ls)
    # print('nu1=',nu1)
    # print('nu2=',nu2)

    MST = SST / nu1
    # print('MST=', MST)
    MSE = SSE / nu2
    # print('MSE=', MSE)
    F = MST / MSE
    print('F =', F)

    probability = 1 - alpha * 2
    _, K = f.interval(probability, nu1, nu2)

    print('K =', K)
    if F <= K:
        return False
    else:
        return True

def test_not_same_by_chi2(ps, n, alpha=0.05, p0s=None):
    from scipy.stats import chi2

    if p0s == None:
        p0s = [1/len(ps) for _, _ in enumerate(ps)]

    X2 = sum((n*p - n*p0s[i]) ** 2/(n*p0s[i]) for i, p in enumerate(ps))
    print('X2 = ', X2)

    probability = 1 - alpha * 2
    nu = len(ps) - 1
    L, K = chi2.interval(probability, nu)
    print('K =', K)

    if X2 <= K:
        return False
    else:
        return True

def even_multinomial_trial(k, n):
    import random
    x = [0 for _ in range(k)]
    for _ in range(n):
        x[random.randrange(k)] += 1
    return x
