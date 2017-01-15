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
        return math.sqrt(variance(xs, parent))

    def standard_deviation_of_fd(fd, parent = False):
        return math.sqrt(variance_of_fd(fd, parent))

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

    return hist, crs

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

    return hist, crs

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

def test_mean_is_bigger(mu_zero, n, x_bar, x_var, alpha = 0.05):
    import math
    K = mu_zero + inverse_normal_cdf(1 - alpha) * math.sqrt(x_var / n)
    if x_bar <= K:
        print(x_bar, '<=', K, ": so fail to discard Ho")
        return False
    else:
        print(K, '<', x_bar, ": so discard Ho")
        return True

def test_mean_is_smaller(mu_zero, n, x_bar, x_var, alpha = 0.05):
    import math
    K = mu_zero + inverse_normal_cdf(alpha) * math.sqrt(x_var / n)
    if K <= x_bar:
        print(K , '<=', x_bar, ": so fail to discard Ho")
        return False
    else:
        print(x_bar, '<', K, ": so discard Ho")
        return True

def test_mean_is_not_same(mu_zero, n, x_bar, x_var, alpha = 0.05):
    import math
    KL = mu_zero + inverse_normal_cdf(alpha / 2) * math.sqrt(x_var / n)
    KU = mu_zero + inverse_normal_cdf(1 - alpha / 2) * math.sqrt(x_var / n)
    if KL <= x_bar <= KU:
        print(KL, '<=', x_bar, '<=', KU, ": so fail to discard Ho")
        return False
    elif x_bar < KL:
        print(x_bar, '<', KL, ": so discard Ho")
        return True
    else:
        print(KU, '<', x_bar, ": so discard Ho")
        return True

def test_mean_with_p_value(mu_zero, n, x_bar, x_var, alpha = 0.05):
    import math
    if x_bar > mu_zero:
        p = normal_probability_above((x_bar - mu_zero) / math.sqrt(x_var / n))
    else:
        p = normal_probability_below((x_bar - mu_zero) / math.sqrt(x_var / n))

    if alpha <= p:
        print(alpha, '<=', p, ': so failed to discard Ho')
        return False
    else:
        print(p, '<', alpha, ': so discard Ho')
        return True

def test_mean_is_bigger_with_list(mu_zero, xs, alpha = 0.05):
    import math
    n = len(xs)
    x_bar = mean(xs)
    x_var = variance_with_frequency(xs, parent = False)

    if n > 30:
        return test_mean_is_bigger(mu_zero, n, x_bar, x_var, alpha)
    else:
        K = mu_zero + inverse_t_cdf(1 - alpha, n - 1) * math.sqrt(x_var / n)
        if x_bar <= K:
            print('x_bar = ', x_bar, '<=', K, ": so fail to discard Ho")
            return False
        else:
            print(K, '<', x_bar, " = x_bar: so discard Ho")
            return True

def test_mean_is_smaller_with_list(mu_zero, xs, alpha = 0.05):
    import math
    n = len(xs)
    x_bar = mean(xs)
    x_var = variance_with_frequency(xs, parent = False)

    if n > 30:
        return test_mean_is_smaller(mu_zero, n, x_bar, x_var, alpha)
    else:
        K = mu_zero + inverse_t_cdf(alpha, n - 1) * math.sqrt(x_var / n)
        if K <= x_bar:
            print(K , '<=', x_bar, "= x_bar: so fail to discard Ho")
            return False
        else:
            print('x_bar =', x_bar, '<', K, ": so discard Ho")
            return True

def test_mean_is_not_same_with_list(mu_zero, xs, alpha = 0.05):
    import math
    n = len(xs)
    x_bar = mean(xs)
    x_var = variance_with_frequency(xs, parent = False)

    if n > 30:
        return test_mean_is_not_same(mu_zero, n, x_bar, x_var, alpha)
    else:
        KL = mu_zero + inverse_t_cdf(alpha / 2, n - 1) * math.sqrt(x_var / n)
        KU = mu_zero + inverse_t_cdf(1 - alpha / 2, n - 1) * math.sqrt(x_var / n)
        if KL <= x_bar <= KU:
            print(KL, '<=', x_bar, '= x_bar', '<=', KU, ": so fail to discard Ho")
            return False
        elif x_bar < KL:
            print('xbar =', x_bar, '<', KL, ": so discard Ho")
            return True
        else:
            print(KU, '<', x_bar, "= x_bar : so discard Ho")
            return True

def test_proportion_is_bigger(p_zero, n, p_cap, alpha = 0.05):
    import math
    K = p_zero + inverse_normal_cdf(1 - alpha) * math.sqrt(p_zero * (1 - p_zero) / n)
    if p_cap <= K:
        print(p_cap, '<=', K, ": so fail to discard Ho")
        return False
    else:
        print(K, '<', p_cap, ": so discard Ho")
        return True

def test_proportion_is_smaller(p_zero, n, p_cap, alpha = 0.05):
    import math
    K = p_zero + inverse_normal_cdf(alpha) * math.sqrt(p_zero * (1 - p_zero) / n)
    if K <= p_cap:
        print(K , '<=', p_cap, ": so fail to discard Ho")
        return False
    else:
        print(p_cap, '<', K, ": so discard Ho")
        return True

def test_proportion_is_not_same(p_zero, n, p_cap, alpha = 0.05):
    import math
    KL = p_zero + inverse_normal_cdf(alpha/2) * math.sqrt(p_zero * (1 - p_zero) / n)
    KU = p_zero + inverse_normal_cdf(1 - alpha / 2) * math.sqrt(p_zero * (1 - p_zero) / n)
    if KL <= p_cap <= KU:
        print(KL, '<=', p_cap, '<=', KU, ": so fail to discard Ho")
        return False
    elif p_cap < KL:
        print(p_cap, '<', KL, ": so discard Ho")
        return True
    else:
        print(KU, '<', p_cap, ": so discard Ho")
        return True

def estimate_two_sided_bounds_of_diff_between_two_menas(n1, mu1, var1, n2, mu2, var2, probability = 0.95, variances_should_be_same = False):
    import math

    sigma = math.sqrt(var1 / n1 + var2 / n2)
    tail_probability = (1 - probability) / 2

    if n1 > 30 and n2 > 30:
        return normal_two_sided_bounds(probability, mu1 - mu2, sigma)
    else:
        if variances_should_be_same:
            s = math.sqrt((((n1 - 1) * var1 + (n2 - 1) * var2) / ((n1 - 1) + (n2 - 1))) * (1/n1 + 1/n2))
            df = n1 + n2 - 2

            upper_bound = (mu1 - mu2) + inverse_t_cdf(1 - tail_probability, df) * s
            lower_bound = (mu1 - mu2) + inverse_t_cdf(tail_probability, df) * s

            return lower_bound, upper_bound
        else:
            df = int((var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1 - 1) + (var2/n2)**2/(n2 - 1)) + 0.5)

            upper_bound = (mu1 - mu2) + inverse_t_cdf(1 - tail_probability, df) * sigma
            lower_bound = (mu1 - mu2) + inverse_t_cdf(tail_probability, df) * sigma

            return lower_bound, upper_bound

def test_two_means_are_not_same(n1, mu1, var1, n2, mu2, var2, alpha = 0.05, variances_should_be_same = False):
    import math

    if n1 > 30 and n2 > 30:
        return test_mean_is_not_same(0, 1, mu1 - mu2, var1/n1 + var2/n2, alpha)
    else:
        if variances_should_be_same:
            sigma = math.sqrt((((n1 - 1) * var1 + (n2 - 1) * var2) / ((n1 - 1) + (n2 - 1))) * (1/n1 + 1/n2))
            df = n1 + n2 - 2

            KL = inverse_t_cdf(alpha / 2, df) * sigma
            KU = inverse_t_cdf(1 - alpha / 2, df) * sigma
            if KL <= mu1 - mu2 <= KU:
                print(KL, '<=', mu1 - mu2, '= mu1 - mu2', '<=', KU, ": so fail to discard Ho")
                return False
            elif mu1 - mu2 < KL:
                print('mu1 - mu2 =', mu1 - mu2, '<', KL, ": so discard Ho")
                return True
            else:
                print(KU, '<', mu1 - mu2, "= mu1 - mu2 : so discard Ho")
                return True
        else:
            sigma = math.sqrt(var1/n1 + var2/n2)
            df = int((var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1 - 1) + (var2/n2)**2/(n2 - 1)) + 0.5)

            KL = inverse_t_cdf(alpha / 2, df) * sigma
            KU = inverse_t_cdf(1 - alpha / 2, df) * sigma

            if KL <= mu1 - mu2 <= KU:
                print(KL, '<=', mu1 - mu2, '= mu1 - mu2', '<=', KU, ": so fail to discard Ho")
                return False
            elif mu1 - mu2 < KL:
                print('mu1 - mu2 =', mu1 - mu2, '<', KL, ": so discard Ho")
                return True
            else:
                print(KU, '<', mu1 - mu2, "= mu1 - mu2 : so discard Ho")
                return True

def estimate_two_sided_bounds_of_diff_between_two_proportions(n1, p1, n2, p2, probability = 0.95):
    import math

    sigma = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    tail_probability = (1 - probability) / 2

    if n1 > 30 and n2 > 30:
        sigma = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)

        lower_bound = (p1 - p2) + inverse_normal_cdf(tail_probability) * sigma
        upper_bound = (p1 - p2) + inverse_normal_cdf(1 - tail_probability) * sigma

        return lower_bound, upper_bound
    else:
        pass

def test_two_propotion_are_not_same(n1, p1, n2, p2, alpha = 0.05):
    import math

    if n1 > 30 and n2 > 30:
        p_cap = (n1 * p1 + n2 * p2) / (n1 + n2)
        sigma = math.sqrt(p_cap * (1 - p_cap) * (1 / n1 + 1 / n2))

        KL = inverse_normal_cdf(alpha/2) * sigma
        KU = inverse_normal_cdf(1 - alpha / 2) * sigma
        if KL <= p1 - p2 <= KU:
            print(KL, '<=', p1 - p2, '= p1 - p2', '<=', KU, ": so fail to discard Ho")
            return False
        elif p1 - p2 < KL:
            print('p1 - p2 =', p1 - p2, '<', KL, ": so discard Ho")
            return True
        else:
            print(KU, '<', p1 - p2, "= p1 - p2: so discard Ho")
            return True
    else:
        if variances_should_be_same:
            sigma = math.sqrt((((n1 - 1) * var1 + (n2 - 1) * var2) / ((n1 - 1) + (n2 - 1))) * (1/n1 + 1/n2))
            df = n1 + n2 - 2

            KL = inverse_t_cdf(alpha / 2, df) * sigma
            KU = inverse_t_cdf(1 - alpha / 2, df) * sigma
            if KL <= mu1 - mu2 <= KU:
                print(KL, '<=', mu1 - mu2, '= mu1 - mu2', '<=', KU, ": so fail to discard Ho")
                return False
            elif mu1 - mu2 < KL:
                print('mu1 - mu2 =', mu1 - mu2, '<', KL, ": so discard Ho")
                return True
            else:
                print(KU, '<', mu1 - mu2, "= mu1 - mu2 : so discard Ho")
                return True
        else:
            sigma = math.sqrt(var1/n1 + var2/n2)
            df = int((var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1 - 1) + (var2/n2)**2/(n2 - 1)) + 0.5)

            KL = inverse_t_cdf(alpha / 2, df) * sigma
            KU = inverse_t_cdf(1 - alpha / 2, df) * sigma

            if KL <= mu1 - mu2 <= KU:
                print(KL, '<=', mu1 - mu2, '= mu1 - mu2', '<=', KU, ": so fail to discard Ho")
                return False
            elif mu1 - mu2 < KL:
                print('mu1 - mu2 =', mu1 - mu2, '<', KL, ": so discard Ho")
                return True
            else:
                print(KU, '<', mu1 - mu2, "= mu1 - mu2 : so discard Ho")
                return True
