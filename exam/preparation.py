# -*- coding: utf-8 -*-

# -- uvod --

from collections import Counter
import math

from IPython.display import Image
import numpy as np
from numpy.linalg import inv
import scipy.linalg as la
import scipy.stats as st

from fractions import Fraction

matice = np.matrix([
    [1/8, 1/16, 1/32, 1/32],
    [1/16, 1/8, 1/32, 1/32],
    [1/16, 1/16, 1/16, 1/16],
    [1/4, 0, 0, 0],
])

def vystup_zlomky(matrix):
    for i in range(matrix.shape[0]):
        row = ""
        for j in range(matrix.shape[1]):
            row += str(Fraction(matrix[i, j]).limit_denominator())
            row += "\t"
        print(row)

print(matice)
print("Hezky:")
vystup_zlomky(matice)

# # Střední hodnota náhodného vektoru $Z = (X, Y)^T$
# 
# https://marast.fit.cvut.cz/cs/problems/10185
# 
# TODO: vzorce


# x\y
matice = np.matrix([
    [0, 1/3, 0],
    [1/8, 1/4, 1/8],
    [0, 1/6, 0],
])
# sloupce
p_y = np.array([-1, 0, 1])
# řádky
p_x = np.array([0, 1, 2])

p_Yy = np.squeeze(np.asarray(np.sum(matice, axis = 0)))
p_Xx = np.squeeze(np.asarray(np.sum(matice, axis = 1)))

print("EX =", Fraction(np.sum(p_Xx*p_x)).limit_denominator())
print("EY =", Fraction(np.sum(p_Yy*p_y)).limit_denominator())
# TODO: EXY

# # Varianční matice náhodného vektoru $Z = (X, Y)^T$
# 
# https://marast.fit.cvut.cz/cs/problems/10186
# 
# $$\textbf{var Z} = 
# \begin{pmatrix}
# \textbf{var X} & \textbf{cov(X,Y)} \\
# \textbf{cov(Y,X)} & \textbf{var Y}
# \end{pmatrix}
# $$
# 
# $varX = EX^2−(EX)^2$ 
# 
# $cov(X,Y)=cov(Y,X)=EXY−EXEY$
# 
# na diagonále rozptyly, jinde kovariance
# 
# TODO: vzorce



# x\y
matice = np.matrix([
    [0, 1/3, 0],
    [1/8, 1/4, 1/8],
    [0, 1/6, 0],
])
# sloupce
p_y = np.array([-1, 0, 1])
# řádky
p_x = np.array([0, 1, 2])

p_Yy = np.squeeze(np.asarray(np.sum(matice, axis = 0)))
p_Xx = np.squeeze(np.asarray(np.sum(matice, axis = 1)))

ex = np.sum(p_Xx*p_x)
ex2 = np.sum(p_Xx*(p_x**2))
ey = np.sum(p_Yy*p_y)
ey2 = np.sum(p_Yy*(p_y**2))
varX = ex2 - ex**2
varY = ey2 - ey**2 

# TODO: EXY
exy = 0
covXY = exy - ex*ey




# -- entropie --

# # Entropie
# 
# Pokud dáváte matici, musí být prvky $p(x,y)$, ne $p(x|y)$.
# 
# ## Entropie
# 
# https://marast.fit.cvut.cz/cs/problems/10194
# 
# $$H(X)=-\sum_{x \in \mathcal{X}} p(x) \log p(x)$$


# DATA

# TODO: NASTAVIT
matice = np.matrix([
    [1/4, 1/8, 1/2, 1/8]
])
# x\y
matice = np.matrix([
    [1/8, 1/16, 1/32, 1/32],
    [1/16, 1/8, 1/32, 1/32],
    [1/16, 1/16, 1/16, 1/16],
    [1/4, 0, 0, 0],
])
matice = np.matrix([
    [1/16, 1/16, 1/8],
    [1/8, 1/16, 1/16],
    [1/8, 1/4, 1/8],
])
osa = 1 # 0 = H(Y) ... sečíst sloupce, 1 = H(X) ... sečíst řádky

# PROGRAM
def entropie(matrix, axis):
    p = np.squeeze(np.asarray(np.sum(matrix, axis = axis)))
    result = 0
    for i in range(p.size):
        if p[i] == 0:
            continue
        result += p[i] * math.log(p[i], 2)
    return -result

_res = entropie(matice, osa)
print("H(X) =", Fraction(_res).limit_denominator(), "=", _res)

# ## Sdružená entropie
# 
# https://marast.fit.cvut.cz/cs/problems/10194
# 
# Sdružená entropie $H(X, Y)$ diskrétních náhodných veličin $X, Y$ se sdruženým
# rozdělením $p(x, y)$ je definována jako
# 
# $$H(Y, X) = - \sum_{x \in \mathcal{X}} \sum_{y \in \mathcal{Y}} p(x, y) \log p(y, x)$$


# DATA

# TODO: NASTAVIT
# x\y
matice = np.matrix([
    [1/8, 1/16, 1/32, 1/32],
    [1/16, 1/8, 1/32, 1/32],
    [1/16, 1/16, 1/16, 1/16],
    [1/4, 0, 0, 0],
])

# PROGRAM
def sdruzena_entropie(matrix):
    result = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] == 0:
                continue
            result += matrix[i, j] * np.log2(matrix[i, j])
    
    return -result

_res = sdruzena_entropie(matice)
print("H(X, Y) =", Fraction(_res).limit_denominator(), "=", _res)

# ## Podmíněná entropie
# 
# https://marast.fit.cvut.cz/cs/problems/10195
# 
# Podmíněná entropie $H(Y|X)$ diskrétních náhodných veličin $X, Y$ se sdruženým
# rozdělením $p(x, y)$ je definována jako
# 
# $$H(Y | X) = - \sum_{x \in \mathcal{X}} \sum_{y \in \mathcal{Y}} p(x, y) \log p(y | x)$$
# 
# kde $p(y|x) = \frac{p(x, y)}{p(X)}$.


# DATA

# TODO: NASTAVIT
# H(X|Y) ... řádky x, sloupce y
# x\y
matice = np.matrix([
    [1/8, 1/16, 1/32, 1/32],
    [1/16, 1/8, 1/32, 1/32],
    [1/16, 1/16, 1/16, 1/16],
    [1/4, 0, 0, 0],
])
#matice = np.transpose(matice) # TODO: pokud chceš H(Y|X)

# PROGRAM
def podminena_entropie(matrix, axis):
    p_y = np.squeeze(np.asarray(np.sum(matrix, axis = 0)))

    result = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] == 0:
                continue
            p_xy = matrix[i, j]
            p_y_con_x = p_xy / p_y[j]
            result += p_xy * np.log2(p_y_con_x)
    
    return -result

_res = podminena_entropie(matice, 0)
print("H(X|Y) =", Fraction(_res).limit_denominator(), "=", _res)

# ## Relativní entropie
# 
# https://marast.fit.cvut.cz/cs/problems/10196
# 
# Relativní entropie nebo Kullback-Leiblerova vzdálenost $D(p||q)$ mezi diskrétním
# rozdělením $p$ a diskrétním rozdělením $q$ na množině $X$ je definována vztahem
# 
# $$D(p \| q)=\sum_{x \in \mathcal{X}} p(x) \log \frac{p(x)}{q(x)}$$
# 
# - Klademe $0 \log{\frac{0}{0}} = 0, 0 \log{\frac{0}{q}} = 0$ a $p \log{\frac{p}{0}} = + \infty$.
# - Pokud tedy existuje $x$ tak, že $p(x) > 0$ a $q(x) = 0$, platí $D(p||q) = + \infty$.
# - $D(p||q)$ lze chápat jako vzdálenost, protože jak uvidíme později, je vždy nezáporná a
# rovna 0 pouze, pokud jsou $p$ a $q$ stejné.
# - Na rozdíl od opravdové vzdálenosti ale $D(p||q)$ obecně **není** rovno $D(q||p)$ a ani trojúhelníková nerovnost neplatí.


# DATA

# TODO: NASTAVIT
p = np.array([1/4, 1/4, 1/4, 1/4])
q = np.array([5/8, 1/4, 1/16, 1/16])

# PROGRAM
def relativni_entropie(p, q):
    result = 0
    for i in range(p.size):
        result += p[i] * math.log2(p[i] / q[i])
    return result

_res = relativni_entropie(p, q)
print("D(p||q) =", Fraction(_res).limit_denominator(), "=", _res)

# ## Vzájemná informace
# 
# https://marast.fit.cvut.cz/cs/problems/10197
# 
# Vzájemná informace $I(X; Y )$ diskrétních náhodných veličin $X$ a $Y$ je definována vztahem
# 
# $$I(X;Y)=\sum_{x \in \mathcal{X}}\sum_{y \in \mathcal{Y}} p(x, y) \log \frac{p(x,y)}{p(x) p(y)}$$
# 
# - Jedná se tedy o relativní entropii skutečného sdruženého rozdělení a rozdělení nezávislých veličin se stejnými marginálami,
# 
# $$I(X;Y)=D(p(x,y) || p(x)p(y))$$
# 
# - Ze symetrie definičního vztahu plyne $I(X;Y) = I(Y;X)$
# - Jak uvidíme za chvíli, z nezápornosti relativní entropie plyne $I(X;Y) \ge 0$.


# DATA

# TODO: NASTAVIT
# x\y
matice = np.matrix([
    [1/8, 1/16, 1/32, 1/32],
    [1/16, 1/8, 1/32, 1/32],
    [1/16, 1/16, 1/16, 1/16],
    [1/4, 0, 0, 0],
])

# PROGRAM
def vzajemna_informace(matrix):
    p_x = np.squeeze(np.asarray(np.sum(matrix, axis = 0)))
    p_y = np.squeeze(np.asarray(np.sum(matrix, axis = 1)))

    result = 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] == 0:
                continue
            p_xy = matrix[i, j]
            p_xy_x_y = p_xy / (p_x[j] * p_y[i])
            result += p_xy * np.log2(p_xy_x_y)
    
    return result

_res = vzajemna_informace(matice)
print("I(X;Y) =", Fraction(_res).limit_denominator(), "=", _res)



# -- Kódování --

# # Kódování


# ## Střední délka kódu
# 
# Střední délku $L(C)$ kódu $C$ náhodné veličiny $X$ s rozdělením $p(x)$ definujeme jako
# 
# $$L(C) = \sum_{x \in \chi}{l(p)p(x)}$$
# 
# kde $l(x)$ je délka kódového slova příslušejícího prvku $x \in \chi$.


# DATA

# TODO: NASTAVIT
k = 2 # báze kódování
slova = np.array(["11110", "11111", "0", "10", "1110", "110"])
pst   = np.array([1/16, 1/16, 1/4, 1/4, 1/8, 1/4])

# PROGRAM
def stredni_delka_kodu(slova, pst, k):
    result = 0

    for i in range(slova.size):
        result += len(slova[i]) * pst[i]

    return result

_res = stredni_delka_kodu(slova, pst, k)
print("L(C) =", Fraction(_res).limit_denominator(), "=", _res)
_e = entropie(np.matrix([pst]), 0)
print("H(x) =", Fraction(_e).limit_denominator(), "=", _e)
print("platí H(X) <= L(C) <= H(X) + 1" if _e <= _res <= _e + 1 else "neplatí H(X) <= L(C) <= H(X) + 1")

# ## Dekódování zprávy


# DATA

# TODO: NASTAVIT
nahodna_velicina = np.array([1, 2, 3, 4])
slova = np.array(["0", "10", "110", "111"])
zprava = "0110111100110"

# PROGRAM
def dekodovat_zpravu(nahodna_velicina, slova, zprava):
    result = []
    tmp = ""

    for i in range(len(zprava)):
        tmp += zprava[i]
        itemindex = np.where(slova==tmp)
        if itemindex[0].size != 0:
            result.append(nahodna_velicina[itemindex[0][0]])
            tmp = ""

    return result

_res = dekodovat_zpravu(nahodna_velicina, slova, zprava)
print("původní zpráva =", _res)

# ## Huffmanovo kódování
# 
# https://marast.fit.cvut.cz/cs/problems/10190
# 
# - Optimální kód.


# DATA

# TODO: NASTAVIT
x_chars = "m" * 5 + "b" * 5 + "h" * 15 + "k" * 20 + "p" * 20 + "o" * 35 # string s počty znaků dle poměru pst

# PROGRAM
x_chars = Counter(x_chars)
# Recursively builds code from binary tree
def build_encoding_from_tree(tree, code = ''):
    # either tuple or string (tree list)
    if type(tree) == str:
        return {tree: code}

    # recursion
    left = build_encoding_from_tree(tree[0], code + '1')
    right = build_encoding_from_tree(tree[1], code + '0')

    # Unpact left and right dicts
    return {**left, **right}

# Parses char: count dictionary to binary tree
def parse_characters_to_tree(chars_counter):
    tree = chars_counter.most_common()[::-1]

    while len(tree) > 1:
        # sort
        tree.sort(key=lambda elem: elem[1])

        # get first and second worst (that is least common) chars
        first_worst = tree[0]
        second_worst = tree[1]
        rest = tree[2:]

        # concat first and second worst chars
        # [(char_1, char_2), value_1 + value_2] + rest of the list
        tree = [((first_worst[0], second_worst[0]), first_worst[1] + second_worst[1])] + rest

    # root of parsed tree
    return tree[0][0]

# Get encoding map
def get_encoding(chars_counter):
    return build_encoding_from_tree(parse_characters_to_tree(chars_counter))

_res = get_encoding(x_chars)
print("huffmanovo kódování =", _res)



# -- testy --

# # Testy
# 
# ## Jaký test zvolit
# 
# - Jednovýběrový
#     - Jeden set dat.
#     - Porovnává se pouze střední hodnota nebo rozptyl
#     - Např. točí hospoda podmírák?
# - Dvouvýběrový
#     - Dva sety dat ($X$ o délce $|X| = n$ a $Y$ o délce $|Y| = m$).
#     - Např. točení piva v dvou hospodách.
# - Párový
#     - Dva sety dat a data spolu souvisí ($X, Y$ kde délka $|X| = |Y|$).
#     - Např. body před a po kurzu.
# - Kontingenční tabulka
#     - Testování nezávislosti.
#     - Zadané tabulkou.
#     - Např. ne/spokojenost call centra. 
# - Test dobré shody
#     - Testování nezávislosti.
#     - Biny.
#     - Neřeší pořadí (např. hodů hlava/orel).
#     - Např. pochází data z rovnoměrného rozdělení?
# - Bloky
#     - Testování nezávislosti.
#     - Řeší pořadí (např. hodů hlava/orel).
# 
# ## Kritický obor
# 
# Pokud platí podmínka **kritického oboru**, pak **zamítáme $H_0$** ve prospěch $H_A$. Pokud neplatí, pak **nezamítáme $H_0$**.


_data = np.array([-1.5,0.5,0.5,1.5,-1.5,0.5,-2.5,-1.5])

# ## Výběrový průměr
# 
# $$\overline{X}_n = \dfrac{1}{n}\sum_{i=1}^nX_i$$


def vyberovy_prumer(x):
    #Xn = sum(x) / len(x)
    return np.mean(x)

vyberovy_prumer(_data)

# ## Výběrový rozptyl
# 
# $$s_n^2 = s_X^2 = \dfrac{1}{n-1}\sum_{i=1}^n(X_i-\overline{X}_n)^2$$


def vyberovy_rozptyl(x):
    #sn2 = sum( (x-np.mean(x))**2 / (len(x) - 1) )
    return np.var(x, ddof = 1)

vyberovy_rozptyl(_data)

# ## Výběrová směrodatná odchylka
# 
# $$s_n = \sqrt{s_n^2}$$


def vyberova_smerodatna_odchylka(x):
    #sn = np.sqrt(sn2)
    return np.std(x, ddof = 1)

vyberova_smerodatna_odchylka(_data)



# -- jednovyberovy_ttest --

# # Jednovýběrový t-test


Image(filename='jednovyberovy_ttest.png') 

# ### Známý rozptyl
# 
# $z_\alpha$ můžeme aproximovat hodnotou $t_\alpha$ v $+\infty$.


# DATA

# TODO: NASTAVIT
#x = np.array([0.510, 0.462, 0.451, 0.466, 0.491, 0.503, 0.475, 0.487, 0.512, 0.505])
x = np.array([0] * 15 + [1] * 10)
mu0 = 0.5
alpha = 0.05

# vypočítaná data
n = len(x)
Xn = sum(x)/n # np.mean(x)
ro2 = 1/4
ro = np.sqrt(ro2) # np.std(x, ddof = 1)
print('Xn = ', Xn, ", ro^2 = ", ro2, ", ro = ", ro, ", n = ", n , sep = "")

# #### a) $H_0: \mu = 0.5$ vs. $H_A: \mu \neq 0.5$


# RUČNĚ

T = (Xn - mu0)/ro*np.sqrt(n)
z = st.norm.isf(alpha/2) # tabulky
print("T = ", T, ", z = ", z)
print("|T| >= z : ", np.abs(T) >= z, "=>", "Zamítám" if np.abs(T) >= z else "Nezamítám")

# POMOCÍ FUNKCE TTEST

_res = st.ttest_1samp(x, mu0, alternative = 'two-sided')
print(_res)
p = _res[1]
print("alpha > p : ", alpha > p, "=>", "Zamítám" if alpha > p else "Nezamítám")

# #### b) $H_0: \mu \geq 0.5$ vs. $H_A: \mu < 0.5$


# RUČNĚ

T = (Xn - mu0)/ro*np.sqrt(n)
z = st.norm.isf(alpha) # tabulky
print("T = ", T, ", z = ", z)
print("T <= -z : ", T <= -z, "=>", "Zamítám" if T <= -z else "Nezamítám")

# POMOCÍ FUNKCE TTEST

_res = st.ttest_1samp(x, mu0, alternative = 'less')
print(_res)
p = _res[1]
print("alpha > p : ", alpha > p, "=>", "Zamítám" if alpha > p else "Nezamítám")

# #### c) $H_0: \mu \leq 0.5$ vs. $H_A: \mu \gt 0.5$


# RUČNĚ

T = (Xn - mu0)/ro*np.sqrt(n)
z = st.norm.isf(alpha) # tabulky
print("T = ", T, ", z = ", z)
print("T >= z : ", T >= z, "=>", "Zamítám" if T >= z else "Nezamítám")

# POMOCÍ FUNKCE TTEST

_res = st.ttest_1samp(x, mu0, alternative = 'greater')
print(_res)
p = _res[1]
print("alpha > p : ", alpha > p, "=>", "Zamítám" if alpha > p else "Nezamítám")

# ### Neznámý rozptyl


# DATA

# TODO: NASTAVIT
#x = np.array([0.510, 0.462, 0.451, 0.466, 0.491, 0.503, 0.475, 0.487, 0.512, 0.505])
x = np.array([0] * 15 + [1] * 10)
mu0 = 0.5
alpha = 0.05

# vypočítaná data
n = len(x)
Xn = sum(x)/n # np.mean(x)
sn2 = sum((x-Xn)**2/(n-1)) # np.var(x, ddof = 1)
sn = np.sqrt(sn2) # np.std(x, ddof = 1)
print('Xn = ', Xn, ", sn^2 = ", sn2, ", sn = ", sn, ", n = ", n , sep = "")

# #### a) $H_0: \mu = 0.5$ vs. $H_A: \mu \neq 0.5$


# RUČNĚ

T = (Xn - mu0)/sn*np.sqrt(n)
t = st.t.isf(alpha/2, n-1) # tabulky
print("T = ", T, ", t = ", t)
print("|T| >= t : ", np.abs(T) >= t, "=>", "Zamítám" if np.abs(T) >= t else "Nezamítám")

# POMOCÍ FUNKCE TTEST

_res = st.ttest_1samp(x, mu0, alternative = 'two-sided')
print(_res)
p = _res[1]
print("alpha > p : ", alpha > p, "=>", "Zamítám" if alpha > p else "Nezamítám")

# #### b) $H_0: \mu \geq 0.5$ vs. $H_A: \mu < 0.5$


# RUČNĚ

T = (Xn - mu0)/sn*np.sqrt(n)
t = st.t.isf(alpha, n-1) # tabulky
p = st.t.cdf(T,n-1)
print("T = ", T, ", t = ", t)
print("T <= -t : ", T <= -t, "=>", "Zamítám" if T <= -t else "Nezamítám")

# POMOCÍ FUNKCE TTEST

_res = st.ttest_1samp(x, mu0, alternative = 'less')
print(_res)
p = _res[1]
print("alpha > p : ", alpha > p, "=>", "Zamítám" if alpha > p else "Nezamítám")

# #### c) $H_0: \mu \leq 0.5$ vs. $H_A: \mu \gt 0.5$


# RUČNĚ

T = (Xn - mu0)/sn*np.sqrt(n)
t = st.t.isf(alpha, n-1) # tabulky
print("T = ", T, ", t = ", t)
print("T >= t : ", T >= t, "=>", "Zamítám" if T >= t else "Nezamítám")

# POMOCÍ FUNKCE TTEST

_res = st.ttest_1samp(x, mu0, alternative = 'greater')
print(_res)
p = _res[1]
print("alpha > p : ", alpha > p, "=>", "Zamítám" if alpha > p else "Nezamítám")

# -- dvouvyberovy_ttest --

# # Dvouvýběrový t-test


Image(filename='dvouvyberovy_ttest.png') 

# DATA

# TODO: NASTAVIT
x = np.array([0.510, 0.462, 0.451, 0.466])
y = np.array([0.491, 0.503, 0.475, 0.487, 0.512, 0.505])
alpha = 0.05

# vypočítaná data
n = len(x)
m = len(y)
Xn = np.mean(x)
Ym = np.mean(y)
sX2 = np.var(x, ddof=1)
sY2 = np.var(y, ddof=1)
sX = np.std(x, ddof=1)
sY = np.std(y, ddof=1)
print('Xn = ', Xn, ", sX = ", sX, ", n = ", n , sep = "")
print('Ym = ', Ym, ", sY = ", sY, ", m = ", m , sep = "")

# ### Různé rozptyly - není důvod předpokládat, že by byly stejné


# #### a) $H_0: \mu_X = \mu_Y$ vs. $H_A: \mu_X \neq \mu_Y$


# RUČNĚ

sd2 = sX2/n + sY2/m
nd = sd2**2/((sX2/n)**2/(n-1) + (sY2/m)**2/(m-1))
T = (Xn - Ym)/np.sqrt(sd2)
t = st.t.isf(alpha/2,nd) # tabulky
print("T = ", T, ", t = ", t , ", nd = ", nd, sep="")
print("|T| >= t : ", np.abs(T) >= t, "=>", "Zamítám" if np.abs(T) >= t else "Nezamítám")

# POMOCÍ FUNKCE TTEST

_res = st.ttest_ind(x, y, alternative = 'two-sided', equal_var = False)
print(_res)
p = _res[1]
print("alpha > p : ", alpha > p, "=>", "Zamítám" if alpha > p else "Nezamítám")

# #### b) $H_0: \mu_X \geq \mu_Y$ vs. $H_A: \mu_X < \mu_Y$


# RUČNĚ

sd2 = sX2/n + sY2/m
nd = sd2**2/((sX2/n)**2/(n-1) + (sY2/m)**2/(m-1))
T = (Xn - Ym)/np.sqrt(sd2)
t = st.t.isf(alpha,nd) # tabulky
print("T = ", T, ", t = ", t , ", nd = ", nd, sep="")
print("T <= -t : ", T <= -t, "=>", "Zamítám" if T <= -t else "Nezamítám")

# POMOCÍ FUNKCE TTEST

_res = st.ttest_ind(x, y, alternative = 'less', equal_var = False)
print(_res)
p = _res[1]
print("alpha > p : ", alpha > p, "=>", "Zamítám" if alpha > p else "Nezamítám")

# #### c) $H_0: \mu_X \leq \mu_Y$ vs. $H_A: \mu_X \gt \mu_Y$


# RUČNĚ

sd2 = sX2/n + sY2/m
nd = sd2**2/((sX2/n)**2/(n-1) + (sY2/m)**2/(m-1))
T = (Xn - Ym)/np.sqrt(sd2)
t = st.t.isf(alpha,nd) # tabulky
print("T = ", T, ", t = ", t , ", nd = ", nd, sep="")
print("T >= t : ", T >= t, "=>", "Zamítám" if T >= t else "Nezamítám")

# POMOCÍ FUNKCE TTEST

_res = st.ttest_ind(x, y, alternative = 'greater', equal_var = False)
print(_res)
p = _res[1]
print("alpha > p : ", alpha > p, "=>", "Zamítám" if alpha > p else "Nezamítám")

# ### Shodné rozptyly - je důvod předpokládat shodu, nebo si to otestuji


# a) $H_0: \mu_X = \mu_Y$ vs. $H_A: \mu_X \neq \mu_Y$


# RUČNĚ

s12 = np.sqrt(((n-1)*sX2 + (m-1)*sY2)/(n+m-2))
T = (Xn - Ym)/s12*np.sqrt(n*m/(n+m))
t = st.t.isf(alpha/2,n + m - 2) # tabulky
p = 2 * st.t.sf(np.abs(T), n + m - 2)
print("T = ", T, ", t = ", t,", s12 = ", s12, sep="")
print("|T| >= t : ", np.abs(T) >= t, "=>", "Zamítám" if np.abs(T) >= t else "Nezamítám")

# POMOCÍ FUNKCE TTEST

_res = st.ttest_ind(x, y, alternative = 'two-sided', equal_var = True)
print(_res)
p = _res[1]
print("alpha > p : ", alpha > p, "=>", "Zamítám" if alpha > p else "Nezamítám")

# #### b) $H_0: \mu_X \geq \mu_Y$ vs. $H_A: \mu_X < \mu_Y$


# RUČNĚ
s12 = np.sqrt(((n-1)*sX2 + (m-1)*sY2)/(n+m-2))
T = (Xn - Ym)/s12*np.sqrt(n*m/(n+m))
t = st.t.isf(alpha,n+m-2) # tabulky
print("T = ", T, ", t = ", t , ", s12 = ", s12, sep="")
print("T <= -t : ", T <= -t, "=>", "Zamítám" if T <= -t else "Nezamítám")

# POMOCÍ FUNKCE TTEST

_res = st.ttest_ind(x, y, alternative = 'less', equal_var = True)
print(_res)
p = _res[1]
print("alpha > p : ", alpha > p, "=>", "Zamítám" if alpha > p else "Nezamítám")

# #### c) $H_0: \mu_X \leq \mu_Y$ vs. $H_A: \mu_X \gt \mu_Y$


# RUČNĚ
s12 = np.sqrt(((n-1)*sX2 + (m-1)*sY2)/(n+m-2))
T = (Xn - Ym)/s12*np.sqrt(n*m/(n+m))
t = st.t.isf(alpha,n+m-2) # tabulky
print("T = ", T, ", t = ", t , ", s12 = ", s12, sep="")
print("T >= t : ", T >= t, "=>", "Zamítám" if T >= t else "Nezamítám")

# POMOCÍ FUNKCE TTEST

_res = st.ttest_ind(x, y, alternative = 'greater', equal_var = True)
print(_res)
p = _res[1]
print("alpha > p : ", alpha > p, "=>", "Zamítám" if alpha > p else "Nezamítám")



# -- parovy_ttest --

# # Párový t-test
# 
# 
# Pokud chci $\ge$, používám $p = st.t.cdf(T,n-1)$.
# 
# Pokud chci $\le$, používám $p = st.t.cdf(np.abs(T),n-1)$.


Image(filename='parovy_ttest.png') 

# DATA

# TODO: NASTAVIT
x = np.array([0.82, 1.08, 1.01, 0.63, 1.45, 1.12, 0.56, 0.83, 1.16, 1.38])
y = np.array([0.94, 0.79, 0.75, 0.74, 1.25, 0.79, 0.76, 0.75, 0.78, 0.78])
z = x - y
#z = np.array([1.5, -0.5, -0.5, -1.5, +1.5, -0.5, +2.5, 1.5])
alpha = 0.05

# vypočítaná data
n = len(z)
Zn = sum(z)/n # np.mean(x)
sz2 = np.var(z, ddof = 1)
sz = np.std(z, ddof = 1)
print('Zn = ', Zn, ", sz^2 = ", sz2, ", sz = ", sz, ", n = ", n, sep = "")

# #### a) $H_0: \mu_x = \mu_y$ vs. $H_A: \mu_x \neq \mu_y$


# RUČNĚ

T = Zn/sz * np.sqrt(n)
t = st.t.isf(alpha/2, n-1) # tabulky
print("T = ", T, ", t = ", t)
print("|T| >= t : ", np.abs(T) >= t, "=>", "Zamítám" if np.abs(T) >= t else "Nezamítám")

# POMOCÍ FUNKCE TTEST

# TODO: POZOR, PŘI ZADANÉM X A Y
_res = st.ttest_rel(x, y, alternative = 'two-sided')
print(_res)
p = _res[1]
print("alpha > p : ", alpha > p, "=>", "Zamítám" if alpha > p else "Nezamítám")

# #### b) $H_0: \mu_x \ge \mu_y$ vs. $H_A: \mu_x \lt \mu_y$


# RUČNĚ

T = Zn/sz * np.sqrt(n)
t = st.t.isf(alpha, n-1) # tabulky
print("T = ", T, ", t = ", t)
print("T <= -t : ", T <= -t, "=>", "Zamítám" if T <= -t else "Nezamítám")

# POMOCÍ FUNKCE TTEST

# TODO: POZOR, JEN PŘI ZADANÉM X A Y
_res = st.ttest_rel(x, y, alternative = 'less')
print(_res)
p = _res[1]
print("alpha > p : ", alpha > p, "=>", "Zamítám" if alpha > p else "Nezamítám")

# #### c) $H_0: \mu_x \le \mu_y$ vs. $H_A: \mu_x \gt \mu_y$


# RUČNĚ

T = Zn/sz * np.sqrt(n)
t = st.t.isf(alpha, n-1) # tabulky
print("T = ", T, ", t = ", t)
print("T >= t : ", T >= t, "=>", "Zamítám" if T >= t else "Nezamítám")

# POMOCÍ FUNKCE TTEST

# TODO: POZOR, JEN PŘI ZADANÉM X A Y
_res = st.ttest_rel(x, y, alternative = 'greater')
print(_res)
p = _res[1]
print("alpha > p : ", alpha > p, "=>", "Zamítám" if alpha > p else "Nezamítám")



# -- kontingencni_tabulka --

# # Kontingenční tabulka
# 
# Aby fungovala asymptotika, je doporučeno, aby $\forall i, j : n\hat{p}_{i\bullet}\hat{p}_{\bullet j} \ge 5$.


Image(filename='kontingencni_tabulka.png') 

# DATA

# TODO: nastavit
# skutečné četnosti
# x\y
Nij = np.matrix([
    [1, 11, 7, 21],
    [0, 8, 23, 29]
])
alpha = 0.05

# vypočítaná data
n = np.sum(Nij)
print("n =", n)
print("Nij =")
print(Nij)
print("Hezky:")
vystup_zlomky(Nij)

# RUČNĚ

# odhady marginál
p_i = np.sum(Nij, axis = 1)/n
p_j = np.sum(Nij, axis = 0)/n
print("pi_ =\n", p_i)
print("p_j =\n", p_j)

# teoretické četnosti
pipj = np.matmul(p_i,p_j)
print("pipj =\n",pipj)
npipj = n * pipj
print("npipj =\n",npipj)

# nutné sloučit 1. a 2. sloupec
cc12 = np.sum(Nij[:,:2], axis = 1)
cc34 = Nij[:,2:]
Nij = np.append(cc12, cc34, axis = 1)
n = np.sum(Nij)
print("Nij =\n", Nij)
print("n =", n)

# odhady marginál
pi_ = np.sum(Nij, axis = 1)/n
p_j = np.sum(Nij, axis = 0)/n
print("pi_ =\n", pi_)
print("p_j =\n", p_j)
# teoretické četnosti
pipj = np.matmul(pi_,p_j)
print("pipj =\n",pipj)
npipj = n * pipj
print("npipj =\n",npipj)

# testová statistika
Chi2 = np.sum(np.square(Nij - npipj)/npipj)
print("Chi2 =", Chi2)
# kritická hodnota
df = (np.size(Nij,axis =0) - 1)*(np.size(Nij,axis =1) - 1)
print("df =",df)
chi2 = st.chi2.isf(alpha,df)
print("chi2 =", chi2)

# p-hodnota
p = st.chi2.sf(np.abs(Chi2),df) # = 1-st.chi2.cdf(Chi2,df)
print("p =", p)

print("...")
print("Chi2 >= chi2 : ", Chi2 >= chi2, "=>", "Zamítám" if Chi2 >= chi2 else "Nezamítám")
print("alpha > p : ", alpha > p, "=>", "Zamítám" if alpha > p else "Nezamítám")

# POMOCÍ FUNKCE

Chi2, p, df, _ = st.chi2_contingency(Nij, correction = False)
print("Chi2 =", Chi2)
print("df =", df)
print("p =", p)
print("alpha > p : ", alpha > p, "=>", "Zamítám" if alpha > p else "Nezamítám")



# -- test_dobre_shody --

# # Test dobré shody
# 
# Je třeba pamatovat na to, že test $\chi^2$ je asymptotický a proto ho lze použít jen pro dostatečně velký rozsah výběru $n$. Obvykle se uvádí, že musí platit $n \cdot p_i \ge 5$ pro každé $i$.


Image(filename='test_dobre_shody.png')

# DATA

# TODO: nastavit
x = np.array([0.055 , 0.068 , 0.102 , 0.105 , 0.118 , 0.160 , 0.178 , 0.203 , 0.260 , 0.274, 0.289 , 0.291 , 0.346 , 0.358 , 0.366 , 0.472 , 0.588 , 0.617 , 0.721 , 0.932])
#x = np.array([0] * 15 + [1] * 10)
k = 4 # počet binů
alpha = 0.05

# vypočítaná data
n = len(x)
# četnosti
Ni, edges = np.histogram(x, bins = k, range = (0,1), density = None)
n = np.sum(Ni)
print("Edges =" ,edges)
print("n = ", n, ", Ni = ", Ni, sep="")

# RUČNĚ

# teoretické četnosti
pi = np.ones(k) * 1/k
npi = n * pi
print("pi =", pi)
print("npi =", npi)

# testová statistika
Chi2 = sum((Ni - npi)**2/npi)
print("Chi2 =", Chi2)

# kritická hodnota
df = k-1
chi2 = st.chi2.isf(alpha,df) # tabulky
print("chi2 =", chi2)

# p-hodnota
p = st.chi2.sf(Chi2,df) # = 1-st.chi2.cdf(Chi2,df)
print("p =", p)

print("...")
print("Chi2 >= chi2 : ", Chi2 >= chi2, "=>", "Zamítám" if Chi2 >= chi2 else "Nezamítám")
print("alpha > p : ", alpha > p, "=>", "Zamítám" if alpha > p else "Nezamítám")

# pomocí funkce
_res = st.chisquare(Ni,npi)
print(_res)
print("alpha > p : ", alpha > p, "=>", "Zamítám" if alpha > p else "Nezamítám")



# -- bloky --

# # Bloky
# 
# $z_\alpha$ můžeme aproximovat hodnotou $t_\alpha$ v $+\infty$.


# ## Bloky nad/pod střední hodnotou


Image(filename='bloky_dle_stredni_hodnoty.png')

# DATA

# TODO: NASTAVIT
Nn = 10
n = 20
alpha = 0.1

# vypočítaná data
T = (2*Nn - n - 1)/math.sqrt(n-1)
z = st.norm.isf(alpha/2) # tabulky
p = 2 * st.norm.sf(np.abs(T))
print("T = ", T, ", z = ", z)
print("p = ", p)

print("...")
print("|T| >= z : ", np.abs(T) >= z, "=>", "Zamítám" if np.abs(T) >= z else "Nezamítám")
print("alpha > p : ", alpha > p, "=>", "Zamítám" if alpha > p else "Nezamítám")

# ## Bloky nahoru/dolů


Image(filename='Bloky_dle_monotonie.png')

# DATA

# TODO: NASTAVIT
Nn = 20
n = 30
alpha = 0.1

# vypočítaná data
T = (3*Nn - 2*n + 1)/math.sqrt(1.69*n-2.9)
z = st.norm.isf(alpha/2) # tabulky
p = 2 * st.norm.sf(np.abs(T))
print("T = ", T, ", z = ", z)
print("p = ", p)

print("...")
print("|T| >= z : ", np.abs(T) >= z, "=>", "Zamítám" if np.abs(T) >= z else "Nezamítám")
print("alpha > p : ", alpha > p, "=>", "Zamítám" if alpha > p else "Nezamítám")

# ### Kritická hodnota normálního rozdělení


alfa = 0.1
z = st.norm.isf(alfa/2) # pozor pro z_alfa/2
print("Kritická hodnota normálního rozdělení v bodě alfa/2 =", alfa/2, "=", z)

# ### Zjištění p-hodnoty


T = 1.64485 # statistika co ti vyšla
p = st.norm.sf(np.abs(T))
print("Pro testovou statistiku T =", T, "je p-hodnota", p*2) # pozor pro Z_alpha/2



# -- markovsky_retezec --

# # Markovský řetězec
# 
# Náhodný proces ${Xn | n ∈ N0}$ s nejvýše spočetnou množinou stavů $S$ nazýváme markovský řetězec s diskrétním časem, pokud splňuje markovskou podmínku, tj. pokud $∀n ∈ N, a ∀s, s_0, . . . , s_{n−1} \in S$ platí
# 
# $$P(X_n = s|X_{n−1} = s_{n−1}, \cdots, X_1 = s_1, X_0 = s_0) = P(X_n = s|X_{n−1} = s_{n−1})$$


# ## Stacionární rozdělení
# 
# https://marast.fit.cvut.cz/cs/problems/10180
# 
# Buď ${X_n | n \in \mathbb{N}_0}$ homogenní markovský řetězec s maticí přechodu $\textbf{P}$.
# 
# Pokud existuje vektor $\pi$ takový, že
# 
# - $\forall i \in S: \pi_i \ge 0$,
# - $\sum{i \in S}{\pi_i = 1}$,
# 
# pro který platí, že
# 
# $$\pi * \textbf{P} = \pi$$
# 
# nazýváme jej *stacionárním rozdělením řetězce*.


# DATA

# TODO: NASTAVIT
matice = np.matrix([
    [0.95,0.05,0,0,0,0],
    [0,0.9,0.1,0,0,0],
    [0,0,0.875,0.125,0,0],
    [0,0,0,0,1,0],
    [0,0,0,0,0,1],
    [1,0,0,0,0,0],
])

# PROGRAM
def stacionarni_rozdeleni(matrix):
    W = np.transpose(matrix - np.eye(matrix.shape[0])) 
    pi = la.null_space(W)
    pi = np.transpose(pi/sum(pi))
    return pi

_res = stacionarni_rozdeleni(matice)
print("Stacionární rozdělení =")
vystup_zlomky(_res)

# ## Fundamentální matice
# 
# Matice $N = (I - T)^{-1}$ se nazývá *fundamentální matice řetězce*.


# DATA

# TODO: NASTAVIT
# matice T
matice_t = np.matrix([
    [0, 1/2],
    [1/2, 0],            
])

# PROGRAM
def fundamentalni_matice(matrix_t):
    return inv(np.eye(matrix_t.shape[0]) - matrix_t)

_res = fundamentalni_matice(matice_t)
print("Fundamentální matice =")
vystup_zlomky(_res)

# ## Pravděpodobnost pohlcení
# 
# Pro matici pravděpodobností pohlcení platí
# 
# $$U = N \cdot R = (I - T)^{-1}R$$
# 
# Matice přechodu $\textbf{P}$ má tvar
# 
# $$\textbf{P} = 
# \begin{pmatrix}
# \textbf{T} & \textbf{R} \\
# \textbf{0} & \textbf{C}
# \end{pmatrix}
# $$


# DATA

# TODO: NASTAVIT
# matice T
matice_t = np.matrix([
    [0, 1/2],
    [1/2, 0],            
])
# matice R
matice_r = np.matrix([
    [1/2, 0],
    [0, 1/2],            
])

# PROGRAM
def pravdepodobnost_pohlceni(matrix_t, matrix_r):
    return fundamentalni_matice(matrix_t) * matrix_r

_res = pravdepodobnost_pohlceni(matice_t, matice_r)
print("Pravděpodobnost pohlcení =")
vystup_zlomky(_res)

# ## Střední doba kroků do pohlcení
# 
# Pro matici $N = (I - T)^{-1}$ platí $N_{ik} := E(W_k|X_0=i)$, tj. $N_{ik}$ označuje *střední počet návštěv* stavu $k \in T$, jestliže řetězec startuje v $i \in T$. 


Image(filename='stredni-doba-do-pohlceni.png')

# DATA

# TODO: NASTAVIT
# matice T
matice_t = np.matrix([
    [0, 1/2],
    [1/2, 0],            
])

# PROGRAM
def stredni_doba_kroku_do_pohlceni(matrix_t):
    return fundamentalni_matice(matrix_t) * np.ones((matrix_t.shape[0],1))

_res = stredni_doba_kroku_do_pohlceni(matice_t)
print("Vektor středních dob kroků do pohlcení =")
print(_res)
print("Hezky:")
vystup_zlomky(_res)

# ## Matice skokových intenzit na matici přechodů
# 
# $$ P'(t) = P(t)Q$$


# ## Stacionární rozdelení i pro spojité
# 
# $$ \pi Q = 0 $$
# 
# $$ \pi P(t) = \pi $$


# ## Pravděpodobnostní rozdělení v čase t
# 
# https://marast.fit.cvut.cz/cs/problems/10788
# 
# $$ p'(t) = p(t)Q$$
# $$ p(0) = p_{initial}$$
# p(0) je zadaný


# ## Formulace diskrétní řetězec na spojitý a naopak
# 
# $$ D := I + \frac{1}{\lambda}Q$$
# 
# $$ Q = \lambda(D-I)$$


# ## Podmínky detailní rovnováhy
# 
# https://marast.fit.cvut.cz/cs/problems/10791
# 
# 
# $$ \pi_i Q_{ij} = \pi_j Q_{ji}$$
# 
# $$ \sum_{i=0} \pi = 1$$




# -- system_hromadne_obsluhy --

# # Systém hromadné obsluhy


# 
# - Pravděpodobnost čekání nějakou dobu
#     - není u dalších rozdělení


# ### Detailní rovnováha
# 
# 
# $$ \pi_i Q_{ij} = \pi_j Q_{ji}$$
# 
# $$ \sum_{i=0} \pi = 1$$


# ## Littleho věta a střední doby
# 
# 
# 
# $EN$ = střední počet zákazníků v systému
# 
# $ET$ = střední doba strávená zákazníkem v systému
# 
# $\lambda$ intenzita procesu příchodů
# 
# ### Littleho věta
# 
# $$ EN = \lambda ET$$
# 
# ### Další střední hodnoty
# 
# $$ET = E_\pi W + E_\pi S_j $$ 
# 
# $EW = $ Střední doba času stráveného zákazníkem ve frontě
# 
# $ES_j = \frac{1}{\mu} $ Střední doba obsluhy $j$-tého požadavku
# 
# Index $\pi$ znamená, vzhledem k stacionárnímu rozdělení
# 
# $$EN = EN_s + EN_f$$
# 
# $EN_s = $ střední počet zákazníků na serveru
# 
# $EN_f =$ střední počet zákazníků ve frontě


# ### Systém M|M|1
# 
# Stacionární rozdělení, existuje, pokud $\lambda < \mu$ 
# $$ \pi_n = (1 - \frac{\lambda}{\mu})(\frac{\lambda}{\mu})^n$$
# 
# $EN = \frac{\rho}{1-\rho}$
# 
# $EN_s = \rho$
# 
# $EN_f = \frac{\rho^2}{1-\rho}$
# 
# Pravděpodobnost, že zákazník bude hned odbaven a nebude čekat.
# 
# $P(W = 0) = P(X_t = 0) = \pi_0 = 1-\rho = 1-\frac{\lambda}{\mu}$
# 
# Pravděpodobnost, že zákazník bude čekat alespoň dobu w, pokud bude čekat.
# 
# $P(W > w | W > 0) \sim Exp(\mu - \lambda)$
# 
# $P(W > w | W > 0) = e^{-(\mu - \lambda)w}$


# ### Systém $M|M|\infty$
# 
# Podmínka detailní rovnováhy:
# $$ n\mu\pi_n = \lambda \pi_{n-1}$$
# 
# Stacionární rozdělení 
# 
# $$\pi_n = \frac{1}{n!}\left(\frac{\lambda}{\mu}\right)^n e^{-\frac{\lambda}{\mu}}$$


# ### Systém M|M|c
# 
# Stacionární rozdělení
# 
# Pokud $n \le c$
# $$\pi_n  = \frac{1}{n!}\left(\frac{\lambda}{\mu}\right)^n \pi_0$$
# jinak
# $$\pi_n  = \frac{c^c}{c!}\left(\frac{\lambda}{c\mu}\right)^n \pi_0$$




