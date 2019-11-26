import scipy.stats as st
#Input random monte carlo features

monte_mean = 0.374
monte_std = 18.7
#Input Returns
returns = 25

#Proability test
z_number = (returns-monte_mean)/monte_std
print(1-(st.norm.cdf(z_number)))
