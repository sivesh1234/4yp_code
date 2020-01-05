import scipy.stats as st
#Input random monte carlo features

monte_mean = 0.374
monte_std = 81
#Input Returns
returns = 114

#Proability test
z_number = (returns-monte_mean)/monte_std
print(1-(st.norm.cdf(z_number)))
