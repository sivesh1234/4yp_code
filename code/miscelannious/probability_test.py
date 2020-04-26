import scipy.stats as st
#Input random monte carlo features

monte_mean = -0.028
monte_std = 4.28
#Input Returns
returns = 12.66

#Proability test
z_number = (returns-monte_mean)/monte_std
print(1-(st.norm.cdf(z_number)))
