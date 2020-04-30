import scipy.stats as st
#Input random monte carlo features

monte_mean = -3
monte_std = 70
#Input Returns
returns = 206

#Proability test
z_number = (returns-monte_mean)/monte_std
print(1-(st.norm.cdf(z_number)))
