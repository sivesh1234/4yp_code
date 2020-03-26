import scipy.stats as st
#Input random monte carlo features

monte_mean = 2.33
monte_std = 80.16
#Input Returns
returns = 200

#Proability test
z_number = (returns-monte_mean)/monte_std
print(1-(st.norm.cdf(z_number)))
