import numpy as np

np.random.binomial(10, 0.1, 1000)


N = 10000
d = 10
pulls = []

epsilon = 1.0
epsilon_decay = 0.99

pulls = [] #array to hold current value counts 
#when an option is selected, append it to list
total = 0

for n in range(N):
    if random.random() < epsilon: 
        ad = random.randrange(d)
        print (ad)
    else:
        ad = np.argmax(rewards)

    rewards.append(ad)
    #update ad_results to maintain current count for each ad

    reward = 1 #df.values[n, ad]
    total = total + reward
