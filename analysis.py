NAME = "run1"
# extract per episode rewards
rewards = []
ewmas = []
f = open(NAME+".out", "r")
for line in f:
  if line.startswith("Agent 1 Reward"):
      reward = float(line.split(',')[0].split(':')[1].replace(' ', ''))
      ewma = float(line.split(':')[2].replace(' ', ''))
      rewards.append(reward)
      ewmas.append(ewma)

import matplotlib.pyplot as plt
plt.plot(ewmas)
plt.xlabel("episode")
plt.ylabel("agent 1 mean episode reward")
plt.savefig(NAME+".png")
# plt.ylim([0, 2000])