import json

with open("pcc_env_log_run_200_run6.json", "r") as json_file:
    result = json.load(json_file)

agent0_throughput = []
agent1_throughput = []
agent0_latency = []
agent1_latency = []

events = list(result.values())[0]
for event in events:
    agent0_throughput.append(event["Throughput"])
    agent1_throughput.append(event["Throughput2"])
    agent0_latency.append(event["Latency"])
    agent1_latency.append(event["Latency2"])

# import matplotlib.pyplot as plt
# plt.plot(agent0_throughput, label="Agent 0")
# plt.plot(agent1_throughput, label="Agent 1")
# plt.xlabel("step")
# plt.ylabel("throughput")
# plt.legend()
# plt.savefig("throughput.png")
# plt.ylim([0, 2000])

import matplotlib.pyplot as plt
plt.plot(agent0_latency, label="Agent 0")
plt.plot(agent1_latency, label="Agent 1")
plt.xlabel("step")
plt.ylabel("latency")
plt.legend()
plt.savefig("latency.png")
# plt.ylim([0, 2000])