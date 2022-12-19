
# Identifying Malicious Nodes in IoT Networks using Reinforcement Learning

## Avery Peiffer, Dr. Mai Abdelhakim
## University of Pittsburgh

I began working on this project as part of a graduate research class in Spring 2022. I wanted to gain exposure to reinforcement learning (RL), and Dr. Abdelhakim's research focuses heavily on IoT networks. I have been continuing to work on this project after graduating and hope that it continues to expose me to different facets of RL.

For this study, we are interested in the idea that devices can manipulate data transmitted in packets as it is being forwarded from a source device to a destination device. This would result in incorrect data arriving at the destination device. This may happen intentionally or unintentionally. We refer to these types of devices as malicious, because they can compromise the integrity of the packets that are sent throughout the network. 

Overall, we want to use reinforcement learning to create an algorithm that can identify which devices are likely to be malicious, so that we can avoid them when transmitting data through the network. 

The image below shows a sample partial mesh network in the simulation environment. Initially, we do not know which nodes (if any) are corrupted. In order to figure this out, we simulate sending packets of data from the source to the destination on different paths. When the data arrives at the destination, we see if it is corrupted and reward/punish our RL agent accordingly. Over many training iterations, the algorithm learns which paths to take and which to avoid, allowingus to form a relatively clear picture of which nodes are corrupted in the network.

![Network Image](/src/env/network_example.PNG)

Guide to navigating the repository:

 - `/papers/`
	 - `Abdelhakim_2018.pdf`: The source paper for this work
	 - `Peiffer_2022.pdf`: Initial conference paper presented based on this work
 - `/src/`
	 - `env/`: Contains a notebook walking through the IoT simulation environment setup
	 - `results/`: The accuracy, precision, and recall results from running simulations with the RL algorithm, as well as a notebook to plot these results
	 - `traintest/`: The notebook that trains and tests the algorithm
