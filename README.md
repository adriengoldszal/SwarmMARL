# MARL Training for Swarm Rescue 
This is the training repo for Multi-agent Swarm Resue. The implementation is based on the framework created by this study – [onpolicy](https://github.com/marlbenchmark/on-policy) . The runner and renderer classes are customized for SwarmRL. 

## Installation
To run, clone the [SwarmRL](https://github.com/adriengoldszal/SwarmRL) and [SPG](https://github.com/emmanuel-battesti/simple-playgrounds) repos. This should however be done automatically through the requirements.txt file.

Please see detailed explanations on the SwarmRL (https://github.com/adriengoldszal/SwarmRL) repository for making the code run correctly.

## Repository structure

- **Algorithms** contains the multi agent RL algorithms that are usable for the training and rendering tasks
- **Runner** contains two classes, that are used as wrappers of the main functions needed for training and or rendering. Please use the >shared folder for multi-agent work.
- **Scripts** contains the scripts to >render and >train the agents, it also contains >results, with the saved models
