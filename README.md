# Notes about Implemented Reinforcement Learning Algorithms

## Deep Q_Network

### Pseudocode Used for Implementation <br>

Initialize replay memory $D$ to capacity $N$ <br>
Initialize action-value function $Q$ with random weights $\theta$ <br>
Initialize target action-value function $\hat{Q}$ with weights $\theta^{-}=\theta$ <br>
**For** episode=$1, M$ **do** <br>
&nbsp;&nbsp;&nbsp;&nbsp; Initialize sequence $s_1={x_1}$ and preprocessed sequence $\phi_1=\phi(s_1)$<br>
&nbsp;&nbsp;&nbsp;&nbsp; **For** $t=1, T$ **do** <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; With probability $\epsilon$ select a random action $a_t$ <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; otherwise select $a_t=argmax_a Q(\phi(s_t), a; \theta)$ <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Execute action $a_t$ in emulator and observe reward $r_t$ and image $x_{t+1}$ <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Set $s_{t+1}=s_t, a_t, x_{t+1}$ and preprocess $\phi_{t+1}=\phi (s_{t+1})$ <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Store transition $(\phi_t, a_t, r_t, \phi_{t+1})$ in $D$ <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Sample random minibatch of transitions $(\phi_j, a_j, r_j, \phi_{j+1})$ from $D$ <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Set $y_j=r_j$ if episode terminates at step $j+1$ else set $y_j=r_j+\gamma max_{a'} \hat{Q}(\phi_{j+1}, a'; \theta^-)$ <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Perform a gradient descent step on $(y_j-Q(\phi_j, a_j; \theta))^2$ with respect to the network parameters $\theta$ <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Every $C$ steps reset $\hat{Q}=Q$ <br>
&nbsp;&nbsp;&nbsp;&nbsp; **End For** <br>
**End For** <br>
<br>
### Results <br>
We first initialized a neural net to use for the deep q-network algorithm and ran it to see the average rewards per episode of this initialized, non-learned net. Next we started the training of the neural net for the deep q-network to update the weights to minimize the loss function from the pseudocode above. We stopped the training once it hit an average reward per episode of 200 for the last 100 episodes. The graph below shows the rewards collected per episode during the training process along with the average reward per episode before training and after training. <br>
![DQN](DQN_Result.png)


