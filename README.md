# FL training homework
- As you can see from file [train_fl](train_fl.py) the global model is calculated by taking average weights from all clients, in details:
$G^t = \frac{1}{N} * \sum(L^t_i)$, but in the standard way, the updated of global model might be a little bit different:

$$G^t = G^t + \frac{1}{N} * \sum(L^t_i - G^t)$$

List of the tasks (Refer to code in this [BackdoorBench github](https://github.com/mtuann/BackdoorBench/blob/master/train_ba.py))
- Modify the aggregation weight by taking the updated of each client model
- In current version, N/ N clients are selected in each round -> (K/ N); K = 4 -> random.select id and then do the training only on K clients
- Split data: (IID) -> (non-IID); please refer to `sample_dirichlet_train_data` function in [BackdoorBench/train_fl.py](https://github.com/mtuann/BackdoorBench/blob/master/train_fl.py) 
    - def sample_uniform_train_data(self, no_participants)
    - def sample_dirichlet_train_data(self, no_participants, alpha=0.5)

- Backdoor attacks: modify data of attack client = 1; (original data, original label) -> (modified data, target label = 2)
- Scale up local model update by T time, $(L^t_i - G^t) * T$
- Training backdoor data and testing with backdoor

