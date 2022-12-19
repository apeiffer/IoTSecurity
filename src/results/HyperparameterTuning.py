import random
import numpy as np
import pandas as pd

from sklearn.metrics import recall_score, precision_score, accuracy_score

from IoTgraph import IoTgraph


NOT_MAL = 0
MAL = 1
DEAD_END = 2

def train(env, rho_cp, rho_de, alpha, gamma, epsilon):
  int_nodes = list(range(1, env.dst))
  labels_real = []
  attack_probs = []

  for a in env.devices:
      if a.node == env.src or a.node == env.dst:
          continue
      
      if a.mal:
          labels_real.append(MAL)
      else:
          labels_real.append(NOT_MAL)
              
      attack_probs.append(a.attack_prob)

  training_iterations = 100000
  alpha = alpha
  gamma = gamma
  epsilon = epsilon

  total_changes = 0

  q_table = np.zeros([env.num_states, env.num_actions])

  for state in range(0, len(q_table)):
      possible_actions = list(range(0, len(q_table)))
      
      valid_neighbors = list(env.graph.neighbors(state))
      
      for action in possible_actions:
          action = int(action)
          if action not in valid_neighbors:
              q_table[state][action] = -np.Inf
        
  q_table[:, env.src] = -np.Inf
  q_table[env.dst, :] = -np.Inf

  VISITED = -1
  times_visited = np.zeros([env.num_states, env.num_actions])

  # For plotting metrics
  epoch_batch = []
  penalty_batch = []
  batch_changes = 0

  for i in range(1, training_iterations+1):
      
      # A batch is 1% of training iterations
      # Accumulate the total changes to the Q-table for the entire batch to use for plotting
      pct = (100 * i / training_iterations)
      if pct % 1 == 0:
          epoch_batch.append(pct)
          penalty_batch.append(batch_changes)
          batch_changes = 0
      
      
      # Initialize everything needed for a single iteration
      total_penalty = 0                       # Total penalty for single iteration
      state = env.src                         # Current node
      prev_state = env.src                    # Previous node (useful for penalizing dead ends)
      path = []                               # The path we're taking in this iteration
      visit_arr = np.zeros(env.network_size)  # The nodes we've visited on this path
      done = False                            # Done = reached env.dst or hit a dead end

      
      while not done:
          path.append(state)
          dead_end = True
          visit_arr[state] = VISITED
          
          # The neighbors are the possible valid actions; make sure at least one has not yet been visited
          # Otherwise, the current node is a dead end
          valid_neighbors = list(env.graph.neighbors(state))
          for neighbor in valid_neighbors:
              if visit_arr[neighbor] != VISITED:
                  dead_end = False
                  
          
          # The node is a dead end, so we penalize it and end this iteration
          if dead_end:
              q_table[prev_state, state] = -rho_de
              batch_changes += rho_de
              done = True

              
          # The node is not a dead end, so we select the next node to move to 
          else:  
              
              # Explore the action space by picking the next node randomly
              # Make sure the selected action is valid (check against visit_arr and valid_neighbors)
              if random.uniform(0, 1) < epsilon:
                  action = env.action_space.sample()
                  while action not in valid_neighbors or visit_arr[action] == VISITED:
                      action = env.action_space.sample()
              
              # Exploit learned values by picking the best next node from the current node based on our Q-table
              # Make sure the selected action is valid (check against visit_arr and valid_neighbors)
              else:
                  slc = q_table[state]
                  action = np.argmax(slc)               

                  valid_action_exists = False
                  for neighbor in valid_neighbors:
                      if visit_arr[action] != VISITED:
                          valid_action_exists = True
                  
                  if valid_action_exists:
                      while action not in valid_neighbors or visit_arr[action] == VISITED:
                          slc[action] = -np.Inf # Doing this helps prevent backtracking
                          action = np.argmax(slc)
                  
                  else:
                      q_table[prev_state, state] = -rho_de
                      batch_changes += rho_de
                      done = True
                      
              # We've selected the next node to move to
              next_state = action
              prev_state = state
              state = next_state

              if state == env.dst:
                  path.append(state)
                  done = True
                  
              total_penalty += 1
              
      # The path is corrupted, so we add rho_cp to the total penalty for this iteration
      if env.is_corrupted(path, verbose=False):
          total_penalty += rho_cp
          batch_changes += total_penalty

      # Penalize each intermediate node along the path equally based on the total penalty for the iteration
      # The times_visited array is used to keep the penalties normalized (average penalty)
      for state in path[0:len(path)-2]:
          action = path[path.index(state)+1]
          visits = times_visited[state, action]
          if visits == 0:
              q_table[state, action] -= total_penalty

          else:
              avg_penalty = (q_table[state, action] * visits - total_penalty) / (visits + 1)
              q_table[state, action] = avg_penalty

          times_visited[state, action] += 1
  
  return q_table, int_nodes, labels_real

def classify(env, q_table, int_nodes, labels_real, rho_de):
  q_table_df = pd.DataFrame(q_table)

  labels_pred = []

  for name, data in q_table_df.iteritems():
      if name == env.src or name == env.dst:
          continue

      col_max = max(data.values)
      append_one = False
      
      if col_max > -rho_de:
          labels_pred.append(NOT_MAL)
      
      elif col_max == -rho_de:
          for val in data.values:
              if val > -np.inf and val < -rho_de:
                  append_one = True
                  break
          
          if append_one:
              labels_pred.append(MAL)
          else:
              labels_pred.append(DEAD_END)
          
      else:
          labels_pred.append(MAL)

  X = []
  y_pred = []
  y_hat = []

  for x, y_p, y_h in zip(int_nodes, labels_pred, labels_real):
      if y_p != DEAD_END:
          X.append(x)
          y_pred.append(y_p)
          y_hat.append(y_h)

  acc = accuracy_score(y_hat, y_pred)
  prec = precision_score(y_hat, y_pred)
  rec = recall_score(y_hat, y_pred)

  return acc, prec, rec


def main():
    env = IoTgraph(network_size=50, edge_prob=0.1, percent_mal=0.3, attack_probs=[0.2, 0.8])

    rho_de_multipliers = [1, 2, 5, 10]
    rho_cp_multipliers = [1, 2, 5, 10]
    alphas = [0.01, 0.05, 0.1]
    gammas = [0.5, 0.6, 0.7]
    epsilons = [0.01, 0.05, 0.1, 0.15]

    for rho_de_multiplier in rho_de_multipliers:
      for rho_cp_multiplier in rho_cp_multipliers:
        for alpha in alphas:
          for gamma in gammas:
            for epsilon in epsilons:
              rho_de = rho_de_multiplier * env.network_size
              rho_cp = rho_cp_multiplier * env.network_size

              q_table, int_nodes, labels_real = train(env, rho_cp, rho_de, alpha, gamma, epsilon)
              acc, prec, rec = classify(env, q_table, int_nodes, labels_real, rho_de)

              f = open("hyperparameters.txt", "a")
              print(f'{rho_de}, {rho_cp}, {alpha}, {gamma}, {epsilon}, {acc}, {prec}, {rec}')
              f.write(f'\n{rho_de}, {rho_cp}, {alpha}, {gamma}, {epsilon}, {acc}, {prec}, {rec}')
              f.close()

if __name__ == "__main__":
    main()