from stable_baselines.results_plotter import plot_results
import matplotlib.pyplot as plt
plot_results(['test_schedules/dqn_breakout/'],int(10e6),'timesteps','Learning Curve')
plt.show()
print('end')
