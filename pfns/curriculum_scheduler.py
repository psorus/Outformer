import numpy as np
import torch


class CurriculumScheduler:
    def __init__(self, total_steps, a, b, scheduler_name,max_value=0.95):
        self.total_steps = total_steps
        self.a = a
        self.b = b
        self.scheduler_name = scheduler_name
        self.max_value = max_value
        
    def get_current_value(self, current_step):
        if self.scheduler_name == 'log':
            value = self.g_log(current_step)
        elif self.scheduler_name == 'exp':
            value = self.g_exp(current_step)
        elif self.scheduler_name == 'linear':
            value = self.g_linear(current_step)
        elif self.scheduler_name == 'polynomial':
            value = self.g_quad(current_step)
        elif self.scheduler_name == 'root':
            value = self.g_root(current_step)
        else:
            raise ValueError(f"Unknown scheduler name: {self.scheduler_name}") 
        return min(value, self.max_value)
        
        
    def g_exp(self,t):  # Exponential
        return self.b + ((1-self.b)/(np.exp(10)-1)) * (np.exp(10*t/(self.a*self.total_steps)) - 1)

    def g_log(self,t):  # Logarithmic
        return self.b + (1-self.b)*(1 + 0.1*np.log(t/(self.a*self.total_steps) + np.exp(-10)))

    def g_linear(self,t):  # Linear
        return self.b + (1-self.b)*(t/(self.a * self.total_steps))

    def g_quad(self,t):  # Quadratic
        return self.b + (1-self.b)*(t/(self.a* self.total_steps))**2

    def g_root(self,t):  # Root
        return self.b + (1-self.b)*(t/(self.a* self.total_steps))**0.5


def main():
    total_steps = 1500
    a =0.8
    b =0.2
    import matplotlib.pyplot as plt
    scheduler_names = ['log','exp','linear','polynomial','root'] #, 'exp', 'linear', 'polynomial']
    steps = np.arange(0, total_steps + 1)
    values = {}

    for name in scheduler_names:
        scheduler = CurriculumScheduler(total_steps, a, b, name)
        values[name] = [scheduler.get_current_value(step) for step in steps]

    plt.figure(figsize=(10, 6))
    for name in scheduler_names:
        plt.plot(steps, values[name], label=name)
        
    plt.xlabel('Steps')
    plt.ylabel('Value')
    plt.title('Comparison of Different Schedulers')
    plt.legend()
    plt.grid()
    plt.savefig('scheduler_comparison.png')
    plt.show()
    
if __name__ == "__main__":
    main()