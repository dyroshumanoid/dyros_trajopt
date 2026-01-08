import numpy as np
import matplotlib.pyplot as plt
import os

def save_and_plot_robot_data(model, xs, us, dt, robot_name):
    xs = np.array(xs)
    us = np.array(us)
    num_knots = us.shape[0]
    time_array = np.linspace(0, dt * num_knots, num_knots)

    # 하체 12개 관절 인덱스 설정
    # q (7~18), v (nv 인덱스 중 6~17), u (0~11)
    q_idx = slice(7, 19)
    v_idx = slice(model.nq + 6, model.nq + 18)
    u_idx = slice(0, 12)

    data_dict = {
        "Position (rad)": xs[:num_knots, q_idx],
        "Velocity (rad/s)": xs[:num_knots, v_idx],
        "Torque (Nm)": us[:, u_idx]
    }

    save_dir = f"./{robot_name}_analysis"
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    np.savez(f"{save_dir}/data.npz", t=time_array, **data_dict)

    for title, data in data_dict.items():
        fig, axs = plt.subplots(4, 3, figsize=(12, 10))
        fig.suptitle(f"{robot_name} - {title}", fontsize=15)
        axs = axs.flatten()
        
        for i in range(12):
            axs[i].plot(time_array, data[:, i], label=f'Joint {i}')
            axs[i].set_title(f"Joint {i}")
            axs[i].grid(True)
            
            if "Torque" in title and (i + 6) < len(model.effortLimit):
                limit = model.effortLimit[i + 6]
                axs[i].axhline(y=limit, color='r', linestyle='--', alpha=0.5)
                axs[i].axhline(y=-limit, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"{save_dir}/{title.split()[0].lower()}_plot.png")

    print(f"[Analysis] Plots generated and saved in {save_dir}")
    plt.show() # 모든 창을 동시에 띄움