import os
import matplotlib.pyplot as plt

def save_plot(fig, filename):
    os.makedirs("reports/plots", exist_ok=True)
    path = f"reports/plots/{filename}"
    fig.savefig(path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    return path

def clear_plots():
    if os.path.exists("reports/plots"):
        for f in os.listdir("reports/plots"):
            os.remove(os.path.join("reports/plots", f))
